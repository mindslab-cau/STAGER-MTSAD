"""Offline USAD training and project-compatible anomaly scoring."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from triad_tsad.contracts import DatasetSpec, RunResult
from triad_tsad.utils import set_random_seed

from .config import USADConfig
from .data import (
    PreparedDataset,
    ReconstructionWindowDataset,
    prepare_dataset,
)
from .model import USAD


class USADRunnerError(RuntimeError):
    """Raised when an offline USAD run cannot be completed."""


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Losses and the delta-monitor checkpoint semantics of a run."""

    train_loss1: tuple[float, ...]
    train_loss2: tuple[float, ...]
    validation_loss1: tuple[float | None, ...]
    validation_loss2: tuple[float | None, ...]
    selected_epoch: int | None
    effective_state_epoch: int
    epochs_completed: int
    stopped_early: bool
    best_state_policy: str


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _snapshot_state(
    model: USAD,
    policy: str,
) -> dict[str, torch.Tensor]:
    """Implement the historical shallow snapshot or an explicit correction."""

    if policy == "legacy_shallow_delta":
        return dict(model.state_dict())
    if policy == "deep_delta_snapshot":
        return copy.deepcopy(model.state_dict())
    raise USADRunnerError(f"unsupported best-state policy: {policy}")


class USADRunner:
    """Train plain USAD without experiment services or graph dependencies."""

    def __init__(
        self,
        config: USADConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model: USAD | None = None
        self.history: TrainingHistory | None = None
        self.prepared: PreparedDataset | None = None
        self.fitted_spec: DatasetSpec | None = None

    def _datasets(
        self,
        spec: DatasetSpec,
    ) -> tuple[ReconstructionWindowDataset, ReconstructionWindowDataset]:
        prepared = prepare_dataset(
            spec,
            normalize=self.config.normalize,
            ewm_alpha=self.config.ewm_alpha,
        )
        feature_count = prepared.train.values.shape[1]
        if (
            self.config.num_channels is not None
            and self.config.num_channels != feature_count
        ):
            raise USADRunnerError(
                f"configuration expects {self.config.num_channels} features "
                f"but dataset provides {feature_count}"
            )
        expected_period = spec.schema.get("expected_period_seconds")
        train = ReconstructionWindowDataset(
            prepared.train,
            window_size=self.config.window_size,
            stride=self.config.stride,
            expected_period_seconds=expected_period,
        )
        test = ReconstructionWindowDataset(
            prepared.test,
            window_size=self.config.window_size,
            stride=self.config.stride,
            expected_period_seconds=expected_period,
        )
        self.prepared = prepared
        return train, test

    def _split_training(
        self,
        dataset: ReconstructionWindowDataset,
    ) -> tuple[Dataset[Any], Dataset[Any] | None]:
        validation_size = int(
            len(dataset) * self.config.validation_fraction
        )
        if validation_size == 0:
            return dataset, None
        training_size = len(dataset) - validation_size
        if training_size == 0:
            raise USADRunnerError(
                "validation_fraction leaves no training windows"
            )
        generator = torch.Generator().manual_seed(self.config.seed)
        return random_split(
            dataset,
            [training_size, validation_size],
            generator=generator,
        )

    def _training_epoch(
        self,
        loader: DataLoader[Any],
        *,
        epoch: int,
        optimizer1: torch.optim.Optimizer,
        optimizer2: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        if self.model is None:
            raise USADRunnerError("model has not been initialized")
        self.model.train()
        total1 = 0.0
        total2 = 0.0
        batch_count = 0
        for batch in loader:
            inputs = batch["given"].to(self.device)

            optimizer1.zero_grad(set_to_none=True)
            loss1, _ = self.model.losses(
                inputs,
                epoch=epoch,
                warmup_epochs=self.config.warmup_epochs,
            )
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            loss1, loss2 = self.model.losses(
                inputs,
                epoch=epoch,
                warmup_epochs=self.config.warmup_epochs,
            )
            loss2.backward()
            optimizer2.step()

            total1 += float(loss1.detach().cpu())
            total2 += float(loss2.detach().cpu())
            batch_count += 1
        if batch_count == 0:
            raise USADRunnerError("training loader contains no examples")
        return total1 / batch_count, total2 / batch_count

    def _validation_epoch(
        self,
        loader: DataLoader[Any],
        *,
        epoch: int,
        warmup_epochs: int,
    ) -> tuple[float, float]:
        if self.model is None:
            raise USADRunnerError("model has not been initialized")
        self.model.eval()
        total1 = 0.0
        total2 = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["given"].to(self.device)
                loss1, loss2 = self.model.losses(
                    inputs,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                )
                total1 += float(loss1.cpu())
                total2 += float(loss2.cpu())
                batch_count += 1
        if batch_count == 0:
            raise USADRunnerError("validation loader contains no examples")
        return total1 / batch_count, total2 / batch_count

    def fit(self, spec: DatasetSpec) -> USADRunner:
        """Fit on the training split and restore the configured delta snapshot."""

        set_random_seed(self.config.seed)
        train_dataset, _ = self._datasets(spec)
        training, validation = self._split_training(train_dataset)
        train_loader = _loader(
            training,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
        )
        validation_loader = (
            None
            if validation is None
            else _loader(
                validation,
                batch_size=self.config.batch_size,
                shuffle=False,
                seed=self.config.seed,
            )
        )
        if self.prepared is None:
            raise USADRunnerError("dataset was not prepared")
        feature_count = self.prepared.train.values.shape[1]
        self.model = USAD(
            num_channels=feature_count,
            window_size=self.config.window_size,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        optimizer_class: type[torch.optim.Optimizer]
        if self.config.optimizer.lower() == "adam":
            optimizer_class = torch.optim.Adam
        else:
            optimizer_class = torch.optim.AdamW
        optimizer1 = optimizer_class(
            list(self.model.encoder.parameters())
            + list(self.model.decoder1.parameters()),
            lr=self.config.learning_rate,
        )
        optimizer2 = optimizer_class(
            list(self.model.encoder.parameters())
            + list(self.model.decoder2.parameters()),
            lr=self.config.learning_rate,
        )

        train_loss1: list[float] = []
        train_loss2: list[float] = []
        validation_loss1: list[float | None] = []
        validation_loss2: list[float | None] = []
        previous_monitor1: float | None = None
        previous_monitor2: float | None = None
        selected_state: dict[str, torch.Tensor] | None = None
        selected_epoch: int | None = None
        patience_counter = 0
        stopped_early = False
        epochs_completed = 0

        epoch_iterator = tqdm(
            range(1, self.config.num_epochs + 1),
            total=self.config.num_epochs,
            desc="USAD training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            epochs_completed = epoch
            if epoch == self.config.warmup_epochs + 1:
                for group in optimizer2.param_groups:
                    group["lr"] = (
                        self.config.learning_rate
                        * self.config.learning_rate_ratio_after_warmup
                    )
            average1, average2 = self._training_epoch(
                train_loader,
                epoch=epoch,
                optimizer1=optimizer1,
                optimizer2=optimizer2,
            )
            train_loss1.append(average1)
            train_loss2.append(average2)

            if epoch <= self.config.warmup_epochs:
                if validation_loader is not None:
                    validation_loss1.append(None)
                    validation_loss2.append(None)
                epoch_iterator.set_postfix(
                    train1=f"{average1:.6f}",
                    train2=f"{average2:.6f}",
                    phase="warmup",
                    lr1=f"{optimizer1.param_groups[0]['lr']:.2e}",
                    lr2=f"{optimizer2.param_groups[0]['lr']:.2e}",
                )
                continue

            if validation_loader is not None:
                monitor1, monitor2 = self._validation_epoch(
                    validation_loader,
                    epoch=epoch,
                    # The local evaluator silently used its default of 10.
                    # Keep that behavior independent of checkpoint storage so
                    # fixing the shallow snapshot does not change validation.
                    warmup_epochs=self.config.validation_warmup_epochs,
                )
                validation_loss1.append(monitor1)
                validation_loss2.append(monitor2)
            else:
                monitor1, monitor2 = average1, average2

            should_stop = False
            if previous_monitor1 is None or previous_monitor2 is None:
                selected_state = _snapshot_state(
                    self.model, self.config.best_state_policy
                )
                selected_epoch = epoch
                patience_counter = 0
            else:
                delta = abs(monitor1 - previous_monitor1) + abs(
                    monitor2 - previous_monitor2
                )
                if delta < self.config.delta_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    selected_state = _snapshot_state(
                        self.model, self.config.best_state_policy
                    )
                    selected_epoch = epoch
            previous_monitor1 = monitor1
            previous_monitor2 = monitor2

            if patience_counter >= self.config.patience:
                stopped_early = True
                should_stop = True

            postfix = {
                "train1": f"{average1:.6f}",
                "train2": f"{average2:.6f}",
                "patience": f"{patience_counter}/{self.config.patience}",
                "lr1": f"{optimizer1.param_groups[0]['lr']:.2e}",
                "lr2": f"{optimizer2.param_groups[0]['lr']:.2e}",
            }
            if validation_loader is not None:
                postfix["valid1"] = f"{monitor1:.6f}"
                postfix["valid2"] = f"{monitor2:.6f}"
            epoch_iterator.set_postfix(postfix)
            if should_stop:
                break

        if selected_state is not None:
            self.model.load_state_dict(selected_state)
        effective_state_epoch = (
            selected_epoch
            if (
                self.config.best_state_policy == "deep_delta_snapshot"
                and selected_epoch is not None
            )
            else epochs_completed
        )
        self.history = TrainingHistory(
            train_loss1=tuple(train_loss1),
            train_loss2=tuple(train_loss2),
            validation_loss1=tuple(validation_loss1),
            validation_loss2=tuple(validation_loss2),
            selected_epoch=selected_epoch,
            effective_state_epoch=effective_state_epoch,
            epochs_completed=epochs_completed,
            stopped_early=stopped_early,
            best_state_policy=self.config.best_state_policy,
        )
        self.fitted_spec = spec
        return self

    def score(self, spec: DatasetSpec) -> RunResult:
        """Return exact local feature streams plus a diagnostic point vector."""

        if (
            self.model is None
            or self.history is None
            or self.prepared is None
            or self.fitted_spec is None
        ):
            raise USADRunnerError("fit must be called before score")
        if spec != self.fitted_spec:
            raise USADRunnerError(
                "score must use the DatasetSpec fitted by this runner"
            )
        test_dataset = ReconstructionWindowDataset(
            self.prepared.test,
            window_size=self.config.window_size,
            stride=self.config.stride,
            expected_period_seconds=spec.schema.get(
                "expected_period_seconds"
            ),
        )
        test_loader = _loader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            seed=self.config.seed,
        )
        self.model.eval()
        feature_batches: list[np.ndarray] = []
        anchor_batches: list[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["given"].to(self.device)
                scores = self.model.legacy_feature_scores(inputs)
                feature_batches.append(scores.cpu().numpy())
                anchor_batches.append(batch["anchor"].cpu().numpy())
        if not feature_batches:
            raise USADRunnerError("test split contains no scoreable windows")
        feature_scores = np.concatenate(feature_batches, axis=0)
        anchors = np.concatenate(anchor_batches, axis=0).astype(np.int64)
        diagnostic_point_scores = feature_scores.mean(axis=1)
        split = self.prepared.test
        labels = None if split.labels is None else split.labels[anchors]
        timestamps = (
            None if split.timestamps is None else split.timestamps[anchors]
        )
        indices = split.source_indices[anchors]
        return RunResult(
            point_scores=diagnostic_point_scores,
            feature_scores=feature_scores,
            labels=labels,
            timestamps=timestamps,
            indices=indices,
            metadata={
                "baseline": "USAD",
                "variant": "plain",
                "score_semantics": "higher_is_more_anomalous",
                "feature_score": "decoder1_window_mean_squared_error",
                "legacy_feature_score_orientation": "[feature, time]",
                "feature_score_orientation": "[time, feature]",
                "evaluation_score_orientation": "[feature, time]",
                "evaluation_score_source": "feature_scores_transposed",
                "point_score_aggregation": (
                    "diagnostic_mean_across_features_only"
                ),
                "point_score_status": "not_used_by_multistream_fpr_evaluation",
                "score_anchor": "first_point_after_input_window",
                "feature_names": list(split.feature_names),
                "best_state_policy": self.history.best_state_policy,
                "evaluation_fpr": self.config.evaluation_fpr,
                "delta_monitor_selected_epoch": self.history.selected_epoch,
                "effective_state_epoch": self.history.effective_state_epoch,
                "epochs_completed": self.history.epochs_completed,
                "stopped_early": self.history.stopped_early,
                "train_loss1": list(self.history.train_loss1),
                "train_loss2": list(self.history.train_loss2),
                "validation_loss1": list(self.history.validation_loss1),
                "validation_loss2": list(self.history.validation_loss2),
            },
            seed=self.config.seed,
            config=self.config.as_dict(),
        )

    def run(self, spec: DatasetSpec) -> RunResult:
        """Fit and score without writing artifacts."""

        return self.fit(spec).score(spec)


def run_usad(
    spec: DatasetSpec,
    config: USADConfig,
    *,
    device: str | torch.device = "cpu",
) -> RunResult:
    """Run plain USAD offline with no graph or tracking dependencies."""

    return USADRunner(config, device=device).run(spec)
