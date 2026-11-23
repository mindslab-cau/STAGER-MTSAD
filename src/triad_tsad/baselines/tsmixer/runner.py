"""Offline TSMixer training and canonical anomaly-score production."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.auto import tqdm

from triad_tsad.contracts import DatasetSpec, RunResult
from triad_tsad.utils import set_random_seed

from .config import TSMixerConfig
from .data import NextPointDataset, PreparedDataset, prepare_dataset
from .model import TSMixer


class TSMixerRunnerError(RuntimeError):
    """Raised when a TSMixer run cannot be completed."""


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Loss history and the restored best epoch."""

    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int
    best_loss: float


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


class TSMixerRunner:
    """Train TSMixer and emit next-point squared-error anomaly scores."""

    def __init__(
        self,
        config: TSMixerConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model: TSMixer | None = None
        self.history: TrainingHistory | None = None
        self.prepared: PreparedDataset | None = None
        self.fitted_spec: DatasetSpec | None = None

    def _datasets(
        self,
        spec: DatasetSpec,
    ) -> tuple[NextPointDataset, NextPointDataset]:
        prepared = prepare_dataset(
            spec,
            normalize=self.config.normalize,
            ewm_alpha=self.config.ewm_alpha,
        )
        feature_count = prepared.train.values.shape[1]
        if (
            self.config.input_dim is not None
            and self.config.input_dim != feature_count
        ):
            raise TSMixerRunnerError(
                f"configuration expects {self.config.input_dim} features "
                f"but dataset provides {feature_count}"
            )
        expected_period = spec.schema.get("expected_period_seconds")
        train_dataset = NextPointDataset(
            prepared.train,
            window_size=self.config.window_size,
            stride=self.config.stride,
            expected_period_seconds=expected_period,
        )
        test_dataset = NextPointDataset(
            prepared.test,
            window_size=self.config.window_size,
            stride=self.config.stride,
            expected_period_seconds=expected_period,
        )
        self.prepared = prepared
        return train_dataset, test_dataset

    def _split_training(
        self,
        dataset: NextPointDataset,
    ) -> tuple[Dataset[Any], Dataset[Any] | None]:
        validation_size = int(
            len(dataset) * self.config.validation_fraction
        )
        if validation_size == 0:
            return dataset, None
        training_size = len(dataset) - validation_size
        if training_size == 0:
            raise TSMixerRunnerError(
                "validation_fraction leaves no training windows"
            )
        generator = torch.Generator().manual_seed(self.config.seed)
        training, validation = random_split(
            dataset,
            [training_size, validation_size],
            generator=generator,
        )
        return training, validation

    def _loss(
        self,
        loader: DataLoader[Any],
        *,
        train: bool,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        if self.model is None:
            raise TSMixerRunnerError("model has not been initialized")
        self.model.train(mode=train)
        total_loss = 0.0
        batch_count = 0
        loss_function = nn.MSELoss()

        for batch in loader:
            given = batch["given"].to(self.device)
            answer = batch["answer"].to(self.device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                prediction = self.model(given)
                loss = loss_function(prediction, answer)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1

        if batch_count == 0:
            raise TSMixerRunnerError("data loader contains no examples")
        return total_loss / batch_count

    def fit(self, spec: DatasetSpec) -> TSMixerRunner:
        """Fit a model using the training split in ``spec``."""

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
            raise TSMixerRunnerError("dataset was not prepared")
        feature_count = self.prepared.train.values.shape[1]
        self.model = TSMixer(
            input_len=self.config.window_size,
            input_dim=feature_count,
            pred_len=self.config.pred_len,
            norm_type=self.config.norm_type,
            activation=self.config.activation,
            n_block=self.config.n_block,
            dropout=self.config.dropout,
            ff_dim=self.config.ff_dim,
            target_slice=None,
        ).to(self.device)
        optimizer_class: type[torch.optim.Optimizer]
        if self.config.optimizer.lower() == "adam":
            optimizer_class = torch.optim.Adam
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            self.model.parameters(), lr=self.config.learning_rate
        )

        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        epoch_iterator = tqdm(
            range(1, self.config.num_epochs + 1),
            total=self.config.num_epochs,
            desc="TSMixer training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            train_loss = self._loss(
                train_loader, train=True, optimizer=optimizer
            )
            train_losses.append(train_loss)
            monitored_loss = train_loss
            if validation_loader is not None:
                monitored_loss = self._loss(
                    validation_loader, train=False
                )
                validation_losses.append(monitored_loss)

            should_stop = False
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_epoch = epoch
                if self.config.best_state_policy == "deep_snapshot":
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    best_state = self.model.state_dict()
                stale_epochs = 0
            else:
                stale_epochs += 1
                should_stop = stale_epochs >= self.config.patience

            postfix = {
                "train": f"{train_loss:.6f}",
                "best": f"{best_loss:.6f}",
                "patience": f"{stale_epochs}/{self.config.patience}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if validation_loader is not None:
                postfix["valid"] = f"{monitored_loss:.6f}"
            epoch_iterator.set_postfix(postfix)
            if should_stop:
                break

        if best_state is None:
            raise TSMixerRunnerError("training did not produce a model state")
        self.model.load_state_dict(best_state)
        self.history = TrainingHistory(
            train_loss=tuple(train_losses),
            validation_loss=tuple(validation_losses),
            best_epoch=best_epoch,
            best_loss=float(best_loss),
        )
        self.fitted_spec = spec
        return self

    def score(self, spec: DatasetSpec) -> RunResult:
        """Score the test split with canonical ``[time, feature]`` errors."""

        if (
            self.model is None
            or self.history is None
            or self.prepared is None
            or self.fitted_spec is None
        ):
            raise TSMixerRunnerError("fit must be called before score")
        if spec != self.fitted_spec:
            raise TSMixerRunnerError(
                "score must use the DatasetSpec fitted by this runner"
            )
        test_dataset = NextPointDataset(
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
                given = batch["given"].to(self.device)
                answer = batch["answer"].to(self.device)
                prediction = self.model(given)
                squared_error = torch.square(prediction - answer)
                feature_batches.append(squared_error.cpu().numpy())
                anchor_batches.append(batch["anchor"].cpu().numpy())
        if not feature_batches:
            raise TSMixerRunnerError("test split contains no scoreable windows")

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
                "baseline": "TSMixer",
                "score_semantics": "higher_is_more_anomalous",
                "feature_score": "squared_prediction_error",
                "evaluation_score_orientation": "[feature, time]",
                "evaluation_score_source": "feature_scores_transposed",
                "point_score_aggregation": (
                    "diagnostic_mean_across_features_only"
                ),
                "feature_score_orientation": "[time, feature]",
                "score_anchor": "first_point_after_input_window",
                "feature_names": list(split.feature_names),
                "best_epoch": self.history.best_epoch,
                "best_loss": self.history.best_loss,
                "best_state_policy": self.config.best_state_policy,
                "evaluation_fpr": self.config.evaluation_fpr,
                "train_loss": list(self.history.train_loss),
                "validation_loss": list(self.history.validation_loss),
            },
            seed=self.config.seed,
            config=self.config.as_dict(),
        )

    def run(self, spec: DatasetSpec) -> RunResult:
        """Fit on ``spec.train_path`` and score ``spec.test_path``."""

        return self.fit(spec).score(spec)


def run_tsmixer(
    spec: DatasetSpec,
    config: TSMixerConfig,
    *,
    device: str | torch.device = "cpu",
) -> RunResult:
    """Run TSMixer offline without writing experiment artifacts."""

    return TSMixerRunner(config, device=device).run(spec)
