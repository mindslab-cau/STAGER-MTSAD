"""Offline, no-write training and scoring adapter for MtsCID."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from stager_mtsad.contracts import DatasetSpec, RunResult
from stager_mtsad.utils import set_random_seed

from .config import MtsCIDConfig
from .data import (
    LastPointWindowDataset,
    PreparedDataset,
    prepare_dataset,
)
from .model import build_model, native_anomaly_energy
from .vendor import EntropyLoss, TransformerVar


class MtsCIDRunnerError(RuntimeError):
    """Raised when the MtsCID adapter cannot complete a run."""


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """In-memory early-stopping record for a single fit."""

    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    selected_epoch: int
    epochs_completed: int
    stopped_early: bool


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    drop_last: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=0,
    )


class MtsCIDRunner:
    """Train the preserved MtsCID model and emit a shared ``RunResult``."""

    def __init__(
        self,
        config: MtsCIDConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise MtsCIDRunnerError(
                "a CUDA device was requested but CUDA is unavailable"
            )
        self.model: TransformerVar | None = None
        self.prepared: PreparedDataset | None = None
        self.feature_count: int | None = None
        self.history: TrainingHistory | None = None
        self.fitted_spec: DatasetSpec | None = None

    def _prepare(self, spec: DatasetSpec) -> PreparedDataset:
        return prepare_dataset(
            spec,
            apply_ewm=self.config.apply_ewm,
            ewm_alpha=self.config.ewm_alpha,
        )

    def _windows(
        self,
        prepared_split: Any,
        spec: DatasetSpec,
        *,
        stride: int,
    ) -> LastPointWindowDataset:
        return LastPointWindowDataset(
            prepared_split,
            window_size=self.config.window_size,
            stride=stride,
            expected_period_seconds=spec.schema.get(
                "expected_period_seconds"
            ),
            continuity=self.config.continuity,
        )

    def _split_training(
        self,
        dataset: LastPointWindowDataset,
    ) -> tuple[Subset[Any], Subset[Any]]:
        dataset_length = len(dataset)
        training_length = int(
            dataset_length
            * (1.0 - self.config.validation_fraction)
        )
        validation_length = dataset_length - training_length
        if training_length <= 0 or validation_length <= 0:
            raise MtsCIDRunnerError(
                "the contiguous validation split requires at least one "
                "training and one validation window"
            )
        indices = list(range(dataset_length))
        return (
            Subset(dataset, indices[:training_length]),
            Subset(dataset, indices[training_length:]),
        )

    def _loss(
        self,
        inputs: torch.Tensor,
        entropy_loss: EntropyLoss,
    ) -> torch.Tensor:
        if self.model is None:
            raise MtsCIDRunnerError("model has not been initialized")
        output = self.model(inputs)
        reconstruction = torch.nn.functional.mse_loss(
            output["out"], inputs, reduction="mean"
        )
        attention = output["attn"]
        attention_term = (
            torch.zeros_like(reconstruction)
            if attention is None
            else entropy_loss(attention) * self.config.alpha
        )
        return reconstruction + attention_term

    def fit(self, spec: DatasetSpec) -> MtsCIDRunner:
        """Fit with contiguous validation and in-memory early stopping."""

        set_random_seed(self.config.seed)
        prepared = self._prepare(spec)
        feature_count = prepared.train.values.shape[1]
        windows = self._windows(prepared.train, spec, stride=self.config.stride)
        training, validation = self._split_training(windows)
        if len(training) < self.config.batch_size:
            raise MtsCIDRunnerError(
                "training windows must be at least batch_size because the "
                "native training loader drops its incomplete final batch"
            )
        training_loader = _loader(
            training,
            batch_size=self.config.batch_size,
            drop_last=True,
        )
        validation_loader = _loader(
            validation,
            batch_size=self.config.batch_size,
            drop_last=False,
        )
        self.model = build_model(
            self.config,
            feature_count=feature_count,
            device=self.device,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        entropy_loss = EntropyLoss()
        best_loss = np.inf
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        stale_epochs = 0
        train_history: list[float] = []
        validation_history: list[float] = []
        stopped_early = False

        epoch_iterator = tqdm(
            range(1, self.config.num_epochs + 1),
            total=self.config.num_epochs,
            desc="MtsCID training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            self.model.train()
            training_total = 0.0
            training_batches = 0
            for batch in training_loader:
                inputs = batch["input"].float().to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(inputs, entropy_loss)
                loss.backward()
                optimizer.step()
                training_total += float(loss.detach().cpu())
                training_batches += 1
            if training_batches == 0:
                raise MtsCIDRunnerError(
                    "training loader produced no complete batches"
                )

            self.model.eval()
            validation_total = 0.0
            validation_batches = 0
            with torch.no_grad():
                for batch in validation_loader:
                    inputs = batch["input"].float().to(self.device)
                    loss = self._loss(inputs, entropy_loss)
                    validation_total += float(loss.cpu())
                    validation_batches += 1
            if validation_batches == 0:
                raise MtsCIDRunnerError(
                    "validation loader produced no batches"
                )
            mean_training = training_total / training_batches
            mean_validation = validation_total / validation_batches
            train_history.append(mean_training)
            validation_history.append(mean_validation)
            should_stop = False
            if mean_validation < best_loss:
                best_loss = mean_validation
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    stopped_early = True
                    should_stop = True

            epoch_iterator.set_postfix(
                train=f"{mean_training:.6f}",
                valid=f"{mean_validation:.6f}",
                best=f"{best_loss:.6f}",
                patience=f"{stale_epochs}/{self.config.patience}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if should_stop:
                break

        if best_state is None:
            raise MtsCIDRunnerError(
                "training did not produce a finite validation checkpoint"
            )
        self.model.load_state_dict(best_state)
        self.prepared = prepared
        self.feature_count = feature_count
        self.history = TrainingHistory(
            training_loss=tuple(train_history),
            validation_loss=tuple(validation_history),
            selected_epoch=best_epoch,
            epochs_completed=len(train_history),
            stopped_early=stopped_early,
        )
        self.fitted_spec = spec
        return self

    def score(self, spec: DatasetSpec) -> RunResult:
        """Score the configured test windows and align to the original rows."""

        if (
            self.model is None
            or self.prepared is None
            or self.feature_count is None
            or self.history is None
            or self.fitted_spec is None
        ):
            raise MtsCIDRunnerError("fit must be called before score")
        if spec != self.fitted_spec:
            raise MtsCIDRunnerError(
                "score must use the DatasetSpec fitted by this runner"
            )
        windows = self._windows(
            self.prepared.test, spec, stride=self.config.test_stride
        )
        if len(windows) == 0:
            raise MtsCIDRunnerError(
                "test split is shorter than window_size"
            )
        loader = _loader(
            windows,
            batch_size=self.config.batch_size,
            drop_last=False,
        )
        point_batches: list[np.ndarray] = []
        anchor_batches: list[np.ndarray] = []
        flatten = self.config.alignment == "flatten"
        offsets = np.arange(1 - self.config.window_size, 1)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input"].float().to(self.device)
                model_output = self.model(inputs, mode="test")
                energy, _ = native_anomaly_energy(
                    model_output,
                    inputs,
                    temperature=self.config.temperature,
                    aggregation=self.config.aggregation,
                )
                anchors = batch["anchor"].detach().cpu().numpy()
                if flatten:
                    point_batches.append(
                        energy.detach().cpu().numpy().reshape(-1)
                    )
                    anchor_batches.append(
                        (anchors[:, None] + offsets).reshape(-1)
                    )
                else:
                    point_batches.append(energy[:, -1].detach().cpu().numpy())
                    anchor_batches.append(anchors)
        points = np.concatenate(point_batches).astype(
            np.float32, copy=False
        )
        anchors = np.concatenate(anchor_batches).astype(
            np.int64, copy=False
        )
        split = self.prepared.test
        labels = None if split.labels is None else split.labels[anchors]
        timestamps = (
            None
            if split.timestamps is None
            else split.timestamps[anchors]
        )
        indices = split.source_indices[anchors]
        return RunResult(
            point_scores=points,
            # The retained MtsCID ``normal_mean`` protocol averages the
            # reconstruction loss across features before multiplying by the
            # memory weight.  Its historical evaluator input is therefore one
            # stream [1, time], not a reconstructed set of feature streams.
            feature_scores=None,
            labels=labels,
            timestamps=timestamps,
            indices=indices,
            metadata={
                "baseline": "mtscid",
                "integration": "external_code_with_adapter",
                "source_license": "MIT",
                "score_anchor": (
                    "all_points_of_each_window"
                    if flatten
                    else "last_point_of_each_window"
                ),
                **(
                    {"test_stride": self.config.test_stride}
                    if self.config.test_stride != 1
                    else {}
                ),
                "score_definition": (
                    "native_reconstruction_memory_harmonic_loss"
                ),
                "score_aggregation": self.config.aggregation,
                "evaluation_score_orientation": "[single_stream, time]",
                "evaluation_score_source": "native_point_scores",
                "point_score_status": "evaluation_input",
                "score_direction": "higher_is_more_anomalous",
                "stride": 1,
                "evaluation_fpr": self.config.evaluation_fpr,
                "continuity": self.config.continuity,
                "checkpoint_storage": "in_memory",
                "learning_rate_schedule": (
                    "constant_matching_effective_source_behavior"
                ),
                "selected_epoch": self.history.selected_epoch,
                "feature_names": list(split.feature_names),
            },
            seed=self.config.seed,
            config=self.config.as_dict(),
        )


def run_mtscid(
    spec: DatasetSpec,
    config: MtsCIDConfig,
    *,
    device: str | torch.device = "cpu",
) -> RunResult:
    """Fit and score MtsCID without network access or artifact writes."""

    runner = MtsCIDRunner(config, device=device)
    runner.fit(spec)
    return runner.score(spec)
