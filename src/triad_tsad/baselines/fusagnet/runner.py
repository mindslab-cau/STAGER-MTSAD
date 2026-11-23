"""Offline FuSAGNet training and scoring adapter."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import (
        BatchSampler, DataLoader, Dataset, RandomSampler, TensorDataset, random_split,
    )
except ImportError as exc:  # pragma: no cover - optional dependency environment
    raise ImportError("FuSAGNetRunner requires PyTorch") from exc
from tqdm.auto import tqdm

from triad_tsad.contracts import RunResult
from triad_tsad.utils import set_random_seed

from .config import FuSAGNetConfig
from .data import NextPointWindows, make_next_point_windows


class _MergeSingletonBatchSampler(BatchSampler):
    """Keep every training sample while avoiding BatchNorm batches of size one."""

    def __iter__(self) -> Iterator[list[int]]:
        pending = None
        for batch in super().__iter__():
            if pending is None:
                pending = batch
            elif len(batch) == 1:
                yield pending + batch
                pending = None
            else:
                yield pending
                pending = batch
        if pending is not None:
            yield pending

    def __len__(self) -> int:
        count = super().__len__()
        return count - int(count > 1 and len(self.sampler) % self.batch_size == 1)


class FuSAGNetRunner:
    """Train and score FuSAGNet without network services or global paths."""

    def __init__(
        self,
        config: FuSAGNetConfig,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        self.config = config
        self.device = self._resolve_device(device)
        self._model: Any | None = None
        self._history: dict[str, list[float] | int | float | str] | None = None
        self._is_fitted = False

    @staticmethod
    def _resolve_device(value: str | torch.device) -> torch.device:
        if str(value) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
        return device

    def _seed_everything(self) -> None:
        set_random_seed(self.config.training.seed)

    def initialize(self) -> Any:
        """Build the model lazily, importing PyTorch Geometric only here."""

        self._seed_everything()
        from .model import FuSAGNet

        self._model = FuSAGNet.from_config(
            self.config.model,
            self.config.layout,
        ).to(self.device)
        self._is_fitted = False
        return self._model

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("FuSAGNet model is not initialized")
        return self._model

    @property
    def history(self) -> dict[str, list[float] | int | float | str] | None:
        return copy.deepcopy(self._history)

    @staticmethod
    def _snapshot_state(
        model: Any,
        *,
        deep: bool,
    ) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        return copy.deepcopy(state) if deep else state.copy()

    def _windows(
        self,
        values: Any,
        feature_order: list[str] | tuple[str, ...],
        valid_starts: Any | None = None,
    ) -> NextPointWindows:
        self.config.layout.validate_observed_order(feature_order)
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("values must have shape [time, feature]")
        if array.shape[1] != self.config.layout.num_features:
            raise ValueError(
                "values feature count does not match the configured feature layout"
            )
        return make_next_point_windows(
            array,
            self.config.model.window_size,
            stride=1,
            valid_starts=valid_starts,
        )

    @staticmethod
    def _dataset(windows: NextPointWindows) -> TensorDataset:
        return TensorDataset(
            torch.tensor(windows.inputs, dtype=torch.float32),
            torch.tensor(windows.targets, dtype=torch.float32),
        )

    @staticmethod
    def _loader(
        dataset: Dataset[Any],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(seed)
        effective_batch_size = min(batch_size, len(dataset))
        if shuffle and effective_batch_size < 2:
            raise ValueError("FuSAGNet training BatchNorm requires batch_size >= 2")
        if shuffle and len(dataset) % effective_batch_size == 1:
            sampler = _MergeSingletonBatchSampler(
                RandomSampler(dataset, generator=generator),
                batch_size=effective_batch_size,
                drop_last=False,
            )
            return DataLoader(dataset, batch_sampler=sampler, generator=generator)
        return DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            drop_last=False,
            generator=generator,
        )

    def fit(
        self,
        train_values: Any,
        *,
        feature_order: list[str] | tuple[str, ...],
        train_valid_starts: Any | None = None,
        validation_values: Any | None = None,
        validation_valid_starts: Any | None = None,
    ) -> dict[str, list[float] | int | float | str]:
        """Fit the model with the configured legacy or deep snapshot policy."""

        self._seed_everything()
        train_windows = self._windows(
            train_values,
            feature_order,
            train_valid_starts,
        )
        if train_windows.inputs.shape[0] < 2:
            raise ValueError("FuSAGNet training requires at least two windows")
        validation_windows = (
            self._windows(
                validation_values,
                feature_order,
                validation_valid_starts,
            )
            if validation_values is not None
            else None
        )

        model = self.initialize()
        training = self.config.training
        full_training_dataset = self._dataset(train_windows)
        training_dataset: Dataset[Any] = full_training_dataset
        validation_dataset: Dataset[Any] | None = None
        if validation_windows is not None:
            validation_dataset = self._dataset(validation_windows)
        elif training.validation_fraction > 0:
            validation_size = int(
                len(full_training_dataset) * training.validation_fraction
            )
            if validation_size > 0:
                training_size = len(full_training_dataset) - validation_size
                if training_size < 1:
                    raise ValueError(
                        "validation_fraction leaves no FuSAGNet training windows"
                    )
                training_dataset, validation_dataset = random_split(
                    full_training_dataset,
                    [training_size, validation_size],
                    generator=torch.Generator().manual_seed(training.seed),
                )
        train_loader = self._loader(
            training_dataset,
            batch_size=training.batch_size,
            shuffle=True,
            seed=training.seed,
        )
        validation_loader = (
            self._loader(
                validation_dataset,
                batch_size=training.batch_size,
                shuffle=False,
                seed=training.seed,
            )
            if validation_dataset is not None
            else None
        )

        optimizer_class = (
            torch.optim.AdamW if training.optimizer == "adamw" else torch.optim.Adam
        )
        optimizer = optimizer_class(
            model.parameters(),
            lr=training.learning_rate,
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        train_losses: list[float] = []
        validation_losses: list[float] = []

        epoch_iterator = tqdm(
            range(training.epochs),
            total=training.epochs,
            desc="FuSAGNet training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            model.train()
            batch_losses: list[float] = []
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                noisy_inputs = inputs + torch.randn_like(inputs) * training.noise_level
                output = model(noisy_inputs)
                total_loss, _, _ = model.compute_loss(
                    output,
                    targets,
                    noisy_inputs,
                )
                total_loss.backward()
                optimizer.step()
                batch_losses.append(float(total_loss.detach().cpu()))

            epoch_train_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_train_loss)
            if validation_loader is None:
                selection_loss = epoch_train_loss
            else:
                selection_loss = self._evaluate_loss(validation_loader)
                validation_losses.append(selection_loss)

            should_stop = False
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state = self._snapshot_state(
                    model,
                    deep=training.deep_snapshot_best_state,
                )
            else:
                stale_epochs += 1
                should_stop = stale_epochs >= training.patience

            postfix = {
                "train": f"{epoch_train_loss:.6f}",
                "best": f"{best_loss:.6f}",
                "patience": f"{stale_epochs}/{training.patience}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if validation_loader is not None:
                postfix["valid"] = f"{selection_loss:.6f}"
            epoch_iterator.set_postfix(postfix)
            if should_stop:
                break

        if best_state is None:
            raise RuntimeError("FuSAGNet training did not produce a model state")
        model.load_state_dict(best_state)
        self._is_fitted = True
        self._history = {
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_state_snapshot": (
                "deep"
                if training.deep_snapshot_best_state
                else "legacy_shallow"
            ),
        }
        return copy.deepcopy(self._history)

    def _evaluate_loss(self, loader: DataLoader) -> float:
        model = self.model
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = model(inputs)
                total_loss, _, _ = model.compute_loss(output, targets, inputs)
                losses.append(float(total_loss.cpu()))
        return float(np.mean(losses))

    def score(
        self,
        values: Any,
        *,
        feature_order: list[str] | tuple[str, ...],
        valid_starts: Any | None = None,
        labels: Any | None = None,
        timestamps: Any | None = None,
        require_fitted: bool = True,
    ) -> RunResult:
        """Return absolute next-point errors aligned to common result semantics."""

        if self._model is None:
            raise RuntimeError("FuSAGNet model is not initialized")
        if require_fitted and not self._is_fitted:
            raise RuntimeError("FuSAGNet model has not been fitted")

        windows = self._windows(values, feature_order, valid_starts)
        loader = self._loader(
            self._dataset(windows),
            batch_size=self.config.training.batch_size,
            shuffle=False,
            seed=self.config.training.seed,
        )
        self.model.eval()
        feature_score_batches: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                forecast = self.model(inputs).forecast
                feature_score_batches.append(
                    torch.abs(forecast - targets).cpu().numpy()
                )

        feature_scores = np.concatenate(feature_score_batches, axis=0)
        point_scores = feature_scores.mean(axis=1)
        aligned_labels = (
            windows.align(labels, "labels") if labels is not None else None
        )
        aligned_timestamps = (
            windows.align(timestamps, "timestamps")
            if timestamps is not None
            else None
        )
        return RunResult(
            point_scores=point_scores,
            feature_scores=feature_scores,
            labels=aligned_labels,
            timestamps=aligned_timestamps,
            indices=windows.target_indices,
            metadata={
                "baseline": "fusagnet",
                "alignment": "next_point",
                "feature_score": "absolute_forecast_error",
                "point_score_reduction": "mean",
                "score_semantics": "higher_is_more_anomalous",
                "false_positive_rate": self.config.false_positive_rate,
                "best_state_snapshot": (
                    "deep"
                    if self.config.training.deep_snapshot_best_state
                    else "legacy_shallow"
                ),
            },
            seed=self.config.training.seed,
            config=asdict(self.config),
        )

    def fit_score(
        self,
        train_values: Any,
        test_values: Any,
        *,
        feature_order: list[str] | tuple[str, ...],
        train_valid_starts: Any | None = None,
        test_valid_starts: Any | None = None,
        validation_values: Any | None = None,
        validation_valid_starts: Any | None = None,
        labels: Any | None = None,
        timestamps: Any | None = None,
    ) -> RunResult:
        """Convenience method for a complete local offline run."""

        self.fit(
            train_values,
            feature_order=feature_order,
            train_valid_starts=train_valid_starts,
            validation_values=validation_values,
            validation_valid_starts=validation_valid_starts,
        )
        return self.score(
            test_values,
            feature_order=feature_order,
            valid_starts=test_valid_starts,
            labels=labels,
            timestamps=timestamps,
        )
