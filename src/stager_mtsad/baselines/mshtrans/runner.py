"""Dataset-independent adapter around the bundled MSHTrans implementation."""

from __future__ import annotations

import copy
import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from stager_mtsad.contracts import DatasetSpec, RunResult
from stager_mtsad.utils import set_random_seed

from .config import MSHTransConfig
from .data import LastPointWindows, make_last_point_windows
from .vendor import (
    MSHTransDependencyError,
    load_external_model_class,
)


def _require_torch():
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise MSHTransDependencyError(
            "MSHTrans execution requires a compatible PyTorch installation"
        ) from exc


def extract_last_point_feature_scores(loss_tensor: Any) -> np.ndarray:
    """Extract `[sample, feature]` scores from native window losses."""

    losses = np.asarray(loss_tensor, dtype=np.float32)
    if losses.ndim == 3:
        return np.ascontiguousarray(losses[:, -1, :])
    if losses.ndim == 2:
        return np.ascontiguousarray(losses)
    raise ValueError(
        "native reconstruction loss must have shape "
        f"[sample, time, feature] or [sample, feature], received {losses.shape}"
    )


class MSHTransRunner:
    """Train and score the bundled MSHTrans source."""

    def __init__(
        self,
        config: MSHTransConfig,
        *,
        device: str | Any = "auto",
        vendor_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.requested_device = device
        self.vendor_root = vendor_root
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model: Any | None = None
        self._num_features: int | None = None
        self._is_fitted = False
        self._history: dict[str, Any] | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("MSHTrans model is not initialized")
        return self._model

    @property
    def device(self) -> Any:
        if self._device is None:
            raise RuntimeError("MSHTrans model is not initialized")
        return self._device

    @property
    def history(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._history)

    def _resolve_device(self, torch: Any) -> Any:
        if str(self.requested_device) == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        device = torch.device(self.requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise MSHTransDependencyError(
                "a CUDA device was requested but CUDA is unavailable"
            )
        return device

    def _legacy_args(self, num_features: int) -> dict[str, Any]:
        model = self.config.model
        training = self.config.training
        return {
            "n_feats": num_features,
            "window_size": model.window_size,
            "seq_len": model.window_size,
            "scale_num": model.scale_count,
            "pool_size_list": list(model.pool_sizes),
            "hyper_num": list(model.hyperedge_counts),
            "head_num": model.head_count,
            "inner_size": model.inner_size,
            "d_model": model.d_model,
            "k": model.topk,
            "update_rate": model.update_rate,
            "lr": training.learning_rate,
            "nb_epoch": training.epochs,
            "patience": training.patience,
            "model_root": "",
            **(
                {"moving_average_windows": list(model.moving_average_windows)}
                if model.moving_average_windows is not None
                else {}
            ),
        }

    def initialize(self, *, num_features: int) -> Any:
        """Load optional dependencies and construct a device-aware model."""

        if isinstance(num_features, bool) or not isinstance(num_features, int):
            raise ValueError("num_features must be a positive integer")
        if num_features <= 0:
            raise ValueError("num_features must be a positive integer")

        torch = _require_torch()
        set_random_seed(self.config.training.seed)
        device = self._resolve_device(torch)
        model_class = load_external_model_class(self.vendor_root)
        model = model_class(
            self._legacy_args(num_features),
            device,
        ).to(device)

        self._torch = torch
        self._device = device
        self._model = model
        self._num_features = num_features
        self._is_fitted = False
        self._history = None
        return model

    def _windows(
        self,
        values: Any,
        *,
        stride: int,
        valid_starts: Any | None = None,
    ) -> LastPointWindows:
        windows = make_last_point_windows(
            values,
            self.config.model.window_size,
            stride=stride,
            valid_starts=valid_starts,
        )
        if (
            self._num_features is not None
            and windows.inputs.shape[-1] != self._num_features
        ):
            raise ValueError(
                "feature count differs from the initialized MSHTrans model"
            )
        return windows

    def _dataset(self, windows: LastPointWindows) -> Any:
        torch = self._torch or _require_torch()
        return torch.utils.data.TensorDataset(
            torch.as_tensor(windows.inputs, dtype=torch.float32)
        )

    def _loader(self, dataset: Any) -> Any:
        torch = self._torch or _require_torch()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.config.training.batch_size, len(dataset)),
            shuffle=False,
            num_workers=self.config.training.num_workers,
            drop_last=False,
        )

    def fit(
        self,
        train_values: Any,
        *,
        train_valid_starts: Any | None = None,
        validation_values: Any | None = None,
        validation_valid_starts: Any | None = None,
    ) -> dict[str, Any]:
        """Fit using the adapted native objective and early-stopping behavior."""

        train_windows = self._windows(
            train_values,
            stride=self.config.train_stride,
            valid_starts=train_valid_starts,
        )
        num_features = int(train_windows.inputs.shape[-1])
        model = self.initialize(num_features=num_features)
        full_training_dataset = self._dataset(train_windows)
        training_dataset = full_training_dataset

        validation_loader = None
        validation_window_count = 0
        validation_source = "disabled"
        if validation_values is not None:
            validation_windows = self._windows(
                validation_values,
                stride=self.config.train_stride,
                valid_starts=validation_valid_starts,
            )
            validation_dataset = self._dataset(validation_windows)
            validation_loader = self._loader(validation_dataset)
            validation_window_count = len(validation_windows.inputs)
            validation_source = "explicit_values"
        elif self.config.training.validation_fraction > 0:
            torch = self._torch or _require_torch()
            total_size = len(full_training_dataset)
            validation_window_count = int(
                total_size * self.config.training.validation_fraction
            )
            if total_size > 1 and validation_window_count > 0:
                training_window_count = (
                    total_size - validation_window_count
                )
                training_dataset, validation_dataset = (
                    torch.utils.data.random_split(
                        full_training_dataset,
                        [training_window_count, validation_window_count],
                        generator=torch.Generator().manual_seed(
                            self.config.training.seed
                        ),
                    )
                )
                validation_loader = self._loader(validation_dataset)
                validation_source = "seeded_random_window_split"
            else:
                validation_window_count = 0
                validation_source = "skipped_too_small"

        train_loader = self._loader(training_dataset)
        args = self._legacy_args(num_features)
        model.train(args, train_loader, valid_loader=validation_loader)
        self._is_fitted = True
        self._history = {
            "trained_epochs": int(
                getattr(model, "trained_epochs", self.config.training.epochs)
            ),
            "total_train_window_count": len(train_windows.inputs),
            "train_window_count": len(training_dataset),
            "validation_window_count": validation_window_count,
            "validation_source": validation_source,
            "validation_split_seed": (
                self.config.training.seed
                if validation_source == "seeded_random_window_split"
                else None
            ),
            "selection": (
                "best_validation_objective"
                if validation_loader is not None
                else "final_epoch"
            ),
        }
        return copy.deepcopy(self._history)

    def score(
        self,
        values: Any,
        *,
        valid_starts: Any | None = None,
        labels: Any | None = None,
        timestamps: Any | None = None,
        dataset_spec: DatasetSpec | None = None,
        require_fitted: bool = True,
    ) -> RunResult:
        """Return native last-point reconstruction MSE in the common contract."""

        if self._model is None:
            raise RuntimeError("MSHTrans model is not initialized")
        if require_fitted and not self._is_fitted:
            raise RuntimeError("MSHTrans model has not been fitted")

        windows = self._windows(
            values,
            stride=self.config.test_stride,
            valid_starts=valid_starts,
        )
        loader = self._loader(self._dataset(windows))
        point_scores, window_losses, _ = self.model.predict_prob(loader)
        point_scores = np.asarray(point_scores, dtype=np.float32).reshape(-1)
        feature_scores = extract_last_point_feature_scores(window_losses)

        if len(point_scores) != len(windows.inputs):
            raise RuntimeError(
                "native point score count does not match generated windows"
            )
        if feature_scores.shape != (
            len(windows.inputs),
            windows.inputs.shape[-1],
        ):
            raise RuntimeError(
                "native feature score shape does not match generated windows"
            )

        native_reduction = feature_scores.mean(axis=1)
        if not np.allclose(point_scores, native_reduction, rtol=1e-5, atol=1e-7):
            raise RuntimeError(
                "native point scores differ from mean last-point feature MSE"
            )

        aligned_labels = (
            windows.align(labels, "labels") if labels is not None else None
        )
        aligned_timestamps = (
            windows.align(timestamps, "timestamps")
            if timestamps is not None
            else None
        )
        metadata = {
            "baseline": "mshtrans",
            "alignment": "last_point",
            "window_count_formula": "N-W+1 before continuity filtering",
            "feature_score": "squared_reconstruction_error",
            "evaluation_score_orientation": "[feature, time]",
            "evaluation_score_source": "feature_scores_transposed",
            "point_score_reduction": "diagnostic_mean_only",
            "score_semantics": "higher_is_more_anomalous",
            "false_positive_rate": self.config.false_positive_rate,
        }
        if dataset_spec is not None:
            metadata["dataset_logical_id"] = dataset_spec.logical_id
        resolved_config = asdict(self.config)
        if self.config.model.moving_average_windows is None:
            resolved_config["model"].pop("moving_average_windows")
        return RunResult(
            point_scores=point_scores,
            feature_scores=feature_scores,
            labels=aligned_labels,
            timestamps=aligned_timestamps,
            indices=windows.anchor_indices,
            metadata=metadata,
            seed=self.config.training.seed,
            config=resolved_config,
        )

    def fit_score(
        self,
        train_values: Any,
        test_values: Any,
        *,
        train_valid_starts: Any | None = None,
        test_valid_starts: Any | None = None,
        validation_values: Any | None = None,
        validation_valid_starts: Any | None = None,
        labels: Any | None = None,
        timestamps: Any | None = None,
        dataset_spec: DatasetSpec | None = None,
    ) -> RunResult:
        """Fit on train windows and score aligned test windows."""

        self.fit(
            train_values,
            train_valid_starts=train_valid_starts,
            validation_values=validation_values,
            validation_valid_starts=validation_valid_starts,
        )
        return self.score(
            test_values,
            valid_starts=test_valid_starts,
            labels=labels,
            timestamps=timestamps,
            dataset_spec=dataset_spec,
        )
