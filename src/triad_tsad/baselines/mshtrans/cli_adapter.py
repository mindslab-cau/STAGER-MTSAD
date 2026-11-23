"""DatasetSpec adapter for the retained MSHTrans experiment pipeline.

MSHTrans reconstructs a window and scores its final point.  The shared
``utils.data`` loader is a next-point loader, so requesting an input window of
``W - 1`` plus one prediction point gives the exact same continuity boundary:
both contracts end at raw row ``start + W - 1``.  MSHTrans still receives the
full ``W`` rows from the loader's preprocessed ``tag_values`` array.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from triad_tsad.contracts import DatasetSpec, RunResult

from .config import MSHTransConfig
from .runner import MSHTransRunner


class MSHTransAdapterError(RuntimeError):
    """Raised when a DatasetSpec cannot satisfy the MSHTrans contract."""


@dataclass(frozen=True, slots=True)
class PreparedMSHTransData:
    """Shared-preprocessed rows and continuity-safe MSHTrans window starts."""

    train_values: np.ndarray
    test_values: np.ndarray
    train_valid_starts: np.ndarray
    test_valid_starts: np.ndarray
    test_anchor_indices: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    feature_names: tuple[str, ...]
    expected_period_seconds: int

RunnerFactory = Callable[..., MSHTransRunner]


def _positive_period(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MSHTransAdapterError(
            "schema.expected_period_seconds must be a positive integer"
        )
    period = int(value)
    if period <= 0:
        raise MSHTransAdapterError(
            "schema.expected_period_seconds must be a positive integer"
        )
    return period


def _declared_feature_order(spec: DatasetSpec) -> tuple[str, ...] | None:
    raw = spec.schema.get(
        "feature_order",
        spec.schema.get("feature_names"),
    )
    if raw is None:
        return None
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(
        raw, Sequence
    ):
        raise MSHTransAdapterError(
            "schema.feature_order must be a sequence"
        )
    return tuple(str(value) for value in raw)


def prepare_mshtrans_data(
    spec: DatasetSpec,
    config: MSHTransConfig,
) -> PreparedMSHTransData:
    """Load a DatasetSpec through the shared train-stat preprocessing path.

    The shared loader supplies per-feature train-stat min-max normalization,
    split-local pandas EWM with alpha 0.8, and timestamp endpoint continuity.
    Starts are retained at stride one here.  ``MSHTransRunner`` subsequently
    applies ``train_stride`` (historically 10) and ``test_stride`` (1).
    """

    from triad_tsad.utils.data import load_dataset
    from triad_tsad.utils.dataset_paths import configure_dataset_spec

    model_window = config.model.window_size
    if model_window < 2:  # Defensive; the typed model config is stricter.
        raise MSHTransAdapterError("MSHTrans window_size must be at least 2")

    train_key, test_key = configure_dataset_spec(spec)
    expected_period = _positive_period(
        spec.schema.get("expected_period_seconds", 10)
    )
    common = {
        # A (W-1)-point input and its next point span the exact W rows that
        # MSHTrans reconstructs and anchor at the same final row.
        "window_size": model_window - 1,
        "stride": 1,
        "time_lag": 1,
        "predict_size": 1,
        "normalize_data": True,
        "apply_ewm": True,
        "ewm_alpha": 0.8,
        "downsampled": True,
        "downsample_interval": expected_period,
    }
    train_dataset, _, _, normalization_stats = load_dataset(
        train_key,
        **common,
    )
    if normalization_stats is None:
        raise MSHTransAdapterError(
            "training preprocessing did not return normalization statistics"
        )
    test_dataset, labels, _, _ = load_dataset(
        test_key,
        normalization_stats=normalization_stats,
        **common,
    )

    if labels is None:
        raise MSHTransAdapterError(
            "the selected test dataset did not provide labels"
        )
    if len(train_dataset) < 1 or len(test_dataset) < 1:
        raise MSHTransAdapterError(
            "timestamp continuity filtering left no train or test windows"
        )

    train_names = tuple(str(value) for value in train_dataset.feature_names)
    test_names = tuple(str(value) for value in test_dataset.feature_names)
    if not train_names or train_names != test_names:
        raise MSHTransAdapterError(
            "train and test feature names must be non-empty and identical"
        )
    expected_count = spec.schema.get("expected_feature_count")
    if expected_count is not None:
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, Integral)
            or int(expected_count) <= 0
        ):
            raise MSHTransAdapterError(
                "schema.expected_feature_count must be a positive integer"
            )
        if len(train_names) != int(expected_count):
            raise MSHTransAdapterError(
                "prepared feature count does not match the dataset manifest: "
                f"expected {int(expected_count)}, found {len(train_names)}"
            )
    declared_order = _declared_feature_order(spec)
    if declared_order is not None and train_names != declared_order:
        raise MSHTransAdapterError(
            "prepared feature order does not match the dataset manifest"
        )

    train_starts = np.asarray(
        train_dataset.get_valid_indices(), dtype=np.int64
    )
    test_starts = np.asarray(
        test_dataset.get_valid_indices(), dtype=np.int64
    )
    test_anchors = test_starts + model_window - 1
    loader_targets = np.asarray(
        test_dataset.get_target_indices(), dtype=np.int64
    )
    if not np.array_equal(test_anchors, loader_targets):
        raise MSHTransAdapterError(
            "shared preprocessing and MSHTrans last-point anchors diverged"
        )

    aligned_labels = np.asarray(labels, dtype=np.int8)
    if len(aligned_labels) != len(test_starts):
        raise MSHTransAdapterError(
            "aligned labels do not match continuity-safe test windows"
        )
    raw_timestamps = np.asarray(test_dataset.ts)
    if test_anchors.size and test_anchors[-1] >= len(raw_timestamps):
        raise MSHTransAdapterError(
            "test timestamps are too short for MSHTrans anchors"
        )

    train_values = np.asarray(train_dataset.tag_values, dtype=np.float32)
    test_values = np.asarray(test_dataset.tag_values, dtype=np.float32)
    if train_values.ndim != 2 or test_values.ndim != 2:
        raise MSHTransAdapterError(
            "shared preprocessing must return [time, feature] arrays"
        )
    if (
        train_values.shape[1] != len(train_names)
        or test_values.shape[1] != len(test_names)
    ):
        raise MSHTransAdapterError(
            "shared preprocessing dropped one or more declared features"
        )
    if train_values.shape[1] != test_values.shape[1]:
        raise MSHTransAdapterError(
            "train and test feature counts differ after preprocessing"
        )

    return PreparedMSHTransData(
        train_values=np.ascontiguousarray(train_values),
        test_values=np.ascontiguousarray(test_values),
        train_valid_starts=train_starts,
        test_valid_starts=test_starts,
        test_anchor_indices=test_anchors,
        labels=aligned_labels,
        timestamps=np.asarray(raw_timestamps[test_anchors]),
        feature_names=train_names,
        expected_period_seconds=expected_period,
    )


def run_mshtrans(
    spec: DatasetSpec,
    config: MSHTransConfig,
    *,
    device: str | Any = "auto",
    vendor_root: str | Path | None = None,
    runner_factory: RunnerFactory = MSHTransRunner,
) -> RunResult:
    """Fit and score MSHTrans from one DatasetSpec without writing artifacts."""

    prepared = prepare_mshtrans_data(spec, config)
    runner = runner_factory(
        config,
        device=device,
        vendor_root=vendor_root,
    )
    result = runner.fit_score(
        prepared.train_values,
        prepared.test_values,
        train_valid_starts=prepared.train_valid_starts,
        test_valid_starts=prepared.test_valid_starts,
        labels=None,
        timestamps=None,
        dataset_spec=spec,
    )

    expected_indices = prepared.test_anchor_indices[:: config.test_stride]
    observed_indices = np.asarray(result.indices, dtype=np.int64)
    if not np.array_equal(observed_indices, expected_indices):
        raise MSHTransAdapterError(
            "MSHTrans scores do not match the shared preprocessing anchors"
        )
    selected = slice(None, None, config.test_stride)
    labels = prepared.labels[selected]
    timestamps = prepared.timestamps[selected]
    if len(result.point_scores) != len(labels):
        raise MSHTransAdapterError(
            "MSHTrans score count does not match aligned labels"
        )

    history = getattr(runner, "history", None) or {}
    return replace(
        result,
        labels=labels,
        timestamps=timestamps,
        metadata={
            **dict(result.metadata),
            "feature_names": list(prepared.feature_names),
            "preprocessing": (
                "shared_data_train_minmax_split_ewm_alpha_0.8"
            ),
            "expected_period_seconds": prepared.expected_period_seconds,
            "train_stride": config.train_stride,
            "test_stride": config.test_stride,
            "train_window_count": history.get("train_window_count"),
            "validation_window_count": history.get(
                "validation_window_count"
            ),
            "validation_source": history.get("validation_source"),
        },
    )


__all__ = [
    "MSHTransAdapterError",
    "PreparedMSHTransData",
    "prepare_mshtrans_data",
    "run_mshtrans",
]
