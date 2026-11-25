"""Shared access to SWaT, WADI, and HAI preprocessing.

This module does not implement another preprocessing policy.  It only adapts
``utils.data.load_dataset`` to the portable :class:`DatasetSpec` boundary used
by the standalone baseline launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from stager_mtsad.contracts import DatasetSpec


class BaselineDataError(ValueError):
    """Raised when the retained preprocessing contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class PreparedNextPointData:
    """Two preprocessed splits and their next-point alignment information."""

    train_dataset: Any
    test_dataset: Any
    labels: np.ndarray
    feature_names: tuple[str, ...]
    expected_period_seconds: int

    @property
    def train_values(self) -> np.ndarray:
        return np.asarray(self.train_dataset.tag_values, dtype=np.float32)

    @property
    def test_values(self) -> np.ndarray:
        return np.asarray(self.test_dataset.tag_values, dtype=np.float32)

    @property
    def train_valid_starts(self) -> np.ndarray:
        return self.train_dataset.get_valid_indices()

    @property
    def test_valid_starts(self) -> np.ndarray:
        return self.test_dataset.get_valid_indices()

    @property
    def target_indices(self) -> np.ndarray:
        return self.test_dataset.get_target_indices()

    @property
    def target_timestamps(self) -> np.ndarray:
        return np.asarray(self.test_dataset.ts)[self.target_indices]


def _positive_int(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise BaselineDataError(f"{name} must be a positive integer")
    return int(value)


def load_next_point_data(
    spec: DatasetSpec,
    *,
    window_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
    ewm_alpha: float = 0.8,
) -> PreparedNextPointData:
    """Run the retained train-stat min-max, EWM, and continuity pipeline.

    The returned datasets preserve the historical ``[window] -> next point``
    anchor.  Labels, timestamps, and indices all refer to ``start + window``.
    """

    from stager_mtsad.utils.data import load_dataset
    from stager_mtsad.utils.dataset_paths import configure_dataset_spec

    window_size = _positive_int(window_size, "window_size", 60)
    train_stride = _positive_int(train_stride, "train_stride", 1)
    test_stride = _positive_int(test_stride, "test_stride", 1)
    expected_period = _positive_int(
        spec.schema.get("expected_period_seconds"),
        "schema.expected_period_seconds",
        10,
    )
    train_key, test_key = configure_dataset_spec(spec)
    common = {
        "window_size": window_size,
        "time_lag": 1,
        "predict_size": 1,
        "normalize_data": True,
        "apply_ewm": True,
        "ewm_alpha": float(ewm_alpha),
        "downsampled": True,
        "downsample_interval": expected_period,
    }
    train_dataset, _, _, normalization_stats = load_dataset(
        train_key,
        stride=train_stride,
        **common,
    )
    if normalization_stats is None:
        raise BaselineDataError(
            "training preprocessing did not return normalization statistics"
        )
    test_dataset, labels, _, _ = load_dataset(
        test_key,
        stride=test_stride,
        normalization_stats=normalization_stats,
        **common,
    )
    if labels is None:
        raise BaselineDataError("the test dataset did not provide labels")
    if len(train_dataset) < 1 or len(test_dataset) < 1:
        raise BaselineDataError(
            "timestamp continuity filtering left no train or test windows"
        )

    train_names = tuple(str(name) for name in train_dataset.feature_names)
    test_names = tuple(str(name) for name in test_dataset.feature_names)
    if not train_names or train_names != test_names:
        raise BaselineDataError(
            "train and test feature names must be non-empty and identical"
        )
    expected_count = spec.schema.get("expected_feature_count")
    if expected_count is not None:
        expected_count = _positive_int(
            expected_count,
            "schema.expected_feature_count",
            len(train_names),
        )
        if len(train_names) != expected_count:
            raise BaselineDataError(
                "prepared feature count does not match the dataset manifest: "
                f"expected {expected_count}, found {len(train_names)}"
            )
    declared_order = spec.schema.get(
        "feature_order",
        spec.schema.get("feature_names"),
    )
    if declared_order is not None:
        if isinstance(declared_order, (str, bytes, bytearray)):
            raise BaselineDataError("schema.feature_order must be a sequence")
        try:
            normalized_order = tuple(str(value) for value in declared_order)
        except TypeError as exc:
            raise BaselineDataError(
                "schema.feature_order must be a sequence"
            ) from exc
        if train_names != normalized_order:
            raise BaselineDataError(
                "prepared feature order does not match the dataset manifest"
            )

    labels_array = np.asarray(labels, dtype=np.float32)
    if labels_array.ndim != 1 or len(labels_array) != len(test_dataset):
        raise BaselineDataError(
            "aligned test labels do not match the retained test windows"
        )
    return PreparedNextPointData(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        labels=labels_array,
        feature_names=train_names,
        expected_period_seconds=expected_period,
    )


__all__ = [
    "BaselineDataError",
    "PreparedNextPointData",
    "load_next_point_data",
]
