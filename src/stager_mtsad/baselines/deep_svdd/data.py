"""Prepared-CSV and sliding-window adapter for the Deep SVDD baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from stager_mtsad.contracts import ColumnRef, DatasetSpec


class DeepSVDDDataError(ValueError):
    """Raised when input data cannot satisfy the Deep SVDD contract."""


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """One split after numeric conversion and shared preprocessing."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    labels: np.ndarray | None
    timestamps: np.ndarray | None
    parsed_timestamps: np.ndarray | None
    source_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Train and test splits transformed with training-set statistics."""

    train: PreparedSplit
    test: PreparedSplit


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Deep SVDD prepared-CSV loading requires pandas"
        ) from exc
    return pd


def _read_frame(path: Path, *, strip_headers: bool) -> Any:
    pd = _pandas()
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DeepSVDDDataError(
            f"failed to read prepared CSV: {path}"
        ) from exc
    if strip_headers:
        frame = frame.rename(columns=lambda value: value.strip())
    if len(set(frame.columns)) != len(frame.columns):
        raise DeepSVDDDataError(
            "prepared CSV contains duplicate column names"
        )
    return frame


def _column_name(frame: Any, reference: ColumnRef, kind: str) -> Any:
    if isinstance(reference, int):
        try:
            return frame.columns[reference]
        except IndexError as exc:
            raise DeepSVDDDataError(
                f"{kind} column index is out of range"
            ) from exc
    if reference not in frame.columns:
        raise DeepSVDDDataError(
            f"{kind} column {reference!r} is missing"
        )
    return reference


def _sequence(value: Any, field_name: str) -> tuple[ColumnRef, ...]:
    if isinstance(value, (str, int)):
        return (value,)
    try:
        result = tuple(value)
    except TypeError as exc:
        raise DeepSVDDDataError(
            f"{field_name} must be a column or a sequence of columns"
        ) from exc
    return result


def _timestamp_columns(frame: Any, spec: DatasetSpec) -> tuple[Any, ...]:
    raw = spec.schema.get("timestamp_fields")
    if raw is None:
        raw = () if spec.timestamp is None else (spec.timestamp,)
    return tuple(
        _column_name(frame, reference, "timestamp")
        for reference in _sequence(raw, "timestamp_fields")
    )


def _label_column(
    frame: Any,
    spec: DatasetSpec,
    *,
    required: bool,
) -> Any | None:
    raw = spec.schema.get("label_field", spec.label)
    if raw is None:
        return None
    try:
        return _column_name(frame, raw, "label")
    except DeepSVDDDataError:
        if required:
            raise
        return None


def _drop_columns(frame: Any, spec: DatasetSpec) -> tuple[Any, ...]:
    raw = spec.schema.get("drop_fields", ())
    return tuple(
        _column_name(frame, reference, "drop")
        for reference in _sequence(raw, "drop_fields")
    )


def _feature_columns(
    frame: Any,
    spec: DatasetSpec,
    *,
    timestamp_columns: tuple[Any, ...],
    label_column: Any | None,
    drop_columns: tuple[Any, ...],
) -> tuple[Any, ...]:
    raw_features = spec.schema.get("features")
    if raw_features is not None:
        features = tuple(
            _column_name(frame, reference, "feature")
            for reference in _sequence(raw_features, "features")
        )
    else:
        excluded = set(timestamp_columns) | set(drop_columns)
        if label_column is not None:
            excluded.add(label_column)
        features = tuple(
            column for column in frame.columns if column not in excluded
        )
    if not features:
        raise DeepSVDDDataError(
            "prepared CSV has no feature columns"
        )
    expected_count = spec.schema.get("expected_feature_count")
    if expected_count is not None and len(features) != int(expected_count):
        raise DeepSVDDDataError(
            f"expected {int(expected_count)} features but found "
            f"{len(features)}"
        )
    return features


def _numeric_values(frame: Any, columns: tuple[Any, ...]) -> np.ndarray:
    pd = _pandas()
    try:
        numeric = frame.loc[:, list(columns)].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise DeepSVDDDataError(
            "feature columns must contain only numeric values"
        ) from exc
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    if not np.all(np.isfinite(values)):
        raise DeepSVDDDataError(
            "feature columns must not contain missing or infinite values"
        )
    return values


def _timestamps(
    frame: Any,
    columns: tuple[Any, ...],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not columns:
        return None, None
    pd = _pandas()
    if len(columns) == 1:
        timestamp_text = frame.loc[:, columns[0]].astype(str)
    else:
        timestamp_text = (
            frame.loc[:, list(columns)]
            .astype(str)
            .agg(" ".join, axis=1)
        )
    display = timestamp_text.to_numpy(copy=True)
    try:
        parsed = pd.to_datetime(
            timestamp_text, errors="raise"
        ).to_numpy()
    except (TypeError, ValueError) as exc:
        raise DeepSVDDDataError(
            "timestamp columns cannot be parsed"
        ) from exc
    return display, parsed


def _labels(
    frame: Any,
    column: Any | None,
    spec: DatasetSpec,
) -> np.ndarray | None:
    if column is None:
        return None
    pd = _pandas()
    raw = frame.loc[:, column].to_numpy(copy=True)
    normal_value = spec.schema.get("normal_value", 0)
    anomaly_value = spec.schema.get(
        "anomaly_value", spec.schema.get("attack_value", 1)
    )
    normal = raw == normal_value
    anomaly = raw == anomaly_value
    unknown = ~(normal | anomaly)
    if np.any(unknown):
        policy = spec.schema.get("unknown_label_policy", "normal")
        if policy != "normal":
            first = int(np.flatnonzero(unknown)[0])
            raise DeepSVDDDataError(
                f"unsupported label at row {first}"
            )
        normal = normal | pd.isna(raw) | unknown
    return anomaly.astype(np.int8)


def _raw_split(
    path: Path,
    spec: DatasetSpec,
    *,
    required_features: tuple[Any, ...] | None,
    require_labels: bool,
) -> PreparedSplit:
    if spec.schema.get("format", "prepared_csv") != "prepared_csv":
        raise DeepSVDDDataError(
            "Deep SVDD requires schema.format='prepared_csv'"
        )
    frame = _read_frame(
        path,
        strip_headers=bool(
            spec.schema.get("strip_header_whitespace", True)
        ),
    )
    timestamps = _timestamp_columns(frame, spec)
    label = _label_column(frame, spec, required=require_labels)
    drops = _drop_columns(frame, spec)
    if required_features is None:
        features = _feature_columns(
            frame,
            spec,
            timestamp_columns=timestamps,
            label_column=label,
            drop_columns=drops,
        )
    else:
        missing = [
            column
            for column in required_features
            if column not in frame.columns
        ]
        if missing:
            raise DeepSVDDDataError(
                f"test split is missing feature columns: {missing}"
            )
        features = required_features
    display_timestamps, parsed_timestamps = _timestamps(frame, timestamps)
    return PreparedSplit(
        values=_numeric_values(frame, features),
        feature_names=tuple(str(value) for value in features),
        labels=_labels(frame, label, spec),
        timestamps=display_timestamps,
        parsed_timestamps=parsed_timestamps,
        source_indices=np.arange(len(frame), dtype=np.int64),
    )


def _replace_values(
    split: PreparedSplit,
    values: np.ndarray,
) -> PreparedSplit:
    return PreparedSplit(
        values=np.asarray(values, dtype=np.float32),
        feature_names=split.feature_names,
        labels=split.labels,
        timestamps=split.timestamps,
        parsed_timestamps=split.parsed_timestamps,
        source_indices=split.source_indices,
    )


def prepare_dataset(
    spec: DatasetSpec,
    *,
    normalize: bool = True,
    ewm_alpha: float | None = 0.8,
) -> PreparedDataset:
    """Load prepared CSV files using the historical frame transformations."""

    train = _raw_split(
        spec.train_path,
        spec,
        required_features=None,
        require_labels=False,
    )
    test = _raw_split(
        spec.test_path,
        spec,
        required_features=tuple(train.feature_names),
        require_labels=(
            spec.label is not None
            or spec.schema.get("label_field") is not None
        ),
    )
    if train.feature_names != test.feature_names:
        raise DeepSVDDDataError(
            "train and test feature order must be identical"
        )

    train_values = train.values
    test_values = test.values
    if normalize:
        minimum = train_values.min(axis=0)
        maximum = train_values.max(axis=0)
        scale = maximum - minimum
        non_constant = scale != 0.0
        normalized_train = np.zeros_like(train_values, dtype=np.float32)
        normalized_test = np.zeros_like(test_values, dtype=np.float32)
        normalized_train[:, non_constant] = (
            train_values[:, non_constant] - minimum[non_constant]
        ) / scale[non_constant]
        normalized_test[:, non_constant] = (
            test_values[:, non_constant] - minimum[non_constant]
        ) / scale[non_constant]
        normalized_test[:, ~non_constant] = (
            test_values[:, ~non_constant] - minimum[~non_constant]
        )
        train_values = normalized_train
        test_values = normalized_test

    if ewm_alpha is not None:
        if not 0.0 < float(ewm_alpha) <= 1.0:
            raise DeepSVDDDataError(
                "ewm_alpha must be in (0, 1] or null"
            )
        pd = _pandas()
        train_values = (
            pd.DataFrame(train_values)
            .ewm(alpha=float(ewm_alpha), adjust=True)
            .mean()
            .to_numpy(dtype=np.float32)
        )
        test_values = (
            pd.DataFrame(test_values)
            .ewm(alpha=float(ewm_alpha), adjust=True)
            .mean()
            .to_numpy(dtype=np.float32)
        )

    return PreparedDataset(
        train=_replace_values(train, train_values),
        test=_replace_values(test, test_values),
    )


class LegacyNextWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Deep SVDD windows labeled at the first point after each window."""

    def __init__(
        self,
        split: PreparedSplit,
        *,
        window_size: int,
        stride: int = 1,
        predict_size: int = 1,
        expected_period_seconds: float | None = None,
    ) -> None:
        if window_size < 1:
            raise DeepSVDDDataError("window_size must be positive")
        if stride < 1:
            raise DeepSVDDDataError("stride must be positive")
        if predict_size < 1:
            raise DeepSVDDDataError("predict_size must be positive")
        last_start = len(split.values) - window_size - predict_size
        if last_start < 0:
            raise DeepSVDDDataError(
                "split is too short for one next-anchored window"
            )

        starts = np.arange(last_start + 1, dtype=np.int64)
        anchors = starts + window_size + predict_size - 1
        if expected_period_seconds is not None:
            if split.parsed_timestamps is None:
                raise DeepSVDDDataError(
                    "continuity checking requires timestamp columns"
                )
            elapsed = (
                split.parsed_timestamps[anchors]
                - split.parsed_timestamps[starts]
            ) / np.timedelta64(1, "s")
            expected = (
                window_size + predict_size - 1
            ) * float(expected_period_seconds)
            valid = elapsed == expected
            starts = starts[valid]
            anchors = anchors[valid]

        self.split = split
        self.window_size = int(window_size)
        self.starts = starts[::stride]
        self.anchors = anchors[::stride]
        if len(self.starts) == 0:
            raise DeepSVDDDataError(
                "no windows satisfy the continuity and stride policy"
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        anchor = int(self.anchors[index])
        values = self.split.values[
            start : start + self.window_size
        ]
        label = (
            0
            if self.split.labels is None
            else int(self.split.labels[anchor])
        )
        return {
            "input": torch.as_tensor(
                values.astype(np.float32, copy=False),
                dtype=torch.float32,
            ).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "anchor": torch.tensor(anchor, dtype=torch.long),
        }
