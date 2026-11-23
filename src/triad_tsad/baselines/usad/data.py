"""Prepared-CSV adapter preserving the legacy USAD next-point anchor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from triad_tsad.contracts import ColumnRef, DatasetSpec


class USADDataError(ValueError):
    """Raised when prepared data cannot satisfy the USAD contract."""


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """One split after schema resolution and before or after preprocessing."""

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
        raise ImportError("USAD prepared-CSV loading requires pandas") from exc
    return pd


def _column_name(frame: Any, reference: ColumnRef, field_name: str) -> str:
    if isinstance(reference, int):
        try:
            return str(frame.columns[reference])
        except IndexError as exc:
            raise USADDataError(
                f"{field_name} column index is out of range"
            ) from exc
    if reference not in frame.columns:
        raise USADDataError(
            f"{field_name} column {reference!r} is missing"
        )
    return str(reference)


def _timestamp_columns(frame: Any, spec: DatasetSpec) -> tuple[str, ...]:
    references = spec.schema.get("timestamp_fields")
    if references is None:
        references = () if spec.timestamp is None else (spec.timestamp,)
    elif isinstance(references, (str, int)):
        references = (references,)
    try:
        return tuple(
            _column_name(frame, reference, "timestamp")
            for reference in references
        )
    except TypeError as exc:
        raise USADDataError(
            "schema.timestamp_fields must be a sequence"
        ) from exc


def _label_column(
    frame: Any,
    spec: DatasetSpec,
    *,
    required: bool,
) -> str | None:
    reference = (
        spec.label
        if spec.label is not None
        else spec.schema.get("label_field")
    )
    if reference is None:
        return None
    try:
        return _column_name(frame, reference, "label")
    except USADDataError:
        if not required:
            return None
        raise


def _feature_columns(
    frame: Any,
    spec: DatasetSpec,
    *,
    timestamp_columns: tuple[str, ...],
    label_column: str | None,
) -> tuple[str, ...]:
    explicit = spec.schema.get("features")
    if explicit is not None:
        if isinstance(explicit, (str, bytes)):
            raise USADDataError("schema.features must be a sequence")
        columns = tuple(
            _column_name(frame, reference, "feature")
            for reference in explicit
        )
    else:
        excluded = set(timestamp_columns)
        if label_column is not None:
            excluded.add(label_column)
        drop_fields = spec.schema.get("drop_fields", ())
        if isinstance(drop_fields, (str, int)):
            drop_fields = (drop_fields,)
        for reference in drop_fields:
            excluded.add(
                _column_name(frame, reference, "drop")
                if isinstance(reference, int)
                else str(reference)
            )
        columns = tuple(
            str(column)
            for column in frame.columns
            if str(column) not in excluded
        )
    if not columns:
        raise USADDataError("prepared CSV has no feature columns")
    expected = spec.schema.get("expected_feature_count")
    if expected is not None and len(columns) != int(expected):
        raise USADDataError(
            f"expected {int(expected)} features but found {len(columns)}"
        )
    return columns


def _timestamps(
    frame: Any,
    columns: tuple[str, ...],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not columns:
        return None, None
    pd = _pandas()
    if len(columns) == 1:
        text = frame.loc[:, columns[0]].astype(str)
    else:
        text = frame.loc[:, list(columns)].astype(str).agg(" ".join, axis=1)
    display = text.to_numpy(copy=True)
    try:
        parsed = pd.to_datetime(text, errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise USADDataError("timestamp columns cannot be parsed") from exc
    return display, parsed


def _labels(
    frame: Any,
    column: str | None,
    spec: DatasetSpec,
) -> np.ndarray | None:
    if column is None:
        return None
    raw = frame.loc[:, column].to_numpy(copy=True)
    normal_value = spec.schema.get("normal_value", 0)
    anomaly_value = spec.schema.get("anomaly_value", 1)
    normal = raw == normal_value
    anomaly = raw == anomaly_value
    if not np.all(normal | anomaly):
        first = int(np.flatnonzero(~(normal | anomaly))[0])
        raise USADDataError(
            f"label column contains an unsupported value at row {first}"
        )
    return anomaly.astype(np.int8)


def _raw_split(
    path: Path,
    spec: DatasetSpec,
    *,
    required_features: tuple[str, ...] | None,
    require_labels: bool,
) -> PreparedSplit:
    if spec.schema.get("format", "prepared_csv") != "prepared_csv":
        raise USADDataError("USAD requires schema.format='prepared_csv'")
    pd = _pandas()
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise USADDataError(f"failed to read prepared CSV: {path}") from exc
    if bool(spec.schema.get("strip_header_whitespace", True)):
        frame = frame.rename(columns=lambda name: name.strip())
    if len(set(frame.columns)) != len(frame.columns):
        raise USADDataError("prepared CSV contains duplicate column names")

    timestamp_columns = _timestamp_columns(frame, spec)
    label_column = _label_column(frame, spec, required=require_labels)
    feature_names = (
        _feature_columns(
            frame,
            spec,
            timestamp_columns=timestamp_columns,
            label_column=label_column,
        )
        if required_features is None
        else required_features
    )
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise USADDataError(
            f"split is missing feature columns: {missing}"
        )
    try:
        numeric = frame.loc[:, list(feature_names)].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise USADDataError(
            "feature columns must contain only numeric values"
        ) from exc
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    if not np.all(np.isfinite(values)):
        raise USADDataError("feature columns must not contain missing values")
    display, parsed = _timestamps(frame, timestamp_columns)
    return PreparedSplit(
        values=values,
        feature_names=feature_names,
        labels=_labels(frame, label_column, spec),
        timestamps=display,
        parsed_timestamps=parsed,
        source_indices=np.arange(len(frame), dtype=np.int64),
    )


def _replace_values(split: PreparedSplit, values: np.ndarray) -> PreparedSplit:
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
    """Apply train-fitted min/max scaling and split-local EWM smoothing."""

    train = _raw_split(
        spec.train_path,
        spec,
        required_features=None,
        require_labels=False,
    )
    test = _raw_split(
        spec.test_path,
        spec,
        required_features=train.feature_names,
        require_labels=spec.label is not None
        or spec.schema.get("label_field") is not None,
    )
    train_values = train.values
    test_values = test.values
    if normalize:
        minimum = train_values.min(axis=0)
        maximum = train_values.max(axis=0)
        scale = maximum - minimum
        non_constant = scale != 0.0
        scaled_train = np.zeros_like(train_values, dtype=np.float32)
        scaled_test = np.zeros_like(test_values, dtype=np.float32)
        scaled_train[:, non_constant] = (
            train_values[:, non_constant] - minimum[non_constant]
        ) / scale[non_constant]
        scaled_test[:, non_constant] = (
            test_values[:, non_constant] - minimum[non_constant]
        ) / scale[non_constant]
        scaled_test[:, ~non_constant] = (
            test_values[:, ~non_constant] - minimum[~non_constant]
        )
        train_values = scaled_train
        test_values = scaled_test
    if ewm_alpha is not None:
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


def next_point_anchors(
    split: PreparedSplit,
    *,
    window_size: int,
    stride: int = 1,
    expected_period_seconds: int | float | None = None,
) -> np.ndarray:
    """Return first-row-after-window anchors retained by the local loader."""

    if window_size < 1:
        raise USADDataError("window_size must be positive")
    if stride < 1:
        raise USADDataError("stride must be positive")
    final_start = len(split.values) - window_size
    if final_start <= 0:
        raise USADDataError(
            "split must contain at least window_size + 1 rows"
        )
    starts = np.arange(final_start, dtype=np.int64)
    if expected_period_seconds is not None and split.parsed_timestamps is not None:
        nanoseconds = int(float(expected_period_seconds) * 1_000_000_000)
        expected_delta = np.timedelta64(nanoseconds, "ns") * window_size
        start_times = split.parsed_timestamps[starts].astype("datetime64[ns]")
        anchor_times = split.parsed_timestamps[
            starts + window_size
        ].astype("datetime64[ns]")
        starts = starts[anchor_times - start_times == expected_delta]
    starts = starts[::stride]
    if len(starts) == 0:
        raise USADDataError("split contains no valid reconstruction windows")
    return starts + window_size


class ReconstructionWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Windows scored as a unit and aligned to their immediately next row."""

    def __init__(
        self,
        split: PreparedSplit,
        *,
        window_size: int,
        stride: int = 1,
        expected_period_seconds: int | float | None = None,
    ) -> None:
        self.split = split
        self.window_size = int(window_size)
        self.anchors = next_point_anchors(
            split,
            window_size=self.window_size,
            stride=stride,
            expected_period_seconds=expected_period_seconds,
        )

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        anchor = int(self.anchors[index])
        start = anchor - self.window_size
        return {
            "given": torch.as_tensor(
                self.split.values[start:anchor],
                dtype=torch.float32,
            ),
            "anchor": torch.tensor(anchor, dtype=torch.int64),
        }
