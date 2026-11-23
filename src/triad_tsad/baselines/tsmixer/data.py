"""Prepared-CSV adapter preserving the legacy TSMixer next-point anchor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from triad_tsad.contracts import ColumnRef, DatasetSpec


class TSMixerDataError(ValueError):
    """Raised when a dataset cannot satisfy the TSMixer data contract."""


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """One numeric split before sliding-window construction."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    labels: np.ndarray | None
    timestamps: np.ndarray | None
    parsed_timestamps: np.ndarray | None
    source_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Train and test splits after train-fitted preprocessing."""

    train: PreparedSplit
    test: PreparedSplit


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "TSMixer prepared-CSV loading requires pandas"
        ) from exc
    return pd


def _column_name(frame: Any, reference: ColumnRef, field_name: str) -> str:
    if isinstance(reference, int):
        try:
            return str(frame.columns[reference])
        except IndexError as exc:
            raise TSMixerDataError(
                f"{field_name} column index is out of range"
            ) from exc
    if reference not in frame.columns:
        raise TSMixerDataError(
            f"{field_name} column {reference!r} is missing"
        )
    return str(reference)


def _timestamp_columns(frame: Any, spec: DatasetSpec) -> tuple[str, ...]:
    raw_fields = spec.schema.get("timestamp_fields")
    if raw_fields is None:
        if spec.timestamp is None:
            return ()
        raw_fields = [spec.timestamp]
    if isinstance(raw_fields, (str, int)):
        raw_fields = [raw_fields]
    try:
        return tuple(
            _column_name(frame, field, "timestamp") for field in raw_fields
        )
    except TypeError as exc:
        raise TSMixerDataError(
            "schema.timestamp_fields must be a sequence"
        ) from exc


def _label_column(
    frame: Any,
    spec: DatasetSpec,
    *,
    required: bool,
) -> str | None:
    reference = spec.label
    if reference is None:
        reference = spec.schema.get("label_field")
    if reference is None:
        return None
    try:
        return _column_name(frame, reference, "label")
    except TSMixerDataError:
        if not required:
            return None
        raise


def _read_frame(path: Path, *, strip_headers: bool) -> Any:
    pd = _pandas()
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise TSMixerDataError(f"failed to read prepared CSV: {path}") from exc
    if strip_headers:
        frame = frame.rename(columns=lambda name: name.strip())
    if len(set(frame.columns)) != len(frame.columns):
        raise TSMixerDataError("prepared CSV contains duplicate column names")
    return frame


def _feature_columns(
    frame: Any,
    spec: DatasetSpec,
    timestamp_columns: tuple[str, ...],
    label_column: str | None,
) -> tuple[str, ...]:
    explicit_features = spec.schema.get("features")
    if explicit_features is not None:
        if isinstance(explicit_features, (str, bytes)):
            raise TSMixerDataError("schema.features must be a sequence")
        columns = tuple(
            _column_name(frame, reference, "feature")
            for reference in explicit_features
        )
    else:
        excluded = set(timestamp_columns)
        if label_column is not None:
            excluded.add(label_column)
        raw_drop_fields = spec.schema.get("drop_fields", ())
        if isinstance(raw_drop_fields, (str, int)):
            raw_drop_fields = [raw_drop_fields]
        for reference in raw_drop_fields:
            if isinstance(reference, int):
                excluded.add(_column_name(frame, reference, "drop"))
            else:
                excluded.add(str(reference))
        columns = tuple(
            str(column) for column in frame.columns if column not in excluded
        )

    if not columns:
        raise TSMixerDataError("prepared CSV has no feature columns")
    expected = spec.schema.get("expected_feature_count")
    if expected is not None and len(columns) != int(expected):
        raise TSMixerDataError(
            f"expected {int(expected)} features but found {len(columns)}"
        )
    return columns


def _numeric_values(frame: Any, columns: tuple[str, ...]) -> np.ndarray:
    pd = _pandas()
    try:
        numeric = frame.loc[:, list(columns)].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise TSMixerDataError(
            "feature columns must contain only numeric values"
        ) from exc
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    if not np.all(np.isfinite(values)):
        raise TSMixerDataError("feature columns must not contain missing values")
    return values


def _timestamps(
    frame: Any,
    columns: tuple[str, ...],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not columns:
        return None, None
    pd = _pandas()
    if len(columns) == 1:
        display = frame.loc[:, columns[0]].astype(str).to_numpy(copy=True)
        timestamp_text = frame.loc[:, columns[0]].astype(str)
    else:
        text_columns = frame.loc[:, list(columns)].astype(str)
        timestamp_text = text_columns.agg(" ".join, axis=1)
        display = timestamp_text.to_numpy(copy=True)
    try:
        parsed = pd.to_datetime(timestamp_text, errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise TSMixerDataError("timestamp columns cannot be parsed") from exc
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
        invalid = np.flatnonzero(~(normal | anomaly))
        raise TSMixerDataError(
            f"label column contains unsupported value at row {int(invalid[0])}"
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
        raise TSMixerDataError("TSMixer requires schema.format='prepared_csv'")
    frame = _read_frame(
        path,
        strip_headers=bool(spec.schema.get("strip_header_whitespace", True)),
    )
    timestamp_columns = _timestamp_columns(frame, spec)
    label_column = _label_column(frame, spec, required=require_labels)
    if required_features is None:
        feature_names = _feature_columns(
            frame, spec, timestamp_columns, label_column
        )
    else:
        missing = [name for name in required_features if name not in frame]
        if missing:
            raise TSMixerDataError(
                f"test split is missing feature columns: {missing}"
            )
        feature_names = required_features

    display_timestamps, parsed_timestamps = _timestamps(
        frame, timestamp_columns
    )
    return PreparedSplit(
        values=_numeric_values(frame, feature_names),
        feature_names=feature_names,
        labels=_labels(frame, label_column, spec),
        timestamps=display_timestamps,
        parsed_timestamps=parsed_timestamps,
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
    """Load both prepared CSVs and apply train-fitted preprocessing.

    Feature order follows the training CSV unless ``schema.features`` is
    explicit. Min/max statistics are fit on training values only. Exponential
    smoothing is reset at the start of each split and uses pandas
    ``adjust=True``, matching the legacy adapter.
    """

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
    if test.feature_names != train.feature_names:
        raise TSMixerDataError(
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
    stride: int,
    expected_period_seconds: int | float | None,
) -> np.ndarray:
    """Return source-row anchors for valid next-point windows."""

    if window_size < 1:
        raise TSMixerDataError("window_size must be positive")
    if stride < 1:
        raise TSMixerDataError("stride must be positive")
    final_start = len(split.values) - window_size
    if final_start <= 0:
        raise TSMixerDataError(
            "split must contain at least window_size + 1 rows"
        )
    starts = np.arange(final_start, dtype=np.int64)

    if expected_period_seconds is not None and split.parsed_timestamps is not None:
        expected_delta = np.timedelta64(
            int(float(expected_period_seconds) * 1_000_000_000), "ns"
        )
        start_times = split.parsed_timestamps[starts].astype("datetime64[ns]")
        anchor_times = split.parsed_timestamps[
            starts + window_size
        ].astype("datetime64[ns]")
        continuous = anchor_times - start_times == expected_delta * window_size
        starts = starts[continuous]

    starts = starts[::stride]
    if len(starts) == 0:
        raise TSMixerDataError("split contains no valid next-point windows")
    return starts + window_size


class NextPointDataset(Dataset[dict[str, torch.Tensor]]):
    """Tensor windows whose target is the immediately following source row."""

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
                self.split.values[start:anchor].astype(
                    np.float32, copy=False
                ),
                dtype=torch.float32,
            ),
            "answer": torch.as_tensor(
                self.split.values[anchor].astype(np.float32, copy=False),
                dtype=torch.float32,
            ),
            "anchor": torch.tensor(anchor, dtype=torch.int64),
        }
