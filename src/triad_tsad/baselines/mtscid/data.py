"""Prepared-CSV preprocessing and stride-one windows for MtsCID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from triad_tsad.contracts import ColumnRef, DatasetSpec


class MtsCIDDataError(ValueError):
    """Raised when a dataset cannot satisfy the MtsCID input contract."""


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """A split transformed using MtsCID's native preprocessing policy."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    labels: np.ndarray | None
    timestamps: np.ndarray | None
    parsed_timestamps: np.ndarray | None
    source_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Training and test splits normalized with training statistics."""

    train: PreparedSplit
    test: PreparedSplit
    train_min: np.ndarray
    train_max: np.ndarray


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "MtsCID prepared-CSV loading requires pandas"
        ) from exc
    return pd


def _read_frame(path: Path) -> Any:
    pd = _pandas()
    try:
        frame = pd.read_csv(path).rename(
            columns=lambda value: value.strip()
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise MtsCIDDataError(
            f"failed to read prepared CSV: {path}"
        ) from exc
    if len(set(frame.columns)) != len(frame.columns):
        raise MtsCIDDataError(
            "prepared CSV contains duplicate column names"
        )
    return frame


def _references(
    value: Any,
    field_name: str,
) -> tuple[ColumnRef, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        return (value,)
    try:
        references = tuple(value)
    except TypeError as exc:
        raise MtsCIDDataError(
            f"{field_name} must be a column or sequence of columns"
        ) from exc
    if any(
        isinstance(reference, bool)
        or not isinstance(reference, (str, int))
        for reference in references
    ):
        raise MtsCIDDataError(
            f"{field_name} contains an invalid column reference"
        )
    return references


def _column_name(
    frame: Any,
    reference: ColumnRef,
    kind: str,
) -> Any:
    if isinstance(reference, int):
        try:
            return frame.columns[reference]
        except IndexError as exc:
            raise MtsCIDDataError(
                f"{kind} column index is out of range"
            ) from exc
    if reference not in frame.columns:
        raise MtsCIDDataError(
            f"{kind} column {reference!r} is missing"
        )
    return reference


def _timestamp_columns(
    frame: Any,
    spec: DatasetSpec,
) -> tuple[Any, ...]:
    raw = spec.schema.get("timestamp_fields")
    if raw is None:
        raw = spec.timestamp
    return tuple(
        _column_name(frame, reference, "timestamp")
        for reference in _references(raw, "timestamp_fields")
    )


def _optional_label_column(
    frame: Any,
    spec: DatasetSpec,
) -> Any | None:
    raw = spec.schema.get("label_field", spec.label)
    if raw is None:
        return None
    try:
        return _column_name(frame, raw, "label")
    except MtsCIDDataError:
        return None


def _drop_columns(
    frame: Any,
    spec: DatasetSpec,
) -> tuple[Any, ...]:
    columns: list[Any] = []
    for reference in _references(
        spec.schema.get("drop_fields"), "drop_fields"
    ):
        try:
            columns.append(_column_name(frame, reference, "drop"))
        except MtsCIDDataError:
            continue
    return tuple(columns)


def _training_features(
    frame: Any,
    spec: DatasetSpec,
    *,
    timestamps: tuple[Any, ...],
    label: Any | None,
    dropped: tuple[Any, ...],
) -> tuple[Any, ...]:
    raw = spec.schema.get("features")
    if raw is not None:
        features = tuple(
            _column_name(frame, reference, "feature")
            for reference in _references(raw, "features")
        )
    else:
        excluded = set(timestamps) | set(dropped)
        if label is not None:
            excluded.add(label)
        features = tuple(
            column for column in frame.columns if column not in excluded
        )
    if not features:
        raise MtsCIDDataError("prepared CSV has no feature columns")
    expected_count = spec.schema.get("expected_feature_count")
    if expected_count is not None and len(features) != int(expected_count):
        raise MtsCIDDataError(
            f"expected {int(expected_count)} features but found "
            f"{len(features)}"
        )
    return features


def _filled_numeric_frame(frame: Any, features: tuple[Any, ...]) -> Any:
    pd = _pandas()
    numeric = frame.loc[:, list(features)].apply(
        pd.to_numeric, errors="coerce"
    )
    # This order is retained from the locally integrated MtsCID loader.
    return numeric.interpolate().bfill().ffill().fillna(0.0)


def _timestamp_values(
    frame: Any,
    columns: tuple[Any, ...],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not columns:
        return None, None
    pd = _pandas()
    if len(columns) == 1:
        text = frame.loc[:, columns[0]].astype(str)
    else:
        text = (
            frame.loc[:, list(columns)]
            .astype(str)
            .agg(" ".join, axis=1)
        )
    display = text.to_numpy(copy=True)
    parsed = pd.to_datetime(text, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    return display, parsed


def _mapped_labels(
    frame: Any,
    column: Any | None,
    spec: DatasetSpec,
) -> np.ndarray | None:
    if column is None:
        return None
    pd = _pandas()
    normal_value = spec.schema.get("normal_value", 0)
    anomaly_value = spec.schema.get(
        "anomaly_value", spec.schema.get("attack_value", 1)
    )
    mapped = []
    for value in frame.loc[:, column].to_numpy(copy=True):
        if pd.isna(value) or value == normal_value:
            mapped.append(0)
        elif value == anomaly_value:
            mapped.append(1)
        else:
            mapped.append(0)
    return np.asarray(mapped, dtype=np.int8)


def _normalize(
    frame: Any,
    *,
    train_min: Any,
    train_max: Any,
) -> Any:
    normalized = frame.copy()
    for column in normalized.columns:
        minimum = train_min[column]
        maximum = train_max[column]
        if minimum == maximum:
            normalized[column] = normalized[column] - minimum
        else:
            normalized[column] = (
                normalized[column] - minimum
            ) / (maximum - minimum)
    return normalized


def prepare_dataset(
    spec: DatasetSpec,
    *,
    apply_ewm: bool = True,
    ewm_alpha: float = 0.8,
) -> PreparedDataset:
    """Load prepared CSVs using the migrated MtsCID frame policy.

    Numeric conversion is permissive, missing values use
    ``interpolate -> bfill -> ffill -> zero``, min-max statistics come from
    training only, values are not clipped, and EWM state restarts for test.
    """

    train_frame = _read_frame(spec.train_path)
    test_frame = _read_frame(spec.test_path)
    train_timestamps = _timestamp_columns(train_frame, spec)
    test_timestamps = _timestamp_columns(test_frame, spec)
    train_label = _optional_label_column(train_frame, spec)
    test_label = _optional_label_column(test_frame, spec)
    train_drop = _drop_columns(train_frame, spec)
    features = _training_features(
        train_frame,
        spec,
        timestamps=train_timestamps,
        label=train_label,
        dropped=train_drop,
    )

    test_working = test_frame.copy()
    for feature in features:
        if feature not in test_working.columns:
            test_working[feature] = np.nan

    train_numeric = _filled_numeric_frame(train_frame, features)
    test_numeric = _filled_numeric_frame(test_working, features)
    train_min = train_numeric.min()
    train_max = train_numeric.max()
    train_normalized = _normalize(
        train_numeric,
        train_min=train_min,
        train_max=train_max,
    )
    test_normalized = _normalize(
        test_numeric,
        train_min=train_min,
        train_max=train_max,
    )
    if apply_ewm:
        train_normalized = train_normalized.ewm(
            alpha=ewm_alpha
        ).mean()
        test_normalized = test_normalized.ewm(
            alpha=ewm_alpha
        ).mean()

    train_display, train_parsed = _timestamp_values(
        train_frame, train_timestamps
    )
    test_display, test_parsed = _timestamp_values(
        test_frame, test_timestamps
    )
    feature_names = tuple(str(feature) for feature in features)
    train_values = train_normalized.to_numpy(
        dtype=np.float32, copy=True
    )
    test_values = test_normalized.to_numpy(
        dtype=np.float32, copy=True
    )
    return PreparedDataset(
        train=PreparedSplit(
            values=train_values,
            feature_names=feature_names,
            labels=None,
            timestamps=train_display,
            parsed_timestamps=train_parsed,
            source_indices=np.arange(
                len(train_values), dtype=np.int64
            ),
        ),
        test=PreparedSplit(
            values=test_values,
            feature_names=feature_names,
            labels=_mapped_labels(test_frame, test_label, spec),
            timestamps=test_display,
            parsed_timestamps=test_parsed,
            source_indices=np.arange(
                len(test_values), dtype=np.int64
            ),
        ),
        train_min=train_min.to_numpy(dtype=np.float64, copy=True),
        train_max=train_max.to_numpy(dtype=np.float64, copy=True),
    )


def _window_starts(
    split: PreparedSplit,
    *,
    window_size: int,
    expected_period_seconds: float | None,
    continuity: str,
) -> np.ndarray:
    if len(split.values) < window_size:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(
        len(split.values) - window_size + 1, dtype=np.int64
    )
    if (
        continuity == "none"
        or expected_period_seconds is None
        or split.parsed_timestamps is None
    ):
        return starts
    parsed = split.parsed_timestamps
    if np.isnat(parsed).all():
        return starts
    ends = starts + window_size - 1
    valid_timestamp = ~np.isnat(parsed[starts]) & ~np.isnat(parsed[ends])
    expected = np.timedelta64(
        int(
            round(
                float(expected_period_seconds)
                * (window_size - 1)
                * 1_000_000_000
            )
        ),
        "ns",
    )
    continuous = (
        parsed[ends] - parsed[starts]
    ) == expected
    valid = starts[valid_timestamp & continuous]
    # The integrated loader falls back to all windows when no endpoint pair
    # satisfies the expected duration.
    return starts if len(valid) == 0 else valid


class LastPointWindowDataset(Dataset[dict[str, Any]]):
    """Continuity-checked MtsCID windows anchored at their last points."""

    def __init__(
        self,
        split: PreparedSplit,
        *,
        window_size: int,
        stride: int = 1,
        expected_period_seconds: float | None = None,
        continuity: str = "legacy_endpoint",
    ) -> None:
        if window_size < 2:
            raise MtsCIDDataError("window_size must be at least 2")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise MtsCIDDataError("stride must be a positive integer")
        if continuity not in {"legacy_endpoint", "none"}:
            raise MtsCIDDataError(
                "continuity must be 'legacy_endpoint' or 'none'"
            )
        self.split = split
        self.window_size = int(window_size)
        self.starts = _window_starts(
            split,
            window_size=self.window_size,
            expected_period_seconds=expected_period_seconds,
            continuity=continuity,
        )
        self.starts = self.starts[::stride]
        self.anchors = self.starts + self.window_size - 1

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self.starts[index])
        stop = start + self.window_size
        return {
            "input": torch.as_tensor(
                np.asarray(
                    self.split.values[start:stop],
                    dtype=np.float32,
                )
            ),
            "anchor": int(self.anchors[index]),
        }
