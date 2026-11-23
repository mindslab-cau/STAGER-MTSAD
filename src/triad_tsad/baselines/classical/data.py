"""Legacy-compatible CSV preprocessing for KNN and OCSVM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from triad_tsad.contracts import ColumnRef, DatasetSpec


class ClassicalDataError(ValueError):
    """Raised when a dataset does not satisfy the classical CSV schema."""


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Columns and label values required by the legacy CSV pipeline."""

    timestamp_fields: tuple[ColumnRef, ...]
    label_field: ColumnRef
    drop_fields: tuple[ColumnRef, ...]
    normal_value: Any
    anomaly_value: Any


@dataclass(frozen=True, slots=True)
class PreparedClassicalData:
    """Preprocessed arrays aligned to the original test rows."""

    train_values: np.ndarray
    test_values: np.ndarray
    labels: np.ndarray | None
    timestamps: np.ndarray
    indices: np.ndarray
    feature_names: tuple[str, ...]


_SCHEMA_PRESETS: dict[str, DatasetSchema] = {
    "swat": DatasetSchema(
        timestamp_fields=("timestamp",),
        label_field="label",
        drop_fields=("label",),
        normal_value="Normal",
        anomaly_value="Attack",
    ),
    "wadi": DatasetSchema(
        timestamp_fields=("Date", "Time"),
        label_field="Attack LABLE (1:No Attack, -1:Attack)",
        drop_fields=(
            "Attack LABLE (1:No Attack, -1:Attack)",
            "Row",
        ),
        normal_value=1,
        anomaly_value=-1,
    ),
}


def _logical_family(logical_id: str) -> str | None:
    normalized = "".join(character for character in logical_id.lower() if character.isalnum())
    if normalized.startswith("swat"):
        return "swat"
    if normalized.startswith("wadi"):
        return "wadi"
    return None


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, int)):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = tuple(value)
        if result:
            return result
    raise ClassicalDataError(f"{name} must be a non-empty column sequence")


def resolve_dataset_schema(spec: DatasetSpec) -> DatasetSchema:
    """Resolve explicit schema hints over the SWaT/WADI migration presets."""

    family = _logical_family(spec.logical_id)
    preset = _SCHEMA_PRESETS.get(family) if family is not None else None
    schema = dict(spec.schema)

    timestamp_value = schema.get(
        "timestamp_fields",
        schema.get(
            "timestamp_columns",
            spec.timestamp if spec.timestamp is not None else (
                preset.timestamp_fields if preset is not None else None
            ),
        ),
    )
    label_value = schema.get(
        "label_field",
        schema.get(
            "label_column",
            spec.label if spec.label is not None else (
                preset.label_field if preset is not None else None
            ),
        ),
    )
    drop_value = schema.get(
        "drop_fields",
        schema.get(
            "drop_columns",
            preset.drop_fields if preset is not None else (),
        ),
    )
    normal_value = schema.get(
        "normal_value",
        preset.normal_value if preset is not None else 0,
    )
    anomaly_value = schema.get(
        "anomaly_value",
        schema.get(
            "attack_value",
            preset.anomaly_value if preset is not None else 1,
        ),
    )

    if timestamp_value is None or label_value is None:
        raise ClassicalDataError(
            "unknown datasets must define timestamp_fields and label_field"
        )
    timestamp_fields = _sequence(timestamp_value, "timestamp_fields")
    drop_fields = _sequence(drop_value, "drop_fields") if drop_value else ()
    if not isinstance(label_value, (str, int)) or isinstance(label_value, bool):
        raise ClassicalDataError("label_field must be a string or integer")

    return DatasetSchema(
        timestamp_fields=timestamp_fields,
        label_field=label_value,
        drop_fields=drop_fields,
        normal_value=normal_value,
        anomaly_value=anomaly_value,
    )


def _column_name(frame: Any, reference: ColumnRef, name: str) -> Any:
    columns = frame.columns
    if isinstance(reference, int):
        try:
            return columns[reference]
        except IndexError as exc:
            raise ClassicalDataError(
                f"{name} column position is outside the CSV schema"
            ) from exc
    if reference not in columns:
        raise ClassicalDataError(f"missing {name} column: {reference!r}")
    return reference


def _resolved_schema(frame: Any, schema: DatasetSchema) -> DatasetSchema:
    timestamps = tuple(
        _column_name(frame, reference, "timestamp")
        for reference in schema.timestamp_fields
    )
    label = _column_name(frame, schema.label_field, "label")
    drops = tuple(
        _column_name(frame, reference, "drop")
        for reference in schema.drop_fields
    )
    return DatasetSchema(
        timestamp_fields=timestamps,
        label_field=label,
        drop_fields=drops,
        normal_value=schema.normal_value,
        anomaly_value=schema.anomaly_value,
    )


def _resolved_test_schema(frame: Any, schema: DatasetSchema) -> DatasetSchema:
    timestamps = tuple(
        _column_name(frame, reference, "timestamp")
        for reference in schema.timestamp_fields
    )
    if isinstance(schema.label_field, int):
        label = (
            frame.columns[schema.label_field]
            if -len(frame.columns) <= schema.label_field < len(frame.columns)
            else schema.label_field
        )
    else:
        label = schema.label_field
    return DatasetSchema(
        timestamp_fields=timestamps,
        label_field=label,
        drop_fields=schema.drop_fields,
        normal_value=schema.normal_value,
        anomaly_value=schema.anomaly_value,
    )


def _timestamps(frame: Any, timestamp_fields: Sequence[Any]) -> np.ndarray:
    if len(timestamp_fields) == 1:
        return frame[timestamp_fields[0]].to_numpy(copy=True)
    values = frame[timestamp_fields[0]].astype(str)
    for field in timestamp_fields[1:]:
        values = values + " " + frame[field].astype(str)
    return values.to_numpy(copy=True)


def _legacy_normalize(frame: Any, minimum: Any, maximum: Any) -> Any:
    """Match the original train-stat min-max normalization behavior."""

    import pandas as pd

    normalized = frame.copy()
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    numeric = numeric.astype(np.float32)
    for column in numeric.columns:
        if column not in minimum:
            raise ClassicalDataError(
                f"test column {column!r} had no numeric training statistics"
            )
        if minimum[column] == maximum[column]:
            normalized[column] = numeric[column] - minimum[column]
        else:
            normalized[column] = (
                (numeric[column] - minimum[column])
                / (maximum[column] - minimum[column])
            )
    return normalized


def _feature_columns(frame: Any, schema: DatasetSchema) -> list[Any]:
    excluded = set(schema.timestamp_fields) | set(schema.drop_fields) | {
        schema.label_field
    }
    features = [column for column in frame.columns if column not in excluded]
    if not features:
        raise ClassicalDataError("dataset has no feature columns after schema drops")
    return features


def load_classical_csv(
    spec: DatasetSpec,
    *,
    ewm_alpha: float = 0.8,
) -> PreparedClassicalData:
    """Read, normalize, and smooth one externally supplied dataset.

    The pipeline preserves training-stat min-max scaling followed by pandas
    EWM smoothing with ``alpha=0.8`` and its default adjusted formulation.
    No dataset file is bundled with the repository.
    """

    if not 0.0 < float(ewm_alpha) <= 1.0:
        raise ClassicalDataError("ewm_alpha must be in (0, 1]")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Classical CSV preprocessing requires the optional pandas dependency"
        ) from exc

    try:
        train_frame = pd.read_csv(spec.train_path)
        test_frame = pd.read_csv(spec.test_path)
    except (OSError, UnicodeError) as exc:
        raise ClassicalDataError(
            f"failed to read dataset {spec.logical_id!r}"
        ) from exc

    unresolved = resolve_dataset_schema(spec)
    train_schema = _resolved_schema(train_frame, unresolved)
    test_schema = _resolved_test_schema(test_frame, unresolved)
    feature_columns = _feature_columns(train_frame, train_schema)
    missing_test_features = [
        column for column in feature_columns if column not in test_frame.columns
    ]
    if missing_test_features:
        raise ClassicalDataError(
            "test CSV is missing training features: "
            + ", ".join(map(str, missing_test_features))
        )

    train_features = train_frame[feature_columns]
    numeric_training = train_features.apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna(axis=1)
    numeric_training = numeric_training.astype(np.float32)
    minimum = numeric_training.min()
    maximum = numeric_training.max()

    normalized_train = _legacy_normalize(train_features, minimum, maximum)
    normalized_test = _legacy_normalize(
        test_frame[feature_columns],
        minimum,
        maximum,
    )
    try:
        smoothed_train = normalized_train.ewm(alpha=ewm_alpha).mean()
        smoothed_test = normalized_test.ewm(alpha=ewm_alpha).mean()
        train_values = smoothed_train.to_numpy(dtype=np.float32)
        test_values = smoothed_test.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ClassicalDataError(
            "all retained feature columns must be numeric after CSV conversion"
        ) from exc

    if not np.all(np.isfinite(train_values)) or not np.all(np.isfinite(test_values)):
        raise ClassicalDataError("preprocessed feature arrays must be finite")

    labels: np.ndarray | None
    if test_schema.label_field in test_frame.columns:
        raw_labels = test_frame[test_schema.label_field].to_numpy()
        labels = np.where(
            raw_labels == test_schema.anomaly_value,
            1,
            0,
        ).astype(np.int8)
    else:
        labels = None

    test_timestamps = _timestamps(test_frame, test_schema.timestamp_fields)
    return PreparedClassicalData(
        train_values=train_values,
        test_values=test_values,
        labels=labels,
        timestamps=test_timestamps,
        indices=test_frame.index.to_numpy(copy=True),
        feature_names=tuple(str(column) for column in feature_columns),
    )
