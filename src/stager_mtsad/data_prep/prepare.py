"""Prepare the SWaT and WADI CSV variants used by the experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SWAT_LABEL = "label"
SWAT_TIMESTAMP = "timestamp"
WADI_LABEL = "Attack LABLE (1:No Attack, -1:Attack)"
WADI_ROW = "Row"
WADI_DATE = "Date"
WADI_TIME = "Time"


class PreparationError(ValueError):
    """Raised when source data cannot satisfy the prepared-data contract."""


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Paths produced by one successful preparation run."""

    dataset: str
    output_dir: Path
    train_path: Path
    test_path: Path
    dataset_manifest_path: Path
    preparation_path: Path


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:
        raise PreparationError(
            "dataset preparation requires pandas; install the 'data' extra"
        ) from exc
    return pd


def _json_write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_source_path(value: str | Path, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise PreparationError(f"{field_name} is not a file: {path}")
    return path


def _validate_parameters(interval_seconds: int, train_skip_rows: int) -> None:
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, int)
        or interval_seconds <= 0
    ):
        raise PreparationError("interval_seconds must be a positive integer")
    if (
        isinstance(train_skip_rows, bool)
        or not isinstance(train_skip_rows, int)
        or train_skip_rows < 0
    ):
        raise PreparationError("train_skip_rows must be a non-negative integer")


def _output_paths(
    output_dir: str | Path,
    sources: Sequence[Path],
    *,
    force: bool,
) -> tuple[Path, dict[str, Path]]:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    targets = {
        "train": destination / "train.csv",
        "test": destination / "test.csv",
        "dataset_manifest": destination / "dataset_manifest.json",
        "preparation": destination / "preparation.json",
    }

    resolved_targets = {path.resolve() for path in targets.values()}
    for source in sources:
        if source.resolve() in resolved_targets:
            raise PreparationError(
                "an input file must not also be one of the output files"
            )

    existing = [path.name for path in targets.values() if path.exists()]
    if existing and not force:
        raise PreparationError(
            "output files already exist; pass force=True to replace them: "
            + ", ".join(sorted(existing))
        )
    return destination, targets


def _normalize_swat_column(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered == "timestamp":
        return SWAT_TIMESTAMP
    if lowered in {
        "normal/attack",
        "normal_attack",
        "normal/ attack",
        "normalattack",
        "label",
    }:
        return SWAT_LABEL
    return stripped


def _normalize_swat_label(value: str) -> str:
    stripped = value.strip()
    compact = stripped.replace(" ", "").lower()
    if compact == "attack":
        return "Attack"
    if compact == "normal":
        return "Normal"
    return stripped


def _swat_header(group_row: list[str], column_row: list[str]) -> list[str]:
    width = max(len(group_row), len(column_row))
    groups = group_row + [""] * (width - len(group_row))
    columns = column_row + [""] * (width - len(column_row))
    merged: list[str] = []
    current_group = ""
    has_groups = any(cell.strip() for cell in groups)

    for index, (raw_group, raw_column) in enumerate(
        zip(groups, columns, strict=True)
    ):
        group = raw_group.strip()
        column = _normalize_swat_column(raw_column)
        if index == 0 or column == SWAT_TIMESTAMP:
            merged.append(column or SWAT_TIMESTAMP)
        elif column == SWAT_LABEL:
            merged.append(column)
        elif not column:
            merged.append("")
        elif has_groups:
            if group:
                current_group = group
            merged.append(
                f"{current_group}_{column}" if current_group else column
            )
        else:
            merged.append(column)
    return merged


def _read_first_csv_row(path: Path) -> list[str]:
    with path.open(
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as stream:
        row = next(csv.reader(stream, skipinitialspace=True), None)
    if row is None:
        raise PreparationError(f"CSV file is empty: {path.name}")
    return row


def _write_swat_intermediate(
    input_path: Path,
    output_path: Path,
    fallback_group_row: list[str],
) -> None:
    with input_path.open(
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as source:
        reader = csv.reader(source, skipinitialspace=True)
        group_row = next(reader, None)
        column_row = next(reader, None)
        if group_row is None or column_row is None:
            raise PreparationError(
                f"SWaT input must contain two header rows: {input_path.name}"
            )
        if not any(cell.strip() for cell in group_row):
            group_row = fallback_group_row
        header = _swat_header(group_row, column_row)
        _validate_unique_header(header, input_path.name)
        try:
            label_index = header.index(SWAT_LABEL)
        except ValueError as exc:
            raise PreparationError(
                f"SWaT label column was not found in {input_path.name}"
            ) from exc

        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(header)
            for row in reader:
                if label_index < len(row):
                    row[label_index] = _normalize_swat_label(row[label_index])
                writer.writerow(row)


def _mode_or_default(values, default):
    modes = values.dropna().mode()
    if len(modes) == 0:
        return default
    return modes.iloc[0]


def _downsample_swat(
    input_path: Path,
    output_path: Path,
    *,
    interval_seconds: int,
    skip_rows: int,
) -> Any:
    pd = _pd()
    frame = pd.read_csv(input_path)
    if skip_rows:
        if len(frame) <= skip_rows:
            raise PreparationError(
                f"SWaT train skip ({skip_rows}) removes every input row"
            )
        frame = frame.iloc[skip_rows:].reset_index(drop=True)

    timestamp_columns = [
        column
        for column in frame.columns
        if str(column).strip().lower() == SWAT_TIMESTAMP
    ]
    if len(timestamp_columns) != 1:
        raise PreparationError(
            "SWaT input must contain exactly one timestamp column"
        )
    if timestamp_columns[0] != SWAT_TIMESTAMP:
        frame = frame.rename(columns={timestamp_columns[0]: SWAT_TIMESTAMP})
    if SWAT_LABEL not in frame.columns:
        raise PreparationError("SWaT input is missing the label column")

    try:
        frame[SWAT_TIMESTAMP] = frame[SWAT_TIMESTAMP].str.strip()
        frame[SWAT_TIMESTAMP] = pd.to_datetime(
            frame[SWAT_TIMESTAMP],
            dayfirst=True,
        )
    except Exception as exc:
        raise PreparationError("failed to parse SWaT timestamps") from exc
    frame = frame.set_index(SWAT_TIMESTAMP).sort_index()

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        raise PreparationError("SWaT input contains no numeric feature columns")
    numeric = frame[numeric_columns].resample(f"{interval_seconds}s").median()
    labels = frame[SWAT_LABEL].resample(f"{interval_seconds}s").agg(
        lambda values: _mode_or_default(values, "Normal")
    )
    parts = [numeric, labels]
    other_columns = [
        column
        for column in frame.columns
        if column not in numeric_columns and column != SWAT_LABEL
    ]
    if other_columns:
        parts.append(
            frame[other_columns].resample(f"{interval_seconds}s").first()
        )
    downsampled = pd.concat(parts, axis=1)
    downsampled = downsampled[
        downsampled[numeric_columns].notna().any(axis=1)
    ].reset_index()

    original_columns = [SWAT_TIMESTAMP, *frame.columns.tolist()]
    downsampled = downsampled[
        [column for column in original_columns if column in downsampled.columns]
    ]
    downsampled.to_csv(output_path, index=False)
    return downsampled


def _is_sequence_header(row: list[str]) -> bool:
    return row == [str(index) for index in range(len(row))]


def _write_wadi_normalized_raw(input_path: Path, output_path: Path) -> None:
    """Strip the optional exported index header and empty trailer rows."""

    with input_path.open(
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as source:
        reader = csv.reader(source)
        first_row = next(reader, None)
        if first_row is None:
            raise PreparationError(f"CSV file is empty: {input_path.name}")
        if _is_sequence_header(first_row):
            header = next(reader, None)
            if header is None:
                raise PreparationError(
                    f"WADI input has no header after its index row: "
                    f"{input_path.name}"
                )
        else:
            header = first_row
        normalized_header = [cell.strip() for cell in header]
        _validate_unique_header(normalized_header, input_path.name)
        required = (WADI_ROW, WADI_DATE, WADI_TIME)
        try:
            required_indices = [
                normalized_header.index(column) for column in required
            ]
        except ValueError as exc:
            raise PreparationError(
                f"WADI input is missing one of {required}: {input_path.name}"
            ) from exc

        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(normalized_header)
            for row in reader:
                if not any(
                    index < len(row) and row[index].strip()
                    for index in required_indices
                ):
                    continue
                writer.writerow(row)


def _wadi_dates(values) -> list[str]:
    formatted: list[str] = []
    for raw_value in values:
        value = str(raw_value)
        parsed = None
        for pattern in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                parsed = datetime.strptime(value, pattern)
                break
            except ValueError:
                continue
        formatted.append(
            parsed.strftime("%m/%d/%Y") if parsed is not None else value
        )
    return formatted


def _wadi_times(dates, values) -> list[str]:
    current_date = None
    hour_counter = 0
    previous_time = None
    formatted: list[str] = []

    for date, raw_value in zip(dates, values, strict=True):
        value = str(raw_value)
        if "." in value:
            value = value.split(".", 1)[0]
        if date != current_date:
            current_date = date
            hour_counter = 0
            previous_time = None

        output = "00:00:00"
        if ":" in value:
            try:
                minutes, seconds = map(int, value.split(":"))
                if previous_time is not None:
                    previous_minutes, previous_seconds = map(
                        int,
                        previous_time.split(":"),
                    )
                    if previous_minutes > minutes or (
                        previous_minutes == 59
                        and previous_seconds == 59
                        and minutes == 0
                        and seconds == 0
                    ):
                        hour_counter += 1
                output = f"{hour_counter:02d}:{minutes:02d}:{seconds:02d}"
                previous_time = value
            except ValueError:
                previous_time = "00:00"
        else:
            previous_time = "00:00"
        formatted.append(output)
    return formatted


def _preprocess_wadi(
    input_path: Path,
    output_path: Path,
    *,
    normal_split: bool,
) -> Any:
    pd = _pd()
    frame = pd.read_csv(input_path)
    if WADI_DATE not in frame.columns or WADI_TIME not in frame.columns:
        raise PreparationError("WADI input must contain Date and Time columns")

    frame[WADI_DATE] = _wadi_dates(frame[WADI_DATE])
    frame[WADI_TIME] = _wadi_times(frame[WADI_DATE], frame[WADI_TIME])
    frame = frame.replace(
        ["nan", "#QNAN", "NaN", "#qnan", "NULL"],
        np.nan,
    )
    empty_columns = frame.columns[frame.isna().all()].tolist()
    frame = frame.drop(columns=empty_columns)

    for column in frame.columns[frame.isna().any()].tolist():
        if column in {WADI_DATE, WADI_TIME}:
            frame[column] = frame[column].ffill().bfill()
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame[column].mean())

    if normal_split:
        frame[WADI_LABEL] = 1
    elif WADI_LABEL not in frame.columns:
        raise PreparationError("WADI attack input is missing its label column")
    else:
        labels = pd.to_numeric(frame[WADI_LABEL], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({-1, 1}):
            raise PreparationError(
                "WADI attack labels must contain only the exact values -1 and 1"
            )
        frame[WADI_LABEL] = labels.astype(int)
    frame.to_csv(output_path, index=False)
    return frame


def _downsample_wadi(
    input_path: Path,
    output_path: Path,
    *,
    interval_seconds: int,
    skip_rows: int,
) -> Any:
    pd = _pd()
    frame = pd.read_csv(input_path)
    if skip_rows:
        if len(frame) <= skip_rows:
            raise PreparationError(
                f"WADI train skip ({skip_rows}) removes every input row"
            )
        frame = frame.iloc[skip_rows:].reset_index(drop=True)
    for column in (WADI_ROW, WADI_DATE, WADI_TIME, WADI_LABEL):
        if column not in frame.columns:
            raise PreparationError(f"WADI input is missing column: {column}")

    try:
        frame["DateTime"] = pd.to_datetime(
            frame[WADI_DATE].astype(str)
            + " "
            + frame[WADI_TIME].astype(str)
        )
    except Exception as exc:
        raise PreparationError("failed to parse WADI timestamps") from exc
    frame = frame.set_index("DateTime")

    excluded = {WADI_ROW, WADI_DATE, WADI_TIME}
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns.tolist()
        if column not in excluded and column != WADI_LABEL
    ]
    if not numeric_columns:
        raise PreparationError("WADI input contains no numeric feature columns")
    numeric = frame[numeric_columns].resample(f"{interval_seconds}s").median()
    labels = frame[WADI_LABEL].resample(f"{interval_seconds}s").agg(
        lambda values: _mode_or_default(values, 1)
    )
    labels = labels.astype(int)
    other_columns = [
        column
        for column in frame.columns
        if column not in numeric_columns
        and column not in excluded
        and column != WADI_LABEL
    ]
    parts = [numeric, labels]
    if other_columns:
        parts.append(
            frame[other_columns].resample(f"{interval_seconds}s").first()
        )
    downsampled = pd.concat(parts, axis=1)
    downsampled = downsampled[
        downsampled[numeric_columns].notna().any(axis=1)
    ].reset_index()
    downsampled[WADI_DATE] = downsampled["DateTime"].dt.strftime("%m/%d/%Y")
    downsampled[WADI_TIME] = downsampled["DateTime"].dt.strftime("%H:%M:%S")
    downsampled[WADI_ROW] = range(1, len(downsampled) + 1)
    downsampled = downsampled.drop(columns="DateTime")

    original_columns = frame.columns.tolist()
    downsampled = downsampled[
        [column for column in original_columns if column in downsampled.columns]
    ]
    downsampled.to_csv(output_path, index=False)
    return downsampled


def _validate_unique_header(header: Sequence[str], source_name: str) -> None:
    empty = [index for index, name in enumerate(header) if not str(name)]
    if empty:
        raise PreparationError(
            f"{source_name} contains empty header fields at indices {empty}"
        )
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise PreparationError(
            f"{source_name} contains duplicate columns: {duplicates}"
        )


def _feature_columns(dataset: str, frame) -> list[str]:
    if dataset == "swat":
        excluded = {SWAT_TIMESTAMP, SWAT_LABEL}
    else:
        excluded = {WADI_ROW, WADI_DATE, WADI_TIME, WADI_LABEL}
    return [column for column in frame.columns if column not in excluded]


def _process_groups(
    dataset: str,
    feature_names: Sequence[str],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, feature in enumerate(feature_names):
        if dataset == "swat":
            prefix = str(feature).split("_", 1)[0]
            process = (
                prefix if prefix in {f"P{i}" for i in range(1, 7)} else None
            )
        elif str(feature).startswith("1_"):
            process = "P1"
        elif str(feature).startswith(("2_", "2A_", "2B_")):
            process = "P2"
        elif str(feature).startswith("3_"):
            process = "P3"
        else:
            process = "P4"
        if process is None:
            raise PreparationError(
                f"cannot derive a process group for feature {feature!r}"
            )
        groups.setdefault(process, []).append(index)
    return groups


def _validate_prepared(
    dataset: str,
    train,
    test,
    *,
    expected_feature_count: int | None,
) -> tuple[list[str], dict[str, list[int]]]:
    if expected_feature_count is not None and (
        isinstance(expected_feature_count, bool)
        or not isinstance(expected_feature_count, int)
        or expected_feature_count <= 0
    ):
        raise PreparationError(
            "expected_feature_count must be a positive integer or None"
        )
    if train.empty:
        raise PreparationError(f"{dataset} training split must not be empty")
    if test.empty:
        raise PreparationError(f"{dataset} test split must not be empty")

    train_features = _feature_columns(dataset, train)
    test_features = _feature_columns(dataset, test)
    if train_features != test_features:
        raise PreparationError("train and test feature order does not match")
    if (
        expected_feature_count is not None
        and len(train_features) != int(expected_feature_count)
    ):
        raise PreparationError(
            f"{dataset} must contain {expected_feature_count} features; "
            f"found {len(train_features)}"
        )
    if len(train_features) != len(set(train_features)):
        raise PreparationError("prepared feature names must be unique")

    arrays = (
        train[train_features].to_numpy(),
        test[test_features].to_numpy(),
    )
    try:
        contains_non_finite = any(
            not np.isfinite(array).all() for array in arrays
        )
    except TypeError as exc:
        raise PreparationError(
            "prepared feature values must all be numeric"
        ) from exc
    if contains_non_finite:
        raise PreparationError("prepared feature values must be finite")

    if dataset == "swat":
        if train[SWAT_LABEL].isna().any() or test[SWAT_LABEL].isna().any():
            raise PreparationError("SWaT labels must not be missing")
        if set(train[SWAT_LABEL].unique()) != {"Normal"}:
            raise PreparationError("SWaT training labels must all be Normal")
        test_labels = set(test[SWAT_LABEL].unique())
        if not test_labels.issubset({"Normal", "Attack"}):
            raise PreparationError("SWaT test labels must be Normal or Attack")
        if "Attack" not in test_labels:
            raise PreparationError(
                "SWaT test split must contain at least one Attack label"
            )
        if (
            train[SWAT_TIMESTAMP].isna().any()
            or test[SWAT_TIMESTAMP].isna().any()
        ):
            raise PreparationError("SWaT timestamps must not be missing")
    else:
        if train[WADI_LABEL].isna().any() or test[WADI_LABEL].isna().any():
            raise PreparationError("WADI labels must not be missing")
        if set(train[WADI_LABEL].astype(int).unique()) != {1}:
            raise PreparationError("WADI training labels must all equal 1")
        test_labels = set(test[WADI_LABEL].astype(int).unique())
        if not test_labels.issubset({-1, 1}):
            raise PreparationError("WADI test labels must equal 1 or -1")
        if -1 not in test_labels:
            raise PreparationError(
                "WADI test split must contain at least one -1 attack label"
            )
        for frame in (train, test):
            if frame[[WADI_DATE, WADI_TIME]].isna().any().any():
                raise PreparationError("WADI timestamps must not be missing")

    groups = _process_groups(dataset, train_features)
    if dataset == "swat" and expected_feature_count == 51:
        expected = {
            "P1": list(range(0, 5)),
            "P2": list(range(5, 16)),
            "P3": list(range(16, 25)),
            "P4": list(range(25, 34)),
            "P5": list(range(34, 47)),
            "P6": list(range(47, 51)),
        }
        if groups != expected:
            raise PreparationError(
                "SWaT process groups do not match the 51-feature contract"
            )
    elif dataset == "wadi" and expected_feature_count == 123:
        expected = {
            "P1": list(range(0, 19)),
            "P2": list(range(19, 105)),
            "P3": list(range(105, 120)),
            "P4": list(range(120, 123)),
        }
        if groups != expected:
            raise PreparationError(
                "WADI process groups do not match the 123-feature contract"
            )
    return train_features, groups


def _label_counts(values) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in values.value_counts(dropna=False).sort_index().items()
    }


def _output_summary(path: Path, frame, label_column: str) -> dict[str, Any]:
    return {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "label_counts": _label_counts(frame[label_column]),
    }


def _source_summary(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "size_bytes": int(path.stat().st_size),
    }


def _write_documents(
    *,
    dataset: str,
    interval_seconds: int,
    train_skip_rows: int,
    normal_source: Path,
    attack_source: Path,
    train_path: Path,
    test_path: Path,
    train,
    test,
    feature_names: list[str],
    process_groups: dict[str, list[int]],
    dataset_manifest_path: Path,
    preparation_path: Path,
) -> None:
    runtime_schema = {
        "strip_header_whitespace": True,
        "numeric_policy": "strict",
        "missing_value_policy": "error",
        "scaler": "train_minmax",
        "constant_policy": "zero",
        "clip_test": False,
        "ewm": {
            "enabled": True,
            "alpha": 0.8,
            "adjust": True,
            "reset_each_split": True,
        },
    }
    if dataset == "swat":
        dataset_entry: dict[str, Any] = {
            "train_path": train_path.name,
            "test_path": test_path.name,
            "timestamp": SWAT_TIMESTAMP,
            "label": SWAT_LABEL,
            "schema": {
                "format": "prepared_csv",
                "timestamp_fields": [SWAT_TIMESTAMP],
                "label_field": SWAT_LABEL,
                "normal_value": "Normal",
                "anomaly_value": "Attack",
                "expected_period_seconds": interval_seconds,
                "expected_feature_count": len(feature_names),
                "feature_names": feature_names,
                "feature_order": feature_names,
                "feature_permutation": list(range(len(feature_names))),
                "process_groups": process_groups,
                "drop_fields": [SWAT_LABEL],
                **runtime_schema,
            },
        }
        protocol = f"swat_{interval_seconds}s_median_mode_v1"
        label_column = SWAT_LABEL
    else:
        dataset_entry = {
            "train_path": train_path.name,
            "test_path": test_path.name,
            "label": WADI_LABEL,
            "schema": {
                "format": "prepared_csv",
                "timestamp_fields": [WADI_DATE, WADI_TIME],
                "label_field": WADI_LABEL,
                "normal_value": 1,
                "anomaly_value": -1,
                "expected_period_seconds": interval_seconds,
                "expected_feature_count": len(feature_names),
                "feature_names": feature_names,
                "feature_order": feature_names,
                "feature_permutation": list(range(len(feature_names))),
                "process_groups": process_groups,
                "drop_fields": [WADI_LABEL, WADI_ROW],
                **runtime_schema,
            },
        }
        protocol = f"wadi_{interval_seconds}s_median_mode_v1"
        label_column = WADI_LABEL

    _json_write(
        dataset_manifest_path,
        {"datasets": {dataset: dataset_entry}},
    )
    pd = _pd()
    _json_write(
        preparation_path,
        {
            "dataset": dataset,
            "protocol": protocol,
            "parameters": {
                "interval_seconds": interval_seconds,
                "label_aggregation": "mode",
                "numeric_aggregation": "median",
                "train_skip_rows": train_skip_rows,
                "train_skip_stage": "before_resampling",
                "runtime_scaling_stage": "experiment_reader",
                "feature_order_policy": "preserve_source_order",
            },
            "runtime": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "sources": {
                "normal": _source_summary(normal_source),
                "attack": _source_summary(attack_source),
            },
            "outputs": {
                "train": _output_summary(train_path, train, label_column),
                "test": _output_summary(test_path, test, label_column),
            },
            "feature_names": feature_names,
            "feature_order": feature_names,
            "feature_permutation": list(range(len(feature_names))),
            "process_groups": process_groups,
        },
    )


def _publish_files(
    temporary: dict[str, Path],
    targets: dict[str, Path],
) -> None:
    for key in ("train", "test", "dataset_manifest", "preparation"):
        os.replace(temporary[key], targets[key])


def prepare_swat(
    normal_raw: str | Path,
    attack_raw: str | Path,
    output_dir: str | Path,
    *,
    interval_seconds: int = 10,
    train_skip_rows: int = 21_600,
    expected_feature_count: int | None = 51,
    force: bool = False,
) -> PreparationResult:
    """Prepare the process-prefixed, 10-second SWaT train and test CSVs."""

    _validate_parameters(interval_seconds, train_skip_rows)
    normal_source = _normalize_source_path(normal_raw, "normal_raw")
    attack_source = _normalize_source_path(attack_raw, "attack_raw")
    destination, targets = _output_paths(
        output_dir,
        (normal_source, attack_source),
        force=force,
    )
    fallback_group_row = _read_first_csv_row(normal_source)

    with tempfile.TemporaryDirectory(
        prefix=".stager-prepare-",
        dir=destination,
    ) as temporary_directory:
        root = Path(temporary_directory)
        normal_intermediate = root / "normal_preprocessed.csv"
        attack_intermediate = root / "attack_preprocessed.csv"
        temporary = {
            "train": root / "train.csv",
            "test": root / "test.csv",
            "dataset_manifest": root / "dataset_manifest.json",
            "preparation": root / "preparation.json",
        }
        _write_swat_intermediate(
            normal_source,
            normal_intermediate,
            fallback_group_row,
        )
        _write_swat_intermediate(
            attack_source,
            attack_intermediate,
            fallback_group_row,
        )
        train = _downsample_swat(
            normal_intermediate,
            temporary["train"],
            interval_seconds=interval_seconds,
            skip_rows=train_skip_rows,
        )
        test = _downsample_swat(
            attack_intermediate,
            temporary["test"],
            interval_seconds=interval_seconds,
            skip_rows=0,
        )
        feature_names, process_groups = _validate_prepared(
            "swat",
            train,
            test,
            expected_feature_count=expected_feature_count,
        )
        _write_documents(
            dataset="swat",
            interval_seconds=interval_seconds,
            train_skip_rows=train_skip_rows,
            normal_source=normal_source,
            attack_source=attack_source,
            train_path=temporary["train"],
            test_path=temporary["test"],
            train=train,
            test=test,
            feature_names=feature_names,
            process_groups=process_groups,
            dataset_manifest_path=temporary["dataset_manifest"],
            preparation_path=temporary["preparation"],
        )
        _publish_files(temporary, targets)

    return PreparationResult(
        dataset="swat",
        output_dir=destination,
        train_path=targets["train"],
        test_path=targets["test"],
        dataset_manifest_path=targets["dataset_manifest"],
        preparation_path=targets["preparation"],
    )


def prepare_wadi(
    normal_raw: str | Path,
    attack_raw: str | Path,
    output_dir: str | Path,
    *,
    interval_seconds: int = 10,
    train_skip_rows: int = 21_600,
    expected_feature_count: int | None = 123,
    force: bool = False,
) -> PreparationResult:
    """Prepare the 10-second WADI train and test CSVs."""

    _validate_parameters(interval_seconds, train_skip_rows)
    normal_source = _normalize_source_path(normal_raw, "normal_raw")
    attack_source = _normalize_source_path(attack_raw, "attack_raw")
    destination, targets = _output_paths(
        output_dir,
        (normal_source, attack_source),
        force=force,
    )

    with tempfile.TemporaryDirectory(
        prefix=".stager-prepare-",
        dir=destination,
    ) as temporary_directory:
        root = Path(temporary_directory)
        normal_normalized = root / "normal_raw.csv"
        attack_normalized = root / "attack_raw.csv"
        normal_intermediate = root / "normal_preprocessed.csv"
        attack_intermediate = root / "attack_preprocessed.csv"
        temporary = {
            "train": root / "train.csv",
            "test": root / "test.csv",
            "dataset_manifest": root / "dataset_manifest.json",
            "preparation": root / "preparation.json",
        }
        _write_wadi_normalized_raw(normal_source, normal_normalized)
        _write_wadi_normalized_raw(attack_source, attack_normalized)
        _preprocess_wadi(
            normal_normalized,
            normal_intermediate,
            normal_split=True,
        )
        _preprocess_wadi(
            attack_normalized,
            attack_intermediate,
            normal_split=False,
        )
        train = _downsample_wadi(
            normal_intermediate,
            temporary["train"],
            interval_seconds=interval_seconds,
            skip_rows=train_skip_rows,
        )
        test = _downsample_wadi(
            attack_intermediate,
            temporary["test"],
            interval_seconds=interval_seconds,
            skip_rows=0,
        )
        feature_names, process_groups = _validate_prepared(
            "wadi",
            train,
            test,
            expected_feature_count=expected_feature_count,
        )
        _write_documents(
            dataset="wadi",
            interval_seconds=interval_seconds,
            train_skip_rows=train_skip_rows,
            normal_source=normal_source,
            attack_source=attack_source,
            train_path=temporary["train"],
            test_path=temporary["test"],
            train=train,
            test=test,
            feature_names=feature_names,
            process_groups=process_groups,
            dataset_manifest_path=temporary["dataset_manifest"],
            preparation_path=temporary["preparation"],
        )
        _publish_files(temporary, targets)

    return PreparationResult(
        dataset="wadi",
        output_dir=destination,
        train_path=targets["train"],
        test_path=targets["test"],
        dataset_manifest_path=targets["dataset_manifest"],
        preparation_path=targets["preparation"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stager-prepare-data",
        description=(
            "Prepare SWaT, WADI, or HAI CSV files."
        ),
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    for dataset in ("swat", "wadi"):
        subparser = subparsers.add_parser(dataset)
        subparser.add_argument("--normal-raw", required=True, type=Path)
        subparser.add_argument("--attack-raw", required=True, type=Path)
        subparser.add_argument("--output-dir", required=True, type=Path)
        subparser.add_argument("--interval-seconds", type=int, default=10)
        subparser.add_argument("--train-skip-rows", type=int, default=21_600)
        subparser.add_argument(
            "--force",
            action="store_true",
            help="replace an existing prepared output set",
        )
    hai = subparsers.add_parser("hai")
    hai.add_argument("--raw-dir", required=True, type=Path)
    hai.add_argument("--output-dir", required=True, type=Path)
    hai.add_argument("--force", action="store_true", help="replace an existing prepared output set")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset preparation command."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.dataset == "hai":
            from .hai import prepare_hai

            result = prepare_hai(arguments.raw_dir, arguments.output_dir, force=arguments.force)
        else:
            function = prepare_swat if arguments.dataset == "swat" else prepare_wadi
            result = function(
                arguments.normal_raw,
                arguments.attack_raw,
                arguments.output_dir,
                interval_seconds=arguments.interval_seconds,
                train_skip_rows=arguments.train_skip_rows,
                force=arguments.force,
            )
    except PreparationError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "dataset": result.dataset,
                "dataset_manifest": str(result.dataset_manifest_path),
                "preparation": str(result.preparation_path),
                "test": str(result.test_path),
                "train": str(result.train_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
