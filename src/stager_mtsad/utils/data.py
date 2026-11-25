"""Load and align prepared SWaT, WADI, and HAI windows."""

from __future__ import annotations

import glob
import os
from datetime import timedelta
from typing import Any

import dateutil.parser
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import trange

from .dataset_paths import DATASET_PATHS, resolve_dataset_name


DATASET_CONFIG = {
    "HAI": {
        "timestamp_field": "timestamp",
        "label_field": "Attack",
        "useless_fields": ["Attack"],
        "normal_value": 0,
        "attack_value": 1,
        "multi_timestamp": False,
    },
    "SWaT": {
        "timestamp_field": "timestamp",
        "label_field": "label",
        "useless_fields": ["label"],
        "normal_value": "Normal",
        "attack_value": "Attack",
        "multi_timestamp": False,
    },
    "WADI": {
        "timestamp_field": ["Date", "Time"],
        "label_field": "Attack LABLE (1:No Attack, -1:Attack)",
        "useless_fields": [
            "Attack LABLE (1:No Attack, -1:Attack)",
            "Row",
        ],
        "normal_value": 1,
        "attack_value": -1,
        "multi_timestamp": True,
    },
}


class MyDataset(Dataset[dict[str, Any]]):
    """Retained next-point dataset with timestamp-continuity filtering."""

    def __init__(
        self,
        timestamps,
        df,
        stride: int = 1,
        time_lag: int = 1,
        window_size: int = 60,
        predict_size: int = 1,
        downsampled: bool = False,
        downsample_interval: int | None = None,
    ) -> None:
        print("Initializing MyDataset..", flush=True)
        self._window_size = int(window_size)
        self._predict_size = int(predict_size)
        self._downsampled = bool(downsampled)
        self._downsample_interval = downsample_interval
        self.ts = np.asarray(timestamps)

        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        df_numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
        self.tag_values = np.asarray(df_numeric, dtype=np.float32)

        if not self._downsampled:
            self.ts = self.ts[::time_lag]
            self.tag_values = self.tag_values[::time_lag]

        valid_indices: list[int] = []
        candidate_count = (
            len(self.ts) - self._window_size - self._predict_size + 1
        )
        if self._downsampled and self._downsample_interval is not None:
            expected_interval = self._downsample_interval
            for left in trange(max(0, candidate_count)):
                right = left + self._window_size
                time_difference = (
                    dateutil.parser.parse(
                        self.ts[right + self._predict_size - 1]
                    )
                    - dateutil.parser.parse(self.ts[left])
                )
                expected_difference = timedelta(
                    seconds=(
                        self._window_size + self._predict_size - 1
                    )
                    * expected_interval
                )
                if time_difference == expected_difference:
                    valid_indices.append(left)
        else:
            for left in trange(max(0, candidate_count)):
                right = left + self._window_size
                time_difference = (
                    dateutil.parser.parse(
                        self.ts[right + self._predict_size - 1]
                    )
                    - dateutil.parser.parse(self.ts[left])
                )
                expected_difference = timedelta(
                    seconds=(
                        self._window_size + self._predict_size - 1
                    )
                    * time_lag
                )
                if time_difference == expected_difference:
                    valid_indices.append(left)

        self.valid_idxs = np.asarray(valid_indices, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        self.feature_names: tuple[str, ...] = ()
        print(f"Number of valid windows: {self.n_idxs}", flush=True)
        if self._downsampled:
            print(
                "Dataset initialized for downsampled data with interval: "
                f"{self._downsample_interval}s",
                flush=True,
            )
        else:
            print(
                f"Dataset initialized for original data with time_lag: {time_lag}",
                flush=True,
            )

    def __len__(self) -> int:
        return self.n_idxs

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self.valid_idxs[index])
        target = start + self._window_size
        final_target = target + self._predict_size - 1
        item: dict[str, Any] = {"ts": self.ts[final_target]}
        item["given"] = torch.as_tensor(
            self.tag_values[start : start + self._window_size]
        )
        if self._predict_size == 1:
            item["answer"] = torch.as_tensor(self.tag_values[target, :])
        else:
            item["answer"] = torch.as_tensor(
                self.tag_values[target : target + self._predict_size, :]
            )
        return item

    def get_valid_indices(self) -> np.ndarray:
        return self.valid_idxs.copy()

    def get_target_indices(self) -> np.ndarray:
        return (
            self.valid_idxs.astype(np.int64)
            + self._window_size
            + self._predict_size
            - 1
        )


def dataframe_from_csv(target: str) -> pd.DataFrame:
    """Create a dataframe and normalize surrounding header whitespace."""

    return pd.read_csv(target, sep=",").rename(columns=lambda value: value.strip())


def dataframe_concat_csvs(targets: list[str]) -> pd.DataFrame:
    """Load and concatenate an ordered list of CSV files."""

    print(f"Loading and concatenating {len(targets)} CSV files")
    return pd.concat([dataframe_from_csv(path) for path in targets])


def normalize(
    frame: pd.DataFrame,
    tag_min: pd.Series,
    tag_max: pd.Series,
) -> pd.DataFrame:
    """Apply the retained per-feature min-max transformation."""

    normalized = frame.copy()
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    numeric = numeric.astype(np.float32)
    for column in numeric.columns:
        if tag_min[column] == tag_max[column]:
            normalized[column] = numeric[column] - tag_min[column]
        else:
            normalized[column] = (
                numeric[column] - tag_min[column]
            ) / (tag_max[column] - tag_min[column])
    return normalized


def load_dataset(
    dataset_name,
    window_size: int = 60,
    stride: int = 1,
    time_lag: int = 1,
    predict_size: int = 1,
    device=None,
    normalize_data: bool = True,
    apply_ewm: bool = True,
    ewm_alpha: float = 0.8,
    normalization_stats=None,
    downsampled: bool = False,
    downsample_interval: int | None = None,
):
    """Load one registered dataset using the retained STAGER preprocessing."""

    del device
    dataset_name = resolve_dataset_name(dataset_name)
    if dataset_name not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Available options: {list(DATASET_PATHS.keys())}"
        )

    is_test = "_Test" in dataset_name
    base_name = dataset_name.replace("_Test", "") if is_test else dataset_name

    auto_downsampled = False
    auto_downsample_interval = None
    if "_down_" in base_name:
        parts = base_name.split("_down_")
        base_name = parts[0]
        if len(parts) > 1:
            auto_downsample_interval = int(parts[1])
            auto_downsampled = True
    if auto_downsampled:
        downsampled = True
        if downsample_interval is None:
            downsample_interval = auto_downsample_interval
        print(
            f"Auto-detected downsampled dataset: {dataset_name} "
            f"with interval {downsample_interval}s"
        )

    if base_name in {"SWaTog", "SWaTogskip"}:
        base_name = "SWaT"
    if base_name in {"WADImodeskip", "WADImode"}:
        base_name = "WADI"
    if base_name not in DATASET_CONFIG:
        raise ValueError(f"Configuration not found for dataset: {base_name}")

    config = DATASET_CONFIG[base_name]
    timestamp_field = config["timestamp_field"]
    label_field = config["label_field"]
    useless_fields = list(config["useless_fields"])
    multi_timestamp = bool(config.get("multi_timestamp", False))
    dataset_path = DATASET_PATHS[dataset_name]
    print(f"Loading {dataset_name} dataset from {dataset_path}")

    if "*" in dataset_path:
        file_paths = sorted(glob.glob(dataset_path))
        if not file_paths:
            raise FileNotFoundError(
                f"No files found matching pattern: {dataset_path}"
            )
        print(f"Found {len(file_paths)} files: {file_paths}")
        frame = dataframe_concat_csvs(file_paths)
    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}")
        frame = dataframe_from_csv(dataset_path)

    if multi_timestamp:
        timestamp_fields = list(timestamp_field)
        frame["combined_timestamp"] = (
            frame[timestamp_fields[0]].astype(str)
            + " "
            + frame[timestamp_fields[1]].astype(str)
        )
        timestamp_field = "combined_timestamp"
        for field in timestamp_fields:
            if field not in useless_fields:
                useless_fields.append(field)

    raw_labels = frame[label_field].values if is_test else None
    fields_to_drop = (
        [timestamp_field]
        if not isinstance(timestamp_field, list)
        else timestamp_field
    )
    fields_to_drop = list(fields_to_drop) + useless_fields
    if multi_timestamp:
        fields_to_drop.append("combined_timestamp")
        fields_to_drop = list(set(fields_to_drop))
    valid_columns = frame.columns.drop(fields_to_drop)
    timestamps = frame[timestamp_field].values

    normalization_stats_output = None
    if normalize_data:
        if normalization_stats is not None:
            print(
                f"[INFO] Using pre-computed normalization statistics for "
                f"{dataset_name}"
            )
            tag_min = normalization_stats["tag_min"]
            tag_max = normalization_stats["tag_max"]
            if not all(column in tag_min.index for column in valid_columns):
                missing = [
                    column
                    for column in valid_columns
                    if column not in tag_min.index
                ]
                raise ValueError(
                    f"Missing columns in normalization_stats: {missing}"
                )
        else:
            print(
                f"[INFO] Computing normalization statistics from {dataset_name}"
            )
            tag_min = frame[valid_columns].min()
            tag_max = frame[valid_columns].max()
            normalization_stats_output = {
                "tag_min": tag_min,
                "tag_max": tag_max,
            }
        transformed = normalize(frame[valid_columns], tag_min, tag_max)
        if apply_ewm:
            transformed = transformed.ewm(alpha=ewm_alpha).mean()
        features = transformed.values
    else:
        features = frame[valid_columns].values

    dataset = MyDataset(
        timestamps=timestamps,
        df=features,
        stride=stride,
        time_lag=time_lag,
        window_size=window_size,
        predict_size=predict_size,
        downsampled=downsampled,
        downsample_interval=downsample_interval,
    )
    dataset.feature_names = tuple(str(column) for column in valid_columns)

    aligned_labels = None
    if is_test and raw_labels is not None:
        aligned_labels = process_labels_for_indices(
            dataset.get_valid_indices(),
            raw_labels,
            config,
            window_size,
            predict_size,
        )
    return dataset, aligned_labels, None, normalization_stats_output


def process_labels_for_indices(
    valid_indices,
    all_labels_raw,
    config,
    window_size,
    predict_size,
) -> np.ndarray:
    """Align raw labels to each retained next-point target."""

    aligned_labels: list[int] = []
    normal_value = config["normal_value"]
    attack_value = config["attack_value"]
    for index in valid_indices:
        prediction_index = int(index) + window_size + predict_size - 1
        if prediction_index >= len(all_labels_raw):
            continue
        label = all_labels_raw[prediction_index]
        if pd.isna(label):
            aligned_labels.append(0)
        elif label == normal_value:
            aligned_labels.append(0)
        elif label == attack_value:
            aligned_labels.append(1)
        else:
            print(f"Unknown label value: {label}")
            aligned_labels.append(0)
    return np.asarray(aligned_labels, dtype=np.float32)


__all__ = [
    "DATASET_CONFIG",
    "MyDataset",
    "dataframe_concat_csvs",
    "dataframe_from_csv",
    "load_dataset",
    "normalize",
    "process_labels_for_indices",
]
