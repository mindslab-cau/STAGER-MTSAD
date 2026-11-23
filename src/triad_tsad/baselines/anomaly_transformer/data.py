"""Prepared-array windowing for the legacy Anomaly Transformer adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _optional_source_vector(
    value: Sequence[Any] | np.ndarray | torch.Tensor | None,
    *,
    source_length: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != source_length:
        raise ValueError(f"{name} must be a one-dimensional source-length vector")
    return array


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class LegacyNextWindowDataset(Dataset[dict[str, Any]]):
    """Create reconstruction windows with the historical next-point anchor.

    A window beginning at row ``i`` contains rows
    ``[i, i + window_size)``. Its anomaly score is deliberately assigned to
    row ``i + window_size``, matching the TSAD integration even though the
    model reconstructs the input window. This alignment is retained for
    reproducibility and is not a claim that the score predicts the next row.

    Input arrays must already have completed baseline preprocessing. Raw SWaT
    and WADI ingestion is outside this adapter.
    """

    def __init__(
        self,
        values: np.ndarray | torch.Tensor,
        *,
        window_size: int,
        stride: int = 1,
        labels: Sequence[Any] | np.ndarray | torch.Tensor | None = None,
        timestamps: Sequence[Any] | np.ndarray | torch.Tensor | None = None,
        indices: Sequence[Any] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        value_tensor = torch.as_tensor(values, dtype=torch.float32)
        if value_tensor.ndim != 2:
            raise ValueError("values must have shape [time, feature]")
        if not torch.isfinite(value_tensor).all():
            raise ValueError("values must not contain NaN or infinite values")
        if window_size < 1:
            raise ValueError("window_size must be at least one")
        if stride < 1:
            raise ValueError("stride must be at least one")
        if len(value_tensor) <= window_size:
            raise ValueError(
                "legacy next anchoring requires at least window_size + 1 rows"
            )

        source_length = len(value_tensor)
        self.values = value_tensor
        self.window_size = window_size
        self.stride = stride
        self.labels = _optional_source_vector(
            labels,
            source_length=source_length,
            name="labels",
        )
        self.timestamps = _optional_source_vector(
            timestamps,
            source_length=source_length,
            name="timestamps",
        )
        if indices is None:
            self.indices = np.arange(source_length, dtype=np.int64)
        else:
            checked_indices = _optional_source_vector(
                indices,
                source_length=source_length,
                name="indices",
            )
            assert checked_indices is not None
            self.indices = checked_indices

        self.starts = np.arange(
            0,
            source_length - window_size,
            stride,
            dtype=np.int64,
        )

    @property
    def anchor_indices(self) -> np.ndarray:
        """Return source positions to which emitted window scores are assigned."""

        return self.starts + self.window_size

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, position: int) -> dict[str, Any]:
        start = int(self.starts[position])
        anchor = start + self.window_size
        item: dict[str, Any] = {
            "given": self.values[start:anchor],
            "index": _python_scalar(self.indices[anchor]),
        }
        if self.labels is not None:
            item["label"] = _python_scalar(self.labels[anchor])
        if self.timestamps is not None:
            item["timestamp"] = _python_scalar(self.timestamps[anchor])
        return item
