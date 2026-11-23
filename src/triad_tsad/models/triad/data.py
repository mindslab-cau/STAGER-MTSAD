"""Prepared-array adapter with explicit next-point alignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def prepared_values(values: Any, *, name: str) -> np.ndarray:
    """Validate an already preprocessed ``[time, feature]`` array."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [time, feature].")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} is too small for time-series forecasting.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


class NextPointDataset(Dataset):
    """Create ``window -> next point`` samples from a prepared array."""

    def __init__(
        self,
        values: Any,
        *,
        window_size: int,
        stride: int = 1,
    ) -> None:
        self.values = prepared_values(values, name="values")
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        if stride < 1:
            raise ValueError("stride must be positive.")
        if len(self.values) <= window_size:
            raise ValueError(
                "values must contain at least window_size + 1 rows."
            )
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.starts = np.arange(
            0,
            len(self.values) - self.window_size,
            self.stride,
            dtype=np.int64,
        )

    @property
    def target_indices(self) -> np.ndarray:
        return self.starts + self.window_size

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[item])
        target = start + self.window_size
        return {
            "given": torch.as_tensor(
                self.values[start:target],
                dtype=torch.float32,
            ),
            "answer": torch.as_tensor(
                self.values[target],
                dtype=torch.float32,
            ),
        }


def align_to_targets(
    values: Sequence[Any] | np.ndarray | None,
    *,
    raw_length: int,
    target_indices: np.ndarray,
    name: str,
) -> np.ndarray | None:
    """Accept either raw-time or already next-point-aligned metadata."""

    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(array) == raw_length:
        return array[target_indices]
    if len(array) == len(target_indices):
        return array
    raise ValueError(
        f"{name} must have raw length {raw_length} or aligned length "
        f"{len(target_indices)}."
    )


__all__ = ["NextPointDataset", "align_to_targets", "prepared_values"]
