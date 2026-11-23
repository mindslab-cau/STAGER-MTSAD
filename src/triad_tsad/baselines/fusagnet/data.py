"""Next-point windowing used by the legacy FuSAGNet experiment path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


class FuSAGNetDataError(ValueError):
    """Raised when input arrays cannot form valid FuSAGNet windows."""


@dataclass(frozen=True, slots=True)
class NextPointWindows:
    """Windows and targets aligned to the point immediately after each window."""

    inputs: np.ndarray
    targets: np.ndarray
    target_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.inputs.ndim != 3:
            raise FuSAGNetDataError("inputs must have shape [sample, time, feature]")
        if self.targets.ndim != 2:
            raise FuSAGNetDataError("targets must have shape [sample, feature]")
        if self.target_indices.ndim != 1:
            raise FuSAGNetDataError("target_indices must be one-dimensional")
        sample_count = self.inputs.shape[0]
        if self.targets.shape[0] != sample_count:
            raise FuSAGNetDataError("inputs and targets must have equal sample counts")
        if self.target_indices.shape[0] != sample_count:
            raise FuSAGNetDataError(
                "inputs and target_indices must have equal sample counts"
            )
        if self.inputs.shape[2] != self.targets.shape[1]:
            raise FuSAGNetDataError("input and target feature counts must match")

    def align(self, values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
        """Select values at the next-point target indices."""

        array = np.asarray(values)
        if array.ndim != 1:
            raise FuSAGNetDataError(f"{name} must be one-dimensional")
        if self.target_indices.size and self.target_indices[-1] >= array.shape[0]:
            raise FuSAGNetDataError(
                f"{name} is too short for the generated target indices"
            )
        return array[self.target_indices]


def make_next_point_windows(
    values: Any,
    window_size: int,
    *,
    stride: int = 1,
    valid_starts: Sequence[int] | np.ndarray | None = None,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> NextPointWindows:
    """Create ``[s:s+window] -> s+window`` samples.

    This deliberately retains the legacy next-point anchor. For ``N`` rows and
    stride one, it returns ``N - window_size`` samples with target indices
    ``window_size .. N-1``.
    """

    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise FuSAGNetDataError("window_size must be a positive integer")
    if window_size <= 0:
        raise FuSAGNetDataError("window_size must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise FuSAGNetDataError("stride must be a positive integer")

    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise FuSAGNetDataError("values must have shape [time, feature]")
    if not np.isfinite(array).all():
        raise FuSAGNetDataError(
            "values contain NaN or infinity; apply an explicit preprocessing policy"
        )
    if array.shape[0] <= window_size:
        raise FuSAGNetDataError(
            "values must contain at least window_size + 1 rows"
        )
    if array.shape[1] == 0:
        raise FuSAGNetDataError("values must contain at least one feature")

    final_start = array.shape[0] - window_size - 1
    if valid_starts is None:
        starts = np.arange(0, final_start + 1, stride, dtype=np.int64)
    else:
        raw_starts = np.asarray(valid_starts)
        if raw_starts.ndim != 1 or raw_starts.size == 0:
            raise FuSAGNetDataError(
                "valid_starts must be a non-empty one-dimensional sequence"
            )
        normalized: list[int] = []
        for item in raw_starts.tolist():
            if isinstance(item, bool) or not isinstance(item, Integral):
                raise FuSAGNetDataError(
                    "valid_starts must contain only integer indices"
                )
            start = int(item)
            if not 0 <= start <= final_start:
                raise FuSAGNetDataError(
                    f"window start {start} is outside 0..{final_start}"
                )
            normalized.append(start)
        starts = np.asarray(normalized, dtype=np.int64)
        if np.any(starts[1:] <= starts[:-1]):
            raise FuSAGNetDataError(
                "valid_starts must be strictly increasing without duplicates"
            )
        starts = np.ascontiguousarray(starts[::stride])
    target_indices = starts + window_size
    inputs = np.stack([array[start : start + window_size] for start in starts])
    targets = array[target_indices]
    return NextPointWindows(
        np.ascontiguousarray(inputs),
        np.ascontiguousarray(targets),
        target_indices,
    )
