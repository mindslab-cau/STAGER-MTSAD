"""Stride-aware, last-point windowing for the MSHTrans adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


class MSHTransDataError(ValueError):
    """Raised when arrays cannot form valid MSHTrans windows."""


@dataclass(frozen=True, slots=True)
class LastPointWindows:
    """Windows whose score and metadata are anchored to each final row."""

    inputs: np.ndarray
    start_indices: np.ndarray
    anchor_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.inputs.ndim != 3:
            raise MSHTransDataError(
                "inputs must have shape [sample, time, feature]"
            )
        if self.start_indices.ndim != 1 or self.anchor_indices.ndim != 1:
            raise MSHTransDataError(
                "start_indices and anchor_indices must be one-dimensional"
            )
        sample_count = self.inputs.shape[0]
        if len(self.start_indices) != sample_count:
            raise MSHTransDataError(
                "inputs and start_indices must have equal sample counts"
            )
        if len(self.anchor_indices) != sample_count:
            raise MSHTransDataError(
                "inputs and anchor_indices must have equal sample counts"
            )

    def align(self, values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
        """Select labels or timestamps at each window's last point."""

        array = np.asarray(values)
        if array.ndim != 1:
            raise MSHTransDataError(f"{name} must be one-dimensional")
        if self.anchor_indices.size and self.anchor_indices[-1] >= len(array):
            raise MSHTransDataError(
                f"{name} is too short for the generated anchor indices"
            )
        return array[self.anchor_indices]


def _normalize_starts(
    valid_starts: Sequence[int] | np.ndarray | None,
    *,
    final_start: int,
    stride: int,
) -> np.ndarray:
    if valid_starts is None:
        return np.arange(0, final_start + 1, stride, dtype=np.int64)

    raw = np.asarray(valid_starts)
    if raw.ndim != 1:
        raise MSHTransDataError("valid_starts must be one-dimensional")
    if raw.size == 0:
        raise MSHTransDataError("valid_starts must contain at least one index")

    normalized: list[int] = []
    for item in raw.tolist():
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise MSHTransDataError(
                "valid_starts must contain only integer indices"
            )
        index = int(item)
        if not 0 <= index <= final_start:
            raise MSHTransDataError(
                f"window start {index} is outside the valid range 0..{final_start}"
            )
        normalized.append(index)

    starts = np.asarray(normalized, dtype=np.int64)
    if np.any(starts[1:] <= starts[:-1]):
        raise MSHTransDataError(
            "valid_starts must be strictly increasing without duplicates"
        )
    # The historical loader first enumerated continuity-safe candidates and
    # then subsampled that ordered list by the requested stride.
    return np.ascontiguousarray(starts[::stride])


def make_last_point_windows(
    values: Any,
    window_size: int,
    *,
    stride: int = 1,
    valid_starts: Sequence[int] | np.ndarray | None = None,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> LastPointWindows:
    """Create windows `[s:s+W]` anchored at `s+W-1`.

    With `N` rows and no continuity filtering, this returns
    `floor((N-W)/stride)+1` windows. A common preprocessor may pass the ordered
    continuity-safe candidate starts in `valid_starts`; this function then
    applies the requested stride to that candidate list before creating
    windows. This retains the historical continuity-then-stride order without
    changing the alignment rule.
    """

    if isinstance(window_size, bool) or not isinstance(window_size, Integral):
        raise MSHTransDataError("window_size must be a positive integer")
    window_size = int(window_size)
    if window_size <= 0:
        raise MSHTransDataError("window_size must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, Integral):
        raise MSHTransDataError("stride must be a positive integer")
    stride = int(stride)
    if stride <= 0:
        raise MSHTransDataError("stride must be a positive integer")

    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise MSHTransDataError("values must have shape [time, feature]")
    if array.shape[1] == 0:
        raise MSHTransDataError("values must contain at least one feature")
    if not np.isfinite(array).all():
        raise MSHTransDataError(
            "values contain NaN or infinity; apply an explicit preprocessing policy"
        )
    if len(array) < window_size:
        raise MSHTransDataError(
            "values must contain at least window_size rows"
        )

    final_start = len(array) - window_size
    starts = _normalize_starts(
        valid_starts,
        final_start=final_start,
        stride=stride,
    )
    inputs = np.stack(
        [array[start : start + window_size] for start in starts],
        axis=0,
    )
    anchors = starts + window_size - 1
    return LastPointWindows(
        inputs=np.ascontiguousarray(inputs),
        start_indices=starts,
        anchor_indices=anchors,
    )
