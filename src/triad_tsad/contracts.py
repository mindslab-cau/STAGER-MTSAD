"""Shared data contracts without baseline-specific preprocessing policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

import numpy as np

ColumnRef: TypeAlias = str | int


class ContractError(ValueError):
    """Raised when an experiment object violates a shared contract."""


def _mapping_copy(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field_name} must be a mapping")
    return dict(value)


def _nested_shape(value: Any, field_name: str, *, top_level: bool = True) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        if top_level:
            raise ContractError(f"{field_name} must be an array-like value, not text")
        return ()

    raw_shape = getattr(value, "shape", None)
    if raw_shape is not None:
        try:
            shape = tuple(int(dimension) for dimension in raw_shape)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{field_name} has an invalid shape") from exc
        if any(dimension < 0 for dimension in shape):
            raise ContractError(f"{field_name} has an invalid shape")
        return shape

    if isinstance(value, Sequence):
        length = len(value)
        if length == 0:
            return (0,)

        child_shapes = [
            _nested_shape(item, field_name, top_level=False)
            for item in value
        ]
        first_shape = child_shapes[0]
        if any(shape != first_shape for shape in child_shapes[1:]):
            raise ContractError(f"{field_name} must not be ragged")
        return (length, *first_shape)

    if top_level:
        raise ContractError(f"{field_name} must be an array-like value")
    return ()


def _one_dimensional_length(value: Any, field_name: str) -> int:
    shape = _nested_shape(value, field_name)
    if len(shape) != 1:
        raise ContractError(f"{field_name} must be one-dimensional; received shape {shape}")
    return shape[0]


def _finite_numeric_array(value: Any, field_name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ContractError(
            f"{field_name} must be convertible to a numeric array"
        ) from exc
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ContractError(f"{field_name} must contain numeric values")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{field_name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Portable locations and schema hints for one logical dataset.

    Paths may point outside the repository and are not required to exist while
    loading a manifest. Preprocessing choices belong to each baseline or run
    configuration rather than this shared dataset contract.
    """

    logical_id: str
    train_path: Path | str
    test_path: Path | str
    timestamp: ColumnRef | None = None
    label: ColumnRef | None = None
    schema: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.logical_id, str) or not self.logical_id.strip():
            raise ContractError("logical_id must be a non-empty string")

        for field_name in ("train_path", "test_path"):
            raw_path = getattr(self, field_name)
            if isinstance(raw_path, str) and not raw_path.strip():
                raise ContractError(f"{field_name} must not be empty")
            try:
                normalized_path = Path(raw_path)
            except TypeError as exc:
                raise ContractError(f"{field_name} must be a path-like value") from exc
            object.__setattr__(self, field_name, normalized_path)

        for field_name in ("timestamp", "label"):
            column = getattr(self, field_name)
            if column is not None and (
                isinstance(column, bool) or not isinstance(column, (str, int))
            ):
                raise ContractError(f"{field_name} must be a string, integer, or null")
            if isinstance(column, str) and not column.strip():
                raise ContractError(f"{field_name} must not be an empty string")

        object.__setattr__(self, "schema", _mapping_copy(self.schema, "schema"))
        object.__setattr__(self, "metadata", _mapping_copy(self.metadata, "metadata"))


@dataclass(slots=True)
class RunResult:
    """Normalized output from a baseline or TriAD run.

    Point scores always follow the convention that a larger value indicates a
    more anomalous point. Feature scores use the internal ``[time, feature]``
    orientation. :attr:`score_streams` exposes the historical evaluator
    contract ``[stream, time]`` and must be used for feature-wise FPR voting.
    Arrays are retained in their original container type.
    """

    SCORE_SEMANTICS: ClassVar[str] = "higher_is_more_anomalous"

    point_scores: Any
    feature_scores: Any | None = None
    labels: Any | None = None
    timestamps: Any | None = None
    indices: Any | None = None
    metrics: Mapping[str, Real] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metrics = _mapping_copy(self.metrics, "metrics")
        self.metadata = _mapping_copy(self.metadata, "metadata")
        self.config = _mapping_copy(self.config, "config")
        self.validate()

    def validate(self) -> RunResult:
        """Validate dimensions and alignment, then return this result."""

        point_count = _one_dimensional_length(self.point_scores, "point_scores")
        if point_count == 0:
            raise ContractError("point_scores must contain at least one score")
        _finite_numeric_array(self.point_scores, "point_scores")

        if self.feature_scores is not None:
            feature_shape = _nested_shape(self.feature_scores, "feature_scores")
            if len(feature_shape) != 2:
                raise ContractError(
                    "feature_scores must be two-dimensional; "
                    f"received shape {feature_shape}"
                )
            if feature_shape[0] != point_count:
                raise ContractError(
                    "feature_scores and point_scores must have the same first dimension"
                )
            if feature_shape[1] == 0:
                raise ContractError(
                    "feature_scores must contain at least one feature"
                )
            _finite_numeric_array(self.feature_scores, "feature_scores")

        for field_name in ("labels", "timestamps", "indices"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if _one_dimensional_length(value, field_name) != point_count:
                raise ContractError(
                    f"{field_name} and point_scores must have the same length"
                )

        if self.labels is not None:
            labels = _finite_numeric_array(self.labels, "labels")
            if not np.all((labels == 0) | (labels == 1)):
                raise ContractError("labels must contain only binary 0/1 values")

        for metric_name, metric_value in self.metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise ContractError("metric names must be non-empty strings")
            if isinstance(metric_value, bool) or not isinstance(metric_value, Real):
                raise ContractError(f"metric {metric_name!r} must be numeric")
            if not np.isfinite(float(metric_value)):
                raise ContractError(f"metric {metric_name!r} must be finite")

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
                raise ContractError("seed must be an integer or null")
            if self.seed < 0:
                raise ContractError("seed must be non-negative")
            self.seed = int(self.seed)

        return self

    @property
    def score_streams(self) -> np.ndarray:
        """Return the exact legacy evaluator input as ``[stream, time]``.

        Neural baselines historically returned one score stream per sensor and
        the evaluator selected one threshold per row before OR-voting. The
        one-dimensional point score retained by this adapter is diagnostic
        only whenever feature scores are available.
        """

        if self.feature_scores is not None:
            features = np.asarray(self.feature_scores)
            return np.ascontiguousarray(features.T)
        points = np.asarray(self.point_scores)
        return np.ascontiguousarray(points.reshape(1, -1))
