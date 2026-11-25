"""Lazy model construction for optional PyOD classical baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np


class ClassicalDependencyError(ImportError):
    """Raised when an optional classical model dependency is unavailable."""


class ClassicalEstimator(Protocol):
    """Minimal estimator API used by the unified runner."""

    def fit(self, values: np.ndarray) -> Any:
        """Fit on normal training samples."""

    def decision_function(self, values: np.ndarray) -> Any:
        """Return one anomaly score per sample."""


def build_estimator(
    model: str,
    params: Mapping[str, Any],
) -> ClassicalEstimator:
    """Construct the original PyOD estimator without importing it eagerly."""

    kwargs = dict(params)
    if kwargs.get("n_jobs") is None:
        kwargs.pop("n_jobs", None)

    if model == "knn":
        try:
            from pyod.models.knn import KNN
        except ImportError as exc:
            raise ClassicalDependencyError(
                "KNN requires the optional PyOD dependency. Install the "
                "project's classical baseline extra."
            ) from exc
        return KNN(**kwargs)

    if model == "ocsvm":
        try:
            from pyod.models.ocsvm import OCSVM
        except ImportError as exc:
            raise ClassicalDependencyError(
                "OCSVM requires the optional PyOD dependency. Install the "
                "project's classical baseline extra."
            ) from exc
        return OCSVM(**kwargs)

    raise ValueError(f"unsupported classical model: {model}")
