"""Unified KNN and OCSVM baseline interface.

Configuration exports stay dependency-light. Data, model, runner, and
evaluation dependencies are imported only when their public names are used.
"""

from __future__ import annotations

import importlib
from typing import Any

from .config import (
    ClassicalConfigError,
    ClassicalRunConfig,
    config_from_mapping,
    load_classical_config,
)

_LAZY_EXPORTS = {
    "ClassicalDataError": (".data", "ClassicalDataError"),
    "DatasetSchema": (".data", "DatasetSchema"),
    "PreparedClassicalData": (".data", "PreparedClassicalData"),
    "load_classical_csv": (".data", "load_classical_csv"),
    "resolve_dataset_schema": (".data", "resolve_dataset_schema"),
    "ClassicalDependencyError": (".models", "ClassicalDependencyError"),
    "ClassicalEstimator": (".models", "ClassicalEstimator"),
    "build_estimator": (".models", "build_estimator"),
    "ClassicalRunError": (".runner", "ClassicalRunError"),
    "run_classical_baseline": (".runner", "run_classical_baseline"),
    "run_classical_sweep": (".runner", "run_classical_sweep"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "ClassicalConfigError",
    "ClassicalDataError",
    "ClassicalDependencyError",
    "ClassicalEstimator",
    "ClassicalRunConfig",
    "ClassicalRunError",
    "DatasetSchema",
    "PreparedClassicalData",
    "build_estimator",
    "config_from_mapping",
    "load_classical_config",
    "load_classical_csv",
    "resolve_dataset_schema",
    "run_classical_baseline",
    "run_classical_sweep",
]
