"""Time-series Deep SVDD baseline with retained MIT attribution."""

from __future__ import annotations

import importlib
from typing import Any

from .config import (
    DeepSVDDConfig,
    DeepSVDDConfigError,
    config_from_mapping,
    load_deep_svdd_config,
)

_LAZY_EXPORTS = {
    "DeepSVDDDataError": (".data", "DeepSVDDDataError"),
    "LegacyNextWindowDataset": (".data", "LegacyNextWindowDataset"),
    "PreparedDataset": (".data", "PreparedDataset"),
    "PreparedSplit": (".data", "PreparedSplit"),
    "prepare_dataset": (".data", "prepare_dataset"),
    "DeepSVDDModelError": (".model", "DeepSVDDModelError"),
    "TimeSeriesLeNet": (".model", "TimeSeriesLeNet"),
    "TimeSeriesLeNetAutoencoder": (".model", "TimeSeriesLeNetAutoencoder"),
    "DeepSVDDRunner": (".runner", "DeepSVDDRunner"),
    "DeepSVDDRunnerError": (".runner", "DeepSVDDRunnerError"),
    "run_deep_svdd": (".runner", "run_deep_svdd"),
    "DeepSVDDTrainingError": (".trainer", "DeepSVDDTrainingError"),
    "TrainingHistory": (".trainer", "TrainingHistory"),
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
    "DeepSVDDConfig",
    "DeepSVDDConfigError",
    "DeepSVDDDataError",
    "DeepSVDDModelError",
    "DeepSVDDRunner",
    "DeepSVDDRunnerError",
    "DeepSVDDTrainingError",
    "LegacyNextWindowDataset",
    "PreparedDataset",
    "PreparedSplit",
    "TimeSeriesLeNet",
    "TimeSeriesLeNetAutoencoder",
    "TrainingHistory",
    "config_from_mapping",
    "load_deep_svdd_config",
    "prepare_dataset",
    "run_deep_svdd",
]
