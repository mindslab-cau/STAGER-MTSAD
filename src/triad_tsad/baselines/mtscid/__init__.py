"""MtsCID external-code adapter with retained MIT attribution.

Model exports are loaded lazily so configuration and preprocessing remain
usable when optional PyTorch model dependencies are not installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from .config import (
    MtsCIDConfig,
    MtsCIDConfigError,
    config_from_mapping,
    load_mtscid_config,
)
_LAZY_EXPORTS = {
    "LastPointWindowDataset": (".data", "LastPointWindowDataset"),
    "MtsCIDDataError": (".data", "MtsCIDDataError"),
    "PreparedDataset": (".data", "PreparedDataset"),
    "PreparedSplit": (".data", "PreparedSplit"),
    "prepare_dataset": (".data", "prepare_dataset"),
    "build_model": (".model", "build_model"),
    "native_anomaly_energy": (".model", "native_anomaly_energy"),
    "MtsCIDRunner": (".runner", "MtsCIDRunner"),
    "MtsCIDRunnerError": (".runner", "MtsCIDRunnerError"),
    "TrainingHistory": (".runner", "TrainingHistory"),
    "run_mtscid": (".runner", "run_mtscid"),
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
    "LastPointWindowDataset",
    "MtsCIDConfig",
    "MtsCIDConfigError",
    "MtsCIDDataError",
    "MtsCIDRunner",
    "MtsCIDRunnerError",
    "PreparedDataset",
    "PreparedSplit",
    "TrainingHistory",
    "build_model",
    "config_from_mapping",
    "load_mtscid_config",
    "native_anomaly_energy",
    "prepare_dataset",
    "run_mtscid",
]
