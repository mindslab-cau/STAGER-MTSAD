"""Native plain-USAD baseline with legacy project score semantics."""

from __future__ import annotations

import importlib
from typing import Any

from .config import (
    USADConfig,
    USADConfigError,
    config_from_mapping,
    load_usad_config,
)

_LAZY_EXPORTS = {
    "PreparedDataset": (".data", "PreparedDataset"),
    "PreparedSplit": (".data", "PreparedSplit"),
    "ReconstructionWindowDataset": (".data", "ReconstructionWindowDataset"),
    "USADDataError": (".data", "USADDataError"),
    "next_point_anchors": (".data", "next_point_anchors"),
    "prepare_dataset": (".data", "prepare_dataset"),
    "Decoder": (".model", "Decoder"),
    "Encoder": (".model", "Encoder"),
    "USAD": (".model", "USAD"),
    "USADModelError": (".model", "USADModelError"),
    "TrainingHistory": (".runner", "TrainingHistory"),
    "USADRunner": (".runner", "USADRunner"),
    "USADRunnerError": (".runner", "USADRunnerError"),
    "run_usad": (".runner", "run_usad"),
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
    "Decoder",
    "Encoder",
    "PreparedDataset",
    "PreparedSplit",
    "ReconstructionWindowDataset",
    "TrainingHistory",
    "USAD",
    "USADConfig",
    "USADConfigError",
    "USADDataError",
    "USADModelError",
    "USADRunner",
    "USADRunnerError",
    "config_from_mapping",
    "load_usad_config",
    "next_point_anchors",
    "prepare_dataset",
    "run_usad",
]
