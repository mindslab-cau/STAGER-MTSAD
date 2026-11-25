"""Native TSMixer baseline with next-point anomaly scoring."""

from __future__ import annotations

import importlib
from typing import Any

from .config import (
    TSMixerConfig,
    TSMixerConfigError,
    config_from_mapping,
    load_tsmixer_config,
)

_LAZY_EXPORTS = {
    "NextPointDataset": (".data", "NextPointDataset"),
    "PreparedDataset": (".data", "PreparedDataset"),
    "PreparedSplit": (".data", "PreparedSplit"),
    "TSMixerDataError": (".data", "TSMixerDataError"),
    "next_point_anchors": (".data", "next_point_anchors"),
    "prepare_dataset": (".data", "prepare_dataset"),
    "TSMixer": (".model", "TSMixer"),
    "TSMixerModelError": (".model", "TSMixerModelError"),
    "build_model": (".model", "build_model"),
    "TSMixerRunner": (".runner", "TSMixerRunner"),
    "TSMixerRunnerError": (".runner", "TSMixerRunnerError"),
    "TrainingHistory": (".runner", "TrainingHistory"),
    "run_tsmixer": (".runner", "run_tsmixer"),
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
    "NextPointDataset",
    "PreparedDataset",
    "PreparedSplit",
    "TSMixer",
    "TSMixerConfig",
    "TSMixerConfigError",
    "TSMixerDataError",
    "TSMixerModelError",
    "TSMixerRunner",
    "TSMixerRunnerError",
    "TrainingHistory",
    "build_model",
    "config_from_mapping",
    "load_tsmixer_config",
    "next_point_anchors",
    "prepare_dataset",
    "run_tsmixer",
]
