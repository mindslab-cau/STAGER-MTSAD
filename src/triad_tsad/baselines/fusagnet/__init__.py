"""FuSAGNet migration held for local use pending upstream licensing."""

from __future__ import annotations

from typing import Any

from .config import (
    FeatureLayout,
    FuSAGNetConfig,
    FuSAGNetConfigError,
    FuSAGNetModelConfig,
    FuSAGNetTrainConfig,
    ProcessGroup,
    load_fusagnet_config,
)

__all__ = [
    "FeatureLayout",
    "FuSAGNet",
    "FuSAGNetConfig",
    "FuSAGNetConfigError",
    "FuSAGNetModelConfig",
    "FuSAGNetRunner",
    "FuSAGNetTrainConfig",
    "NextPointWindows",
    "ProcessGroup",
    "load_fusagnet_config",
    "make_next_point_windows",
]


def __getattr__(name: str) -> Any:
    """Delay NumPy, PyTorch, and PyG imports until the interface is requested."""

    if name in {"NextPointWindows", "make_next_point_windows"}:
        from .data import NextPointWindows, make_next_point_windows

        return {
            "NextPointWindows": NextPointWindows,
            "make_next_point_windows": make_next_point_windows,
        }[name]
    if name == "FuSAGNet":
        from .model import FuSAGNet

        return FuSAGNet
    if name == "FuSAGNetRunner":
        from .runner import FuSAGNetRunner

        return FuSAGNetRunner
    raise AttributeError(name)
