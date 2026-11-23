"""MSHTrans adapter with lazily loaded model dependencies."""

from .config import (
    MSHTransConfig,
    MSHTransConfigError,
    MSHTransModelConfig,
    MSHTransTrainingConfig,
    load_mshtrans_config,
)
from .data import (
    LastPointWindows,
    MSHTransDataError,
    make_last_point_windows,
)

__all__ = [
    "LastPointWindows",
    "MSHTransConfig",
    "MSHTransConfigError",
    "MSHTransDataError",
    "MSHTransModelConfig",
    "MSHTransRunner",
    "MSHTransTrainingConfig",
    "load_mshtrans_config",
    "make_last_point_windows",
]


def __getattr__(name: str):
    if name == "MSHTransRunner":
        from .runner import MSHTransRunner

        return MSHTransRunner
    raise AttributeError(name)
