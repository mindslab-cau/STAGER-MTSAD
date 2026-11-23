"""Baseline implementations and adapters.

The registry is dependency-light. Individual model packages are imported only
when a caller explicitly requests them.
"""

from .registry import (
    BASELINES,
    BaselineInfo,
    BaselineRegistryError,
    get_baseline_info,
    import_baseline,
)

__all__ = [
    "BASELINES",
    "BaselineInfo",
    "BaselineRegistryError",
    "get_baseline_info",
    "import_baseline",
]
