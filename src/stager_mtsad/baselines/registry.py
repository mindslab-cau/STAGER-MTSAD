"""Discovery-only metadata without importing optional model dependencies.

The registry does not pretend that heterogeneous model families share one
configuration or execution signature. Callers select a family here, then use
that module's typed configuration and runner API.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType


class BaselineRegistryError(KeyError):
    """Raised when a baseline name is unknown."""


@dataclass(frozen=True, slots=True)
class BaselineInfo:
    """Static import metadata for one runnable family."""

    name: str
    module: str
    integration: str
    dependency_group: str


_ENTRIES = (
    BaselineInfo(
        "knn",
        "stager_mtsad.baselines.classical",
        "native",
        "classical",
    ),
    BaselineInfo(
        "ocsvm",
        "stager_mtsad.baselines.classical",
        "native",
        "classical",
    ),
    BaselineInfo(
        "tsmixer",
        "stager_mtsad.baselines.tsmixer",
        "native",
        "torch",
    ),
    BaselineInfo(
        "anomaly_transformer",
        "stager_mtsad.baselines.anomaly_transformer",
        "native",
        "torch",
    ),
    BaselineInfo(
        "usad",
        "stager_mtsad.baselines.usad",
        "native",
        "torch",
    ),
    BaselineInfo(
        "fusagnet",
        "stager_mtsad.baselines.fusagnet",
        "native",
        "graph",
    ),
    BaselineInfo(
        "deep_svdd",
        "stager_mtsad.baselines.deep_svdd",
        "preserved_source_native_adapter",
        "torch",
    ),
    BaselineInfo(
        "mtscid",
        "stager_mtsad.baselines.mtscid",
        "external_adapter",
        "mtscid",
    ),
    BaselineInfo(
        "mshtrans",
        "stager_mtsad.baselines.mshtrans",
        "external_adapter",
        "mshtrans",
    ),
)

BASELINES = {entry.name: entry for entry in _ENTRIES}
_ALIASES = {
    "anomaly-transformer": "anomaly_transformer",
    "anomalytransformer": "anomaly_transformer",
    "deepsvdd": "deep_svdd",
    "deep-svdd": "deep_svdd",
    "knn": "knn",
    "msh-trans": "mshtrans",
    "mts-cid": "mtscid",
    "oneclasssvm": "ocsvm",
    "one-class-svm": "ocsvm",
    "oc-svm": "ocsvm",
    "ts-mixer": "tsmixer",
}


def _canonical_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return _ALIASES.get(normalized, normalized)


def get_baseline_info(name: str) -> BaselineInfo:
    """Return static information for a canonical name or public alias."""

    canonical = _canonical_name(name)
    try:
        return BASELINES[canonical]
    except KeyError as exc:
        available = ", ".join(BASELINES)
        raise BaselineRegistryError(
            f"unknown baseline {name!r}; available: {available}"
        ) from exc


def import_baseline(name: str) -> ModuleType:
    """Import a selected module; this is discovery, not a generic launcher."""

    return importlib.import_module(get_baseline_info(name).module)
