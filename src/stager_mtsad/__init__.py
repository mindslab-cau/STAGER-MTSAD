"""Public interfaces for STAGER experiments."""

from .config import ManifestError, load_dataset_manifest
from .contracts import ContractError, DatasetSpec, RunResult

__all__ = [
    "ContractError",
    "DatasetSpec",
    "ManifestError",
    "RunResult",
    "load_dataset_manifest",
]

__version__ = "0.1.0"
