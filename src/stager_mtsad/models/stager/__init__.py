"""STAGER model, prepared-array runner, and typed configuration."""

from .config import StagerConfig
from .data import NextPointDataset
from .model import STAGER
from .runner import StagerExecution, fit_score_stager_datasets, run_stager
from .trainer import StagerTrainer

__all__ = [
    "NextPointDataset",
    "STAGER",
    "StagerConfig",
    "StagerExecution",
    "StagerTrainer",
    "fit_score_stager_datasets",
    "run_stager",
]
