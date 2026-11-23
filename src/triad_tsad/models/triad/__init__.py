"""TriAD model, prepared-array runner, and typed configuration."""

from .config import TriADConfig
from .data import NextPointDataset
from .model import TriAD
from .runner import TriADExecution, fit_score_triad_datasets, run_triad
from .trainer import TriADTrainer

__all__ = [
    "NextPointDataset",
    "TriAD",
    "TriADConfig",
    "TriADExecution",
    "TriADTrainer",
    "fit_score_triad_datasets",
    "run_triad",
]
