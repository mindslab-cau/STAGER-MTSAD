"""First-party model implementations."""

from .triad import (
    NextPointDataset,
    TriAD,
    TriADConfig,
    TriADTrainer,
    run_triad,
)

__all__ = [
    "NextPointDataset",
    "TriAD",
    "TriADConfig",
    "TriADTrainer",
    "run_triad",
]
