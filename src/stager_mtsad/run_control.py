"""Shared CLI-only controls for seeds and bounded smoke-test runs."""

from __future__ import annotations

import argparse
import secrets
from dataclasses import fields, is_dataclass, replace
from numbers import Integral
from typing import Any


MAX_SEED = 2**32 - 1

# These fields directly bound training loops in the migrated runners. Warm-up
# and patience values are deliberately not changed: once the enclosing loop is
# capped, they cannot increase the number of executed epochs.
EPOCH_LOOP_FIELDS = frozenset(
    {
        "epochs",
        "num_epochs",
        "autoencoder_num_epochs",
        "pretrain_epochs",
        "pretraining_epochs",
        "train_epochs",
        "n_epochs",
    }
)


def add_sweep_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plan", action="store_true",
        help="Write the sweep plan without loading data or training.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a sweep and skip completed seeds.",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="Continue the sweep after a failed combination.",
    )


def positive_count(value: Any, name: str) -> int:
    """Validate a positive integer count and return a plain ``int``."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def generate_unique_seeds(count: int) -> tuple[int, ...]:
    """Generate unique unsigned 32-bit seeds from the operating system RNG."""

    requested = positive_count(count, "seed_iter")
    if requested > MAX_SEED + 1:
        raise ValueError("seed_iter exceeds the 32-bit seed space")
    selected: set[int] = set()
    ordered: list[int] = []
    while len(ordered) < requested:
        candidate = secrets.randbits(32)
        if candidate in selected:
            continue
        selected.add(candidate)
        ordered.append(candidate)
    return tuple(ordered)


def cap_epoch_loops(config: Any, maximum: int) -> Any:
    """Recursively cap dataclass fields that directly drive epoch loops.

    Baseline and STAGER configurations are frozen dataclasses. Returning
    replacements keeps the preset immutable and makes the smoke-test override
    visible in each run's recorded effective configuration.
    """

    limit = positive_count(maximum, "max_epochs")
    if not is_dataclass(config) or isinstance(config, type):
        return config

    updates: dict[str, Any] = {}
    for item in fields(config):
        original = getattr(config, item.name)
        if item.name in EPOCH_LOOP_FIELDS:
            if (
                isinstance(original, bool)
                or not isinstance(original, Integral)
                or original < 1
            ):
                raise ValueError(
                    f"epoch loop field {item.name!r} must be a positive integer"
                )
            updated = min(int(original), limit)
        else:
            updated = cap_epoch_loops(original, limit)
        if updated is not original and updated != original:
            updates[item.name] = updated
    return replace(config, **updates) if updates else config


__all__ = [
    "EPOCH_LOOP_FIELDS",
    "MAX_SEED",
    "add_sweep_options",
    "cap_epoch_loops",
    "generate_unique_seeds",
    "positive_count",
]
