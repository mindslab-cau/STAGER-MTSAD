"""Portable configuration for the local-only MSHTrans migration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any


class MSHTransConfigError(ValueError):
    """Raised when an MSHTrans configuration is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise MSHTransConfigError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise MSHTransConfigError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise MSHTransConfigError(f"{name} must be a positive number")
    return float(value)


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MSHTransConfigError(f"{name} must be a sequence of positive integers")
    normalized = tuple(_positive_int(item, name) for item in value)
    if not normalized:
        raise MSHTransConfigError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class MSHTransModelConfig:
    """Architecture settings retained from the adapted native baseline."""

    window_size: int = 100
    pool_sizes: tuple[int, ...] = (2, 2)
    hyperedge_counts: tuple[int, ...] = (50, 30, 20)
    head_count: int = 3
    inner_size: int = 5
    d_model: int = 64
    topk: int = 5
    update_rate: float = 0.3
    moving_average_windows: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "window_size",
            "head_count",
            "inner_size",
            "d_model",
            "topk",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )

        pools = _positive_int_tuple(self.pool_sizes, "pool_sizes")
        hyperedges = _positive_int_tuple(
            self.hyperedge_counts,
            "hyperedge_counts",
        )
        object.__setattr__(self, "pool_sizes", pools)
        object.__setattr__(self, "hyperedge_counts", hyperedges)
        if self.moving_average_windows is not None:
            object.__setattr__(
                self,
                "moving_average_windows",
                _positive_int_tuple(
                    self.moving_average_windows, "moving_average_windows"
                ),
            )

        if len(pools) != len(hyperedges) - 1:
            raise MSHTransConfigError(
                "pool_sizes must contain one fewer entry than hyperedge_counts"
            )

        scale_length = self.window_size
        for pool_size in pools:
            scale_length //= pool_size
        if scale_length < 6:
            raise MSHTransConfigError(
                "the coarsest temporal scale must contain at least 6 points; "
                "increase window_size or reduce pooling"
            )

        if isinstance(self.update_rate, bool) or not isinstance(
            self.update_rate, Real
        ):
            raise MSHTransConfigError("update_rate must be a number")
        update_rate = float(self.update_rate)
        if not 0 <= update_rate <= 1:
            raise MSHTransConfigError("update_rate must be between 0 and 1")
        object.__setattr__(self, "update_rate", update_rate)

    @property
    def scale_count(self) -> int:
        return len(self.hyperedge_counts)


@dataclass(frozen=True, slots=True)
class MSHTransTrainingConfig:
    """Offline training settings without external tracking or parallel mode."""

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    patience: int = 10
    validation_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 1

    def __post_init__(self) -> None:
        for field_name in ("batch_size", "epochs"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )
        for field_name in ("patience", "seed", "num_workers"):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "learning_rate",
            _positive_float(self.learning_rate, "learning_rate"),
        )
        if isinstance(self.validation_fraction, bool) or not isinstance(
            self.validation_fraction, Real
        ):
            raise MSHTransConfigError(
                "validation_fraction must be a number"
            )
        validation_fraction = float(self.validation_fraction)
        if not 0 <= validation_fraction < 1:
            raise MSHTransConfigError(
                "validation_fraction must be in [0, 1)"
            )
        object.__setattr__(
            self,
            "validation_fraction",
            validation_fraction,
        )


@dataclass(frozen=True, slots=True)
class MSHTransConfig:
    """Complete MSHTrans run configuration.

    The paper-table artifacts used train stride 10 and test stride 1. The two
    values remain separate so a stride-one ablation cannot be mistaken for the
    table-generating setup. Only the shared FPR value of 0.002 is exposed.
    Threshold selection is handled by the common evaluator, not by the model
    runner.
    """

    model: MSHTransModelConfig = field(default_factory=MSHTransModelConfig)
    training: MSHTransTrainingConfig = field(
        default_factory=MSHTransTrainingConfig
    )
    train_stride: int = 10
    test_stride: int = 1
    false_positive_rate: float = 0.002

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "train_stride",
            _positive_int(self.train_stride, "train_stride"),
        )
        test_stride = _positive_int(self.test_stride, "test_stride")
        if test_stride != 1:
            raise MSHTransConfigError(
                "MSHTrans evaluation fixes test_stride at 1"
            )
        object.__setattr__(self, "test_stride", test_stride)

        if isinstance(self.false_positive_rate, bool) or not isinstance(
            self.false_positive_rate, Real
        ):
            raise MSHTransConfigError("false_positive_rate must be a number")
        fpr = float(self.false_positive_rate)
        if fpr != 0.002:
            raise MSHTransConfigError(
                "MSHTrans migration fixes false_positive_rate at 0.002"
            )
        object.__setattr__(self, "false_positive_rate", fpr)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MSHTransConfig:
        if not isinstance(value, Mapping):
            raise MSHTransConfigError("configuration root must be a mapping")

        model_value = value.get("model", {})
        training_value = value.get("training", {})
        if not isinstance(model_value, Mapping):
            raise MSHTransConfigError("model must be a mapping")
        if not isinstance(training_value, Mapping):
            raise MSHTransConfigError("training must be a mapping")

        try:
            return cls(
                model=MSHTransModelConfig(**dict(model_value)),
                training=MSHTransTrainingConfig(**dict(training_value)),
                train_stride=value.get(
                    "train_stride",
                    value.get("stride", 10),
                ),
                test_stride=value.get("test_stride", 1),
                false_positive_rate=value.get("false_positive_rate", 0.002),
            )
        except TypeError as exc:
            raise MSHTransConfigError(
                f"unknown or missing MSHTrans configuration field: {exc}"
            ) from exc


def load_mshtrans_config(path: str | Path) -> MSHTransConfig:
    """Load JSON, TOML, or optional YAML configuration."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            import json

            raw = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            import tomllib

            with config_path.open("rb") as stream:
                raw = tomllib.load(stream)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise MSHTransConfigError(
                    "YAML configuration requires the optional 'yaml' dependency"
                ) from exc
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raise MSHTransConfigError(
                "configuration must use a .json, .toml, .yaml, or .yml extension"
            )
    except MSHTransConfigError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise MSHTransConfigError(
            f"failed to read MSHTrans configuration: {config_path}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise MSHTransConfigError("configuration root must be a mapping")
    return MSHTransConfig.from_mapping(raw)
