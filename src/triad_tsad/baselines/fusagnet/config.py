"""Portable, explicitly validated configuration for the FuSAGNet baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any


class FuSAGNetConfigError(ValueError):
    """Raised when a FuSAGNet configuration is incomplete or inconsistent."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise FuSAGNetConfigError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
        raise FuSAGNetConfigError(f"{name} must be a non-negative number")
    return float(value)


@dataclass(frozen=True, slots=True)
class ProcessGroup:
    """One process and the exact feature positions assigned to it."""

    name: str
    feature_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise FuSAGNetConfigError("process group names must be non-empty strings")
        if not self.feature_indices:
            raise FuSAGNetConfigError(
                f"process group {self.name!r} must contain at least one feature"
            )

        normalized: list[int] = []
        for index in self.feature_indices:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise FuSAGNetConfigError(
                    f"process group {self.name!r} contains a non-integer index"
                )
            if index < 0:
                raise FuSAGNetConfigError(
                    f"process group {self.name!r} contains a negative index"
                )
            normalized.append(int(index))

        if len(set(normalized)) != len(normalized):
            raise FuSAGNetConfigError(
                f"process group {self.name!r} contains duplicate indices"
            )
        object.__setattr__(self, "feature_indices", tuple(normalized))


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Feature order and process membership used by the process embeddings."""

    feature_order: tuple[str, ...]
    process_groups: tuple[ProcessGroup, ...]

    def __post_init__(self) -> None:
        if not self.feature_order:
            raise FuSAGNetConfigError("feature_order must not be empty")
        if any(not isinstance(name, str) or not name.strip() for name in self.feature_order):
            raise FuSAGNetConfigError(
                "feature_order must contain only non-empty strings"
            )
        if len(set(self.feature_order)) != len(self.feature_order):
            raise FuSAGNetConfigError("feature_order must not contain duplicates")
        if not self.process_groups:
            raise FuSAGNetConfigError("at least one process group is required")

        group_names = [group.name for group in self.process_groups]
        if len(set(group_names)) != len(group_names):
            raise FuSAGNetConfigError("process group names must be unique")

        assigned = [
            index
            for group in self.process_groups
            for index in group.feature_indices
        ]
        expected = set(range(len(self.feature_order)))
        assigned_set = set(assigned)
        if len(assigned_set) != len(assigned):
            raise FuSAGNetConfigError(
                "each feature index must belong to exactly one process group"
            )
        if assigned_set != expected:
            missing = sorted(expected - assigned_set)
            extra = sorted(assigned_set - expected)
            raise FuSAGNetConfigError(
                "process groups must cover feature_order exactly; "
                f"missing={missing}, out_of_range={extra}"
            )

    @property
    def num_features(self) -> int:
        return len(self.feature_order)

    def validate_observed_order(self, observed: Sequence[str]) -> None:
        """Reject reordered or missing columns before model execution."""

        actual = tuple(observed)
        if actual == self.feature_order:
            return

        if len(actual) != len(self.feature_order):
            raise FuSAGNetConfigError(
                "observed feature count does not match configured feature_order: "
                f"expected {len(self.feature_order)}, received {len(actual)}"
            )

        mismatch = next(
            index
            for index, (expected, received) in enumerate(
                zip(self.feature_order, actual, strict=True)
            )
            if expected != received
        )
        raise FuSAGNetConfigError(
            "observed feature order differs from the configured order at "
            f"position {mismatch}: expected {self.feature_order[mismatch]!r}, "
            f"received {actual[mismatch]!r}"
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FeatureLayout:
        try:
            raw_features = value["feature_order"]
            raw_groups = value["process_groups"]
        except KeyError as exc:
            raise FuSAGNetConfigError(
                f"feature layout is missing {exc.args[0]!r}"
            ) from exc

        if isinstance(raw_features, (str, bytes)) or not isinstance(
            raw_features, Sequence
        ):
            raise FuSAGNetConfigError("feature_order must be a sequence")
        if isinstance(raw_groups, (str, bytes)) or not isinstance(
            raw_groups, Sequence
        ):
            raise FuSAGNetConfigError("process_groups must be a sequence")

        groups: list[ProcessGroup] = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                raise FuSAGNetConfigError("each process group must be a mapping")
            try:
                name = raw_group["name"]
                indices = raw_group["feature_indices"]
            except KeyError as exc:
                raise FuSAGNetConfigError(
                    f"process group is missing {exc.args[0]!r}"
                ) from exc
            if isinstance(indices, (str, bytes)) or not isinstance(
                indices, Sequence
            ):
                raise FuSAGNetConfigError(
                    "process group feature_indices must be a sequence"
                )
            groups.append(ProcessGroup(str(name), tuple(indices)))

        return cls(tuple(raw_features), tuple(groups))


@dataclass(frozen=True, slots=True)
class FuSAGNetModelConfig:
    """Architecture settings for the migrated local implementation."""

    window_size: int = 60
    embedding_dim: int = 16
    output_hidden_dim: int = 16
    output_layers: int = 1
    topk: int = 10
    latent_size: int = 16
    autoencoder_layers: int = 2
    alpha: float = 0.5
    beta: float = 1.0
    process_rnn_layers: int = 3
    dropout: float = 0.2
    edge_set_count: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "window_size",
            "embedding_dim",
            "output_hidden_dim",
            "output_layers",
            "topk",
            "latent_size",
            "autoencoder_layers",
            "process_rnn_layers",
            "edge_set_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )

        if self.embedding_dim % 2:
            raise FuSAGNetConfigError(
                "embedding_dim must be even for the bidirectional process GRUs"
            )
        for field_name in ("alpha", "dropout"):
            value = _non_negative_float(getattr(self, field_name), field_name)
            if value > 1:
                raise FuSAGNetConfigError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "beta", _non_negative_float(self.beta, "beta"))


@dataclass(frozen=True, slots=True)
class FuSAGNetTrainConfig:
    """Offline training settings with no external tracking integration."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    patience: int = 10
    validation_fraction: float = 0.2
    noise_level: float = 0.1
    optimizer: str = "adamw"
    seed: int = 0
    deep_snapshot_best_state: bool = False

    def __post_init__(self) -> None:
        for field_name in ("batch_size", "epochs", "patience"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "learning_rate",
            _non_negative_float(self.learning_rate, "learning_rate"),
        )
        if self.learning_rate == 0:
            raise FuSAGNetConfigError("learning_rate must be greater than zero")
        object.__setattr__(
            self, "noise_level", _non_negative_float(self.noise_level, "noise_level")
        )
        validation_fraction = _non_negative_float(
            self.validation_fraction,
            "validation_fraction",
        )
        if validation_fraction >= 1:
            raise FuSAGNetConfigError(
                "validation_fraction must be in [0, 1)"
            )
        object.__setattr__(
            self,
            "validation_fraction",
            validation_fraction,
        )
        if self.optimizer.lower() not in {"adam", "adamw"}:
            raise FuSAGNetConfigError("optimizer must be either 'adam' or 'adamw'")
        object.__setattr__(self, "optimizer", self.optimizer.lower())
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise FuSAGNetConfigError("seed must be a non-negative integer")
        if self.seed < 0:
            raise FuSAGNetConfigError("seed must be a non-negative integer")
        object.__setattr__(self, "seed", int(self.seed))
        if not isinstance(self.deep_snapshot_best_state, bool):
            raise FuSAGNetConfigError(
                "deep_snapshot_best_state must be a boolean"
            )


@dataclass(frozen=True, slots=True)
class FuSAGNetConfig:
    """Complete FuSAGNet run configuration."""

    layout: FeatureLayout
    model: FuSAGNetModelConfig = field(default_factory=FuSAGNetModelConfig)
    training: FuSAGNetTrainConfig = field(default_factory=FuSAGNetTrainConfig)
    false_positive_rate: float = 0.002

    def __post_init__(self) -> None:
        fpr = _non_negative_float(self.false_positive_rate, "false_positive_rate")
        if fpr != 0.002:
            raise FuSAGNetConfigError(
                "FuSAGNet migration fixes false_positive_rate at 0.002"
            )
        if self.model.topk > self.layout.num_features:
            raise FuSAGNetConfigError(
                "topk must not exceed the configured feature count"
            )
        object.__setattr__(self, "false_positive_rate", fpr)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FuSAGNetConfig:
        if not isinstance(value, Mapping):
            raise FuSAGNetConfigError("configuration root must be a mapping")
        layout = FeatureLayout.from_mapping(value.get("layout", {}))
        model_value = value.get("model", {})
        training_value = value.get("training", {})
        if not isinstance(model_value, Mapping):
            raise FuSAGNetConfigError("model must be a mapping")
        if not isinstance(training_value, Mapping):
            raise FuSAGNetConfigError("training must be a mapping")
        return cls(
            layout=layout,
            model=FuSAGNetModelConfig(**dict(model_value)),
            training=FuSAGNetTrainConfig(**dict(training_value)),
            false_positive_rate=value.get("false_positive_rate", 0.002),
        )


def load_fusagnet_config(path: str | Path) -> FuSAGNetConfig:
    """Load JSON, TOML, or optional YAML configuration from a portable path."""

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
                raise FuSAGNetConfigError(
                    "YAML configuration requires the optional PyYAML dependency"
                ) from exc
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raise FuSAGNetConfigError(
                "FuSAGNet configuration must use .json, .toml, .yaml, or .yml"
            )
    except OSError as exc:
        raise FuSAGNetConfigError(
            f"cannot read FuSAGNet configuration: {config_path}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise FuSAGNetConfigError("configuration root must be a mapping")
    return FuSAGNetConfig.from_mapping(raw)
