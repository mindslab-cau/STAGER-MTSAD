"""Validated portable configuration for the native USAD runner."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class USADConfigError(ValueError):
    """Raised when a USAD configuration is invalid."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise USADConfigError(f"{field_name} must be a mapping")
    return dict(value)


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise USADConfigError(
            f"{field_name} contains unknown fields: {', '.join(unknown)}"
        )


@dataclass(frozen=True, slots=True)
class USADConfig:
    """Model, training, score, and preprocessing settings."""

    num_channels: int | None = None
    window_size: int = 60
    stride: int = 1
    latent_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    num_epochs: int = 50
    warmup_epochs: int = 5
    validation_warmup_epochs: int = 10
    learning_rate_ratio_after_warmup: float = 0.5
    patience: int = 10
    delta_threshold: float = 1e-4
    validation_fraction: float = 0.2
    best_state_policy: str = "deep_delta_snapshot"
    seed: int = 0
    normalize: bool = True
    ewm_alpha: float | None = 0.8
    evaluation_fpr: float = 0.002

    def __post_init__(self) -> None:
        if self.num_channels is not None and self.num_channels < 1:
            raise USADConfigError("num_channels must be positive or null")
        for field_name in (
            "window_size",
            "stride",
            "latent_dim",
            "batch_size",
            "num_epochs",
            "patience",
        ):
            if getattr(self, field_name) < 1:
                raise USADConfigError(f"{field_name} must be positive")
        if self.warmup_epochs < 0:
            raise USADConfigError("warmup_epochs must be non-negative")
        if self.validation_warmup_epochs < 0:
            raise USADConfigError(
                "validation_warmup_epochs must be non-negative"
            )
        if float(self.learning_rate) <= 0.0:
            raise USADConfigError("learning_rate must be positive")
        if self.optimizer.lower() not in {"adam", "adamw"}:
            raise USADConfigError("optimizer must be 'Adam' or 'AdamW'")
        if not 0.0 < float(self.learning_rate_ratio_after_warmup) <= 1.0:
            raise USADConfigError(
                "learning_rate_ratio_after_warmup must be in (0, 1]"
            )
        if float(self.delta_threshold) < 0.0:
            raise USADConfigError("delta_threshold must be non-negative")
        if not 0.0 <= float(self.validation_fraction) < 1.0:
            raise USADConfigError(
                "validation_fraction must be in [0, 1)"
            )
        if self.best_state_policy not in {
            "legacy_shallow_delta",
            "deep_delta_snapshot",
        }:
            raise USADConfigError(
                "best_state_policy must be 'legacy_shallow_delta' or "
                "'deep_delta_snapshot'"
            )
        if self.seed < 0:
            raise USADConfigError("seed must be non-negative")
        if not isinstance(self.normalize, bool):
            raise USADConfigError("normalize must be boolean")
        if self.ewm_alpha is not None and not (
            0.0 < float(self.ewm_alpha) <= 1.0
        ):
            raise USADConfigError("ewm_alpha must be in (0, 1] or null")
        if float(self.evaluation_fpr) != 0.002:
            raise USADConfigError("evaluation_fpr is fixed at 0.002")

    def as_dict(self) -> dict[str, Any]:
        """Return the complete reproducibility record for ``RunResult``."""

        document = asdict(self)
        document["score"] = {
            "feature_score": "decoder1_window_mean_squared_error",
            "legacy_feature_orientation": "[feature, time]",
            "canonical_feature_orientation": "[time, feature]",
            "evaluation_input": "feature_streams_transposed",
            "point_aggregation": "diagnostic_mean_across_features_only",
            "anchor": "first_point_after_input_window",
        }
        return document


def config_from_mapping(document: Mapping[str, Any]) -> USADConfig:
    """Build a configuration from a nested JSON/TOML/YAML mapping."""

    root = _mapping(document, "configuration")
    if "fixed" in root or "grid" in root:
        raise USADConfigError(
            "sweep documents cannot be loaded as one run; select one value "
            "from every grid entry and merge it into fixed first"
        )
    model = _mapping(root.get("model", {}), "model")
    data = _mapping(root.get("data", {}), "data")
    training = _mapping(root.get("training", {}), "training")
    preprocessing = _mapping(
        root.get("preprocessing", {}), "preprocessing"
    )
    score = _mapping(root.get("score", {}), "score")
    _reject_unknown(
        root,
        {
            "preset",
            "dataset",
            "model",
            "data",
            "training",
            "preprocessing",
            "score",
            "evaluation",
            "seed",
            "window_size",
            "run",
        },
        "configuration",
    )
    _reject_unknown(
        model,
        {"num_channels", "window_size", "latent_dim", "z_size"},
        "model",
    )
    _reject_unknown(data, {"window_size", "stride"}, "data")
    _reject_unknown(
        training,
        {
            "batch_size",
            "learning_rate",
            "optimizer",
            "num_epochs",
            "warmup_epochs",
            "validation_warmup_epochs",
            "learning_rate_ratio_after_warmup",
            "lr_ratio_after_warmup",
            "patience",
            "delta_threshold",
            "validation_fraction",
            "best_state_policy",
            "seed",
        },
        "training",
    )
    _reject_unknown(
        preprocessing,
        {"normalize", "ewm_alpha"},
        "preprocessing",
    )
    _reject_unknown(score, {"definition"}, "score")
    evaluation = _mapping(root.get("evaluation", {}), "evaluation")
    _reject_unknown(evaluation, {"fpr"}, "evaluation")
    for metadata_name in ("preset", "dataset", "run"):
        if metadata_name in root:
            _mapping(root[metadata_name], metadata_name)
    score_definition = score.get(
        "definition", "legacy_decoder1_window_mse"
    )
    if score_definition != "legacy_decoder1_window_mse":
        raise USADConfigError(
            "score.definition must be 'legacy_decoder1_window_mse'"
        )
    return USADConfig(
        num_channels=model.get("num_channels"),
        window_size=data.get(
            "window_size", model.get("window_size", root.get("window_size", 60))
        ),
        stride=data.get("stride", 1),
        latent_dim=model.get("latent_dim", model.get("z_size", 128)),
        batch_size=training.get("batch_size", 32),
        learning_rate=training.get("learning_rate", 1e-4),
        optimizer=training.get("optimizer", "AdamW"),
        num_epochs=training.get("num_epochs", 50),
        warmup_epochs=training.get("warmup_epochs", 5),
        validation_warmup_epochs=training.get(
            "validation_warmup_epochs",
            10,
        ),
        learning_rate_ratio_after_warmup=training.get(
            "learning_rate_ratio_after_warmup",
            training.get("lr_ratio_after_warmup", 0.5),
        ),
        patience=training.get("patience", 10),
        delta_threshold=training.get("delta_threshold", 1e-4),
        validation_fraction=training.get("validation_fraction", 0.2),
        best_state_policy=training.get(
            "best_state_policy", "deep_delta_snapshot"
        ),
        seed=training.get("seed", root.get("seed", 0)),
        normalize=preprocessing.get("normalize", True),
        ewm_alpha=preprocessing.get("ewm_alpha", 0.8),
        evaluation_fpr=evaluation.get("fpr", 0.002),
    )


def load_usad_config(path: str | Path) -> USADConfig:
    """Load a JSON, TOML, or YAML configuration."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            document = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            with config_path.open("rb") as stream:
                document = tomllib.load(stream)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise USADConfigError(
                    "YAML configuration requires the optional yaml dependency"
                ) from exc
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raise USADConfigError(
                "configuration must use .json, .toml, .yaml, or .yml"
            )
    except USADConfigError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise USADConfigError(
            f"failed to load USAD configuration: {config_path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise USADConfigError("configuration root must be a mapping")
    return config_from_mapping(document)
