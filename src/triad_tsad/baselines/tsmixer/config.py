"""Validated, portable configuration for the native TSMixer runner."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class TSMixerConfigError(ValueError):
    """Raised when a TSMixer configuration is invalid."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TSMixerConfigError(f"{field_name} must be a mapping")
    return dict(value)


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TSMixerConfigError(
            f"{field_name} contains unknown fields: {', '.join(unknown)}"
        )


@dataclass(frozen=True, slots=True)
class TSMixerConfig:
    """Training, preprocessing, and model settings for one TSMixer run."""

    input_dim: int | None = None
    window_size: int = 60
    stride: int = 1
    pred_len: int = 1
    norm_type: str = "L"
    activation: str = "relu"
    n_block: int = 2
    dropout: float = 0.1
    ff_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    patience: int = 10
    best_state_policy: str = "deep_snapshot"
    validation_fraction: float = 0.2
    optimizer: str = "AdamW"
    seed: int = 0
    normalize: bool = True
    ewm_alpha: float | None = 0.8
    evaluation_fpr: float = 0.002

    def __post_init__(self) -> None:
        if self.input_dim is not None and self.input_dim < 1:
            raise TSMixerConfigError("input_dim must be positive or null")
        for field_name in (
            "window_size",
            "stride",
            "pred_len",
            "n_block",
            "ff_dim",
            "batch_size",
            "num_epochs",
            "patience",
        ):
            if getattr(self, field_name) < 1:
                raise TSMixerConfigError(f"{field_name} must be positive")
        if self.pred_len != 1:
            raise TSMixerConfigError(
                "pred_len must be 1 to preserve the next-point score anchor"
            )
        if self.norm_type not in {"L", "B"}:
            raise TSMixerConfigError("norm_type must be 'L' or 'B'")
        if self.activation.lower() not in {
            "relu",
            "gelu",
            "identity",
            "linear",
            "none",
        }:
            raise TSMixerConfigError(
                "activation must be 'relu', 'gelu', or 'identity'"
            )
        if not 0.0 <= float(self.dropout) < 1.0:
            raise TSMixerConfigError("dropout must be in [0, 1)")
        if float(self.learning_rate) <= 0.0:
            raise TSMixerConfigError("learning_rate must be positive")
        if not 0.0 <= float(self.validation_fraction) < 1.0:
            raise TSMixerConfigError(
                "validation_fraction must be in [0, 1)"
            )
        if self.optimizer.lower() not in {"adam", "adamw"}:
            raise TSMixerConfigError("optimizer must be 'Adam' or 'AdamW'")
        if self.best_state_policy not in {
            "legacy_shallow",
            "deep_snapshot",
        }:
            raise TSMixerConfigError(
                "best_state_policy must be 'legacy_shallow' or "
                "'deep_snapshot'"
            )
        if self.seed < 0:
            raise TSMixerConfigError("seed must be non-negative")
        if not isinstance(self.normalize, bool):
            raise TSMixerConfigError("normalize must be boolean")
        if self.ewm_alpha is not None and not (
            0.0 < float(self.ewm_alpha) <= 1.0
        ):
            raise TSMixerConfigError("ewm_alpha must be in (0, 1] or null")
        if float(self.evaluation_fpr) != 0.002:
            raise TSMixerConfigError("evaluation_fpr is fixed at 0.002")

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration recorded in ``RunResult``."""

        document = asdict(self)
        document["target_slice"] = None
        document["score"] = {
            "feature_score": "squared_prediction_error",
            "evaluation_input": "feature_streams_transposed",
            "point_aggregation": "diagnostic_mean_across_features_only",
            "feature_orientation": "[time, feature]",
            "anchor": "first_point_after_input_window",
        }
        return document


def config_from_mapping(document: Mapping[str, Any]) -> TSMixerConfig:
    """Build a TSMixer configuration from a nested serialized mapping."""

    root = _mapping(document, "configuration")
    if "fixed" in root or "grid" in root:
        raise TSMixerConfigError(
            "sweep documents cannot be loaded as one run; select one value "
            "from every grid entry and merge it into fixed first"
        )
    model = _mapping(root.get("model", {}), "model")
    data = _mapping(root.get("data", {}), "data")
    training = _mapping(root.get("training", {}), "training")
    preprocessing = _mapping(
        root.get("preprocessing", {}), "preprocessing"
    )
    _reject_unknown(
        root,
        {
            "preset",
            "dataset",
            "model",
            "data",
            "training",
            "preprocessing",
            "evaluation",
            "seed",
            "window_size",
            "run",
        },
        "configuration",
    )
    _reject_unknown(
        model,
        {
            "input_dim",
            "input_len",
            "pred_len",
            "target_slice",
            "norm_type",
            "activation",
            "n_block",
            "dropout",
            "ff_dim",
        },
        "model",
    )
    _reject_unknown(
        data,
        {"window_size", "stride", "predict_size"},
        "data",
    )
    _reject_unknown(
        training,
        {
            "batch_size",
            "learning_rate",
            "num_epochs",
            "patience",
            "best_state_policy",
            "validation_fraction",
            "optimizer",
            "seed",
        },
        "training",
    )
    _reject_unknown(
        preprocessing,
        {"normalize", "ewm_alpha"},
        "preprocessing",
    )
    evaluation = _mapping(root.get("evaluation", {}), "evaluation")
    _reject_unknown(evaluation, {"fpr"}, "evaluation")
    for metadata_name in ("preset", "dataset", "run"):
        if metadata_name in root:
            _mapping(root[metadata_name], metadata_name)

    if model.get("target_slice") is not None:
        raise TSMixerConfigError(
            "model.target_slice must be null for all-feature scoring"
        )

    return TSMixerConfig(
        input_dim=model.get("input_dim"),
        window_size=data.get(
            "window_size", model.get("input_len", root.get("window_size", 60))
        ),
        stride=data.get("stride", 1),
        pred_len=model.get("pred_len", data.get("predict_size", 1)),
        norm_type=model.get("norm_type", "L"),
        activation=model.get("activation", "relu"),
        n_block=model.get("n_block", 2),
        dropout=model.get("dropout", 0.1),
        ff_dim=model.get("ff_dim", 64),
        batch_size=training.get("batch_size", 32),
        learning_rate=training.get("learning_rate", 1e-4),
        num_epochs=training.get("num_epochs", 50),
        patience=training.get("patience", 10),
        best_state_policy=training.get(
            "best_state_policy", "deep_snapshot"
        ),
        validation_fraction=training.get("validation_fraction", 0.2),
        optimizer=training.get("optimizer", "AdamW"),
        seed=training.get("seed", root.get("seed", 0)),
        normalize=preprocessing.get("normalize", True),
        ewm_alpha=preprocessing.get("ewm_alpha", 0.8),
        evaluation_fpr=evaluation.get("fpr", 0.002),
    )


def load_tsmixer_config(path: str | Path) -> TSMixerConfig:
    """Load a JSON, TOML, or YAML TSMixer configuration."""

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
                raise TSMixerConfigError(
                    "YAML configuration requires the optional yaml dependency"
                ) from exc
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raise TSMixerConfigError(
                "configuration must use .json, .toml, .yaml, or .yml"
            )
    except TSMixerConfigError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise TSMixerConfigError(
            f"failed to load TSMixer configuration: {config_path}"
        ) from exc

    if not isinstance(document, Mapping):
        raise TSMixerConfigError("configuration root must be a mapping")
    return config_from_mapping(document)
