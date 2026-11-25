"""Validated configuration for the time-series Deep SVDD baseline."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class DeepSVDDConfigError(ValueError):
    """Raised when a Deep SVDD configuration is invalid."""


@dataclass(frozen=True, slots=True)
class DeepSVDDConfig:
    """Training, model, and preprocessing settings for one run."""

    window_size: int = 60
    stride: int = 1
    predict_size: int = 1
    rep_dim: int = 128
    objective: str = "soft-boundary"
    nu: float = 0.1
    learning_rate: float = 1e-4
    num_epochs: int = 50
    learning_rate_milestones: tuple[int, ...] = ()
    batch_size: int = 32
    weight_decay: float = 1e-5
    optimizer: str = "AdamW"
    warm_up_epochs: int = 10
    pretrain: bool = True
    autoencoder_num_epochs: int = 100
    autoencoder_learning_rate: float = 1e-4
    normalize: bool = True
    ewm_alpha: float | None = 0.8
    seed: int = 42
    evaluation_fpr: float = 0.002

    def __post_init__(self) -> None:
        for field_name in (
            "window_size",
            "stride",
            "predict_size",
            "rep_dim",
            "num_epochs",
            "batch_size",
            "autoencoder_num_epochs",
        ):
            if getattr(self, field_name) < 1:
                raise DeepSVDDConfigError(
                    f"{field_name} must be positive"
                )
        if self.predict_size != 1:
            raise DeepSVDDConfigError(
                "predict_size must be 1 for the legacy next-point anchor"
            )
        if self.window_size < 8:
            raise DeepSVDDConfigError(
                "window_size must be at least 8 for three pooling stages"
            )
        if self.objective not in {"one-class", "soft-boundary"}:
            raise DeepSVDDConfigError(
                "objective must be 'one-class' or 'soft-boundary'"
            )
        if not 0.0 < float(self.nu) <= 1.0:
            raise DeepSVDDConfigError("nu must be in (0, 1]")
        if float(self.learning_rate) <= 0.0:
            raise DeepSVDDConfigError(
                "learning_rate must be positive"
            )
        if float(self.autoencoder_learning_rate) <= 0.0:
            raise DeepSVDDConfigError(
                "autoencoder_learning_rate must be positive"
            )
        if float(self.weight_decay) < 0.0:
            raise DeepSVDDConfigError(
                "weight_decay must be non-negative"
            )
        if self.optimizer.lower() not in {"adam", "adamw", "amsgrad"}:
            raise DeepSVDDConfigError(
                "optimizer must be 'Adam', 'AdamW', or 'AMSGrad'"
            )
        if self.warm_up_epochs < 0:
            raise DeepSVDDConfigError(
                "warm_up_epochs must be non-negative"
            )
        if any(
            not isinstance(value, int) or value < 1
            for value in self.learning_rate_milestones
        ):
            raise DeepSVDDConfigError(
                "learning_rate_milestones must contain positive integers"
            )
        if self.pretrain and self.batch_size < 2:
            raise DeepSVDDConfigError(
                "batch_size must be at least 2 when pretraining is enabled"
            )
        if self.ewm_alpha is not None and not (
            0.0 < float(self.ewm_alpha) <= 1.0
        ):
            raise DeepSVDDConfigError(
                "ewm_alpha must be in (0, 1] or null"
            )
        if self.seed < 0:
            raise DeepSVDDConfigError("seed must be non-negative")
        if float(self.evaluation_fpr) != 0.002:
            raise DeepSVDDConfigError(
                "evaluation_fpr is fixed at 0.002 for STAGER"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the full recorded configuration and score semantics."""

        document = asdict(self)
        document["learning_rate_milestones"] = list(
            self.learning_rate_milestones
        )
        document["score"] = {
            "one-class": "squared_distance_to_center",
            "soft-boundary": (
                "squared_distance_to_center_minus_squared_radius"
            ),
            "direction": "higher_is_more_anomalous",
            "anchor": "first_point_after_input_window",
        }
        return document


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DeepSVDDConfigError(f"{field_name} must be a mapping")
    return dict(value)


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DeepSVDDConfigError(
            f"{field_name} contains unknown fields: {', '.join(unknown)}"
        )


def config_from_mapping(
    document: Mapping[str, Any],
) -> DeepSVDDConfig:
    """Build a configuration from a nested serialized mapping."""

    root = _mapping(document, "configuration")
    model = _mapping(root.get("model", {}), "model")
    data = _mapping(root.get("data", {}), "data")
    training = _mapping(root.get("training", {}), "training")
    pretraining = _mapping(
        root.get("pretraining", {}), "pretraining"
    )
    preprocessing = _mapping(
        root.get("preprocessing", {}), "preprocessing"
    )
    evaluation = _mapping(root.get("evaluation", {}), "evaluation")
    _reject_unknown(
        root,
        {
            "preset",
            "dataset",
            "migration",
            "model",
            "data",
            "training",
            "pretraining",
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
        {"rep_dim", "objective", "nu"},
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
            "learning_rate",
            "num_epochs",
            "learning_rate_milestones",
            "batch_size",
            "weight_decay",
            "optimizer",
            "warm_up_epochs",
            "seed",
        },
        "training",
    )
    _reject_unknown(
        pretraining,
        {"enabled", "num_epochs", "learning_rate"},
        "pretraining",
    )
    _reject_unknown(
        preprocessing,
        {"normalize", "ewm_alpha"},
        "preprocessing",
    )
    _reject_unknown(evaluation, {"fpr"}, "evaluation")
    for metadata_name in ("preset", "dataset", "migration", "run"):
        if metadata_name in root:
            _mapping(root[metadata_name], metadata_name)
    milestones = training.get("learning_rate_milestones", ())
    return DeepSVDDConfig(
        window_size=data.get("window_size", root.get("window_size", 60)),
        stride=data.get("stride", 1),
        predict_size=data.get("predict_size", 1),
        rep_dim=model.get("rep_dim", 128),
        objective=model.get("objective", "soft-boundary"),
        nu=model.get("nu", 0.1),
        learning_rate=training.get("learning_rate", 1e-4),
        num_epochs=training.get("num_epochs", 50),
        learning_rate_milestones=tuple(milestones),
        batch_size=training.get("batch_size", 32),
        weight_decay=training.get("weight_decay", 1e-5),
        optimizer=training.get("optimizer", "AdamW"),
        warm_up_epochs=training.get("warm_up_epochs", 10),
        pretrain=pretraining.get("enabled", True),
        autoencoder_num_epochs=pretraining.get("num_epochs", 100),
        autoencoder_learning_rate=pretraining.get(
            "learning_rate", 1e-4
        ),
        normalize=preprocessing.get("normalize", True),
        ewm_alpha=preprocessing.get("ewm_alpha", 0.8),
        seed=training.get("seed", root.get("seed", 42)),
        evaluation_fpr=evaluation.get("fpr", 0.002),
    )


def load_deep_svdd_config(path: str | Path) -> DeepSVDDConfig:
    """Load JSON, TOML, or YAML without importing a training dependency."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            document = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        elif suffix == ".toml":
            with config_path.open("rb") as stream:
                document = tomllib.load(stream)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise DeepSVDDConfigError(
                    "YAML configuration requires the optional yaml dependency"
                ) from exc
            document = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
        else:
            raise DeepSVDDConfigError(
                "configuration must use .json, .toml, .yaml, or .yml"
            )
    except DeepSVDDConfigError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise DeepSVDDConfigError(
            f"failed to load Deep SVDD configuration: {config_path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise DeepSVDDConfigError(
            "configuration root must be a mapping"
        )
    return config_from_mapping(document)
