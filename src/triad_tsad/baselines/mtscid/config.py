"""Validated, portable configuration for the MtsCID adapter."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class MtsCIDConfigError(ValueError):
    """Raised when an MtsCID configuration violates a locked contract."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MtsCIDConfigError(f"{field_name} must be a mapping")
    return dict(value)


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MtsCIDConfigError(
            f"{field_name} contains unknown fields: {', '.join(unknown)}"
        )


@dataclass(frozen=True, slots=True)
class MtsCIDConfig:
    """MtsCID model, training, preprocessing, and score settings.

    Training uses stride one. Test windows can use the last point or
    retain every point in non-overlapping windows.
    """

    window_size: int = 100
    stride: int = 1
    alignment: str = "last"
    d_model: int | None = None
    temperature: float = 0.1
    encoder_layers: int = 1
    branches_group_embedding: str = "False_False"
    multiscale_kernel_size: tuple[int, ...] = (5,)
    multiscale_patch_size: tuple[int, ...] = (10, 20)
    branch1_networks: tuple[str, ...] = (
        "fc_linear",
        "intra_fc_transformer",
        "multiscale_ts_attention",
    )
    branch1_match_dimension: str = "first"
    branch2_networks: tuple[str, ...] = (
        "multiscale_conv1d",
        "inter_fc_transformer",
    )
    branch2_match_dimension: str = "first"
    decoder_networks: tuple[str, ...] = ("linear",)
    decoder_layers: int = 1
    decoder_group_embedding: str = "False"
    embedding_init: str = "normal"
    memory_guided: str = "sinusoid"
    aggregation: str = "normal_mean"
    dropout: float = 0.3
    batch_size: int = 64
    num_epochs: int = 20
    patience: int = 10
    learning_rate: float = 2e-3
    weight_decay: float = 5e-5
    validation_fraction: float = 0.2
    alpha: float = 1.0
    seed: int = 0
    apply_ewm: bool = True
    ewm_alpha: float = 0.8
    continuity: str = "legacy_endpoint"
    evaluation_fpr: float = 0.002
    test_stride: int = 1

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise MtsCIDConfigError("window_size must be at least 2")
        if self.stride != 1:
            raise MtsCIDConfigError("MtsCID stride is fixed at 1")
        if (
            isinstance(self.test_stride, bool)
            or not isinstance(self.test_stride, int)
            or self.test_stride < 1
        ):
            raise MtsCIDConfigError("test_stride must be a positive integer")
        if self.alignment not in {"last", "flatten"} or (
            self.alignment == "flatten" and self.test_stride != self.window_size
        ):
            raise MtsCIDConfigError(
                "MtsCID alignment is fixed at 'last' unless 'flatten' uses "
                "test_stride equal to window_size"
            )
        if self.d_model is not None and self.d_model < 1:
            raise MtsCIDConfigError("d_model must be positive or null")
        for field_name in (
            "encoder_layers",
            "decoder_layers",
            "batch_size",
            "num_epochs",
            "patience",
        ):
            if getattr(self, field_name) < 1:
                raise MtsCIDConfigError(f"{field_name} must be positive")
        if float(self.temperature) <= 0.0:
            raise MtsCIDConfigError("temperature must be positive")
        if float(self.learning_rate) <= 0.0:
            raise MtsCIDConfigError("learning_rate must be positive")
        if float(self.weight_decay) < 0.0:
            raise MtsCIDConfigError("weight_decay must be non-negative")
        if not 0.0 <= float(self.validation_fraction) < 1.0:
            raise MtsCIDConfigError(
                "validation_fraction must be in [0, 1)"
            )
        if float(self.alpha) < 0.0:
            raise MtsCIDConfigError("alpha must be non-negative")
        if self.seed < 0:
            raise MtsCIDConfigError("seed must be non-negative")
        if not isinstance(self.apply_ewm, bool):
            raise MtsCIDConfigError("apply_ewm must be boolean")
        if not 0.0 < float(self.ewm_alpha) <= 1.0:
            raise MtsCIDConfigError("ewm_alpha must be in (0, 1]")
        if self.continuity not in {"legacy_endpoint", "none"}:
            raise MtsCIDConfigError(
                "continuity must be 'legacy_endpoint' or 'none'"
            )
        if float(self.evaluation_fpr) != 0.002:
            raise MtsCIDConfigError("evaluation_fpr is fixed at 0.002")
        if self.aggregation != "normal_mean":
            raise MtsCIDConfigError(
                "the migrated MtsCID preset currently supports only "
                "'normal_mean'"
            )
        if not self.multiscale_kernel_size or any(
            int(value) < 1 for value in self.multiscale_kernel_size
        ):
            raise MtsCIDConfigError(
                "multiscale_kernel_size must contain positive integers"
            )
        if not self.multiscale_patch_size or any(
            int(value) < 1 for value in self.multiscale_patch_size
        ):
            raise MtsCIDConfigError(
                "multiscale_patch_size must contain positive integers"
            )
        incompatible = [
            value
            for value in self.multiscale_patch_size
            if self.window_size % int(value) != 0
        ]
        if (
            "multiscale_ts_attention" in self.branch1_networks
            and incompatible
        ):
            raise MtsCIDConfigError(
                "window_size must be divisible by every multiscale patch "
                f"size; incompatible values: {incompatible}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible reproducibility record."""

        document = asdict(self)
        if self.test_stride == 1:
            document.pop("test_stride")
        for field_name in (
            "multiscale_kernel_size",
            "multiscale_patch_size",
            "branch1_networks",
            "branch2_networks",
            "decoder_networks",
        ):
            document[field_name] = list(document[field_name])
        document["score"] = {
            "definition": (
                "native_reconstruction_memory_harmonic_loss"
            ),
            "aggregation": self.aggregation,
            "anchor": (
                "all_points_of_each_window"
                if self.alignment == "flatten"
                else "last_point_of_each_window"
            ),
            "direction": "higher_is_more_anomalous",
        }
        return document

    def model_mapping(
        self,
        *,
        feature_count: int,
        device: Any,
    ) -> dict[str, Any]:
        """Build the mapping consumed by the preserved model constructor."""

        d_model = self.d_model or feature_count
        return {
            "win_size": self.window_size,
            "input_c": feature_count,
            "output_c": feature_count,
            "d_model": d_model,
            "temperature": self.temperature,
            "encoder_layers": self.encoder_layers,
            "branches_group_embedding": self.branches_group_embedding,
            "multiscale_kernel_size": list(self.multiscale_kernel_size),
            "multiscale_patch_size": list(self.multiscale_patch_size),
            "branch1_networks": list(self.branch1_networks),
            "branch1_match_dimension": self.branch1_match_dimension,
            "branch2_networks": list(self.branch2_networks),
            "branch2_match_dimension": self.branch2_match_dimension,
            "decoder_networks": list(self.decoder_networks),
            "decoder_layers": self.decoder_layers,
            "decoder_group_embedding": self.decoder_group_embedding,
            "embedding_init": self.embedding_init,
            "memory_guided": self.memory_guided,
            "device": device,
        }


def config_from_mapping(document: Mapping[str, Any]) -> MtsCIDConfig:
    """Build a configuration from a nested JSON/TOML/YAML mapping."""

    root = _mapping(document, "configuration")
    model = _mapping(root.get("model", {}), "model")
    data = _mapping(root.get("data", {}), "data")
    training = _mapping(root.get("training", {}), "training")
    preprocessing = _mapping(
        root.get("preprocessing", {}), "preprocessing"
    )
    score = _mapping(root.get("score", {}), "score")
    evaluation = _mapping(root.get("evaluation", {}), "evaluation")
    _reject_unknown(
        root,
        {
            "baseline",
            "migration",
            "model",
            "data",
            "training",
            "preprocessing",
            "score",
            "evaluation",
            "seed",
            "run",
        },
        "configuration",
    )
    _reject_unknown(
        model,
        {
            "window_size",
            "d_model",
            "temperature",
            "encoder_layers",
            "branches_group_embedding",
            "multiscale_kernel_size",
            "multiscale_patch_size",
            "branch1_networks",
            "branch1_match_dimension",
            "branch2_networks",
            "branch2_match_dimension",
            "decoder_networks",
            "decoder_layers",
            "decoder_group_embedding",
            "embedding_init",
            "memory_guided",
            "aggregation",
            "dropout",
        },
        "model",
    )
    _reject_unknown(
        data,
        {"window_size", "stride", "test_stride", "continuity"},
        "data",
    )
    _reject_unknown(
        training,
        {
            "batch_size",
            "num_epochs",
            "patience",
            "learning_rate",
            "weight_decay",
            "validation_fraction",
            "alpha",
            "seed",
        },
        "training",
    )
    _reject_unknown(
        preprocessing,
        {
            "missing_values",
            "normalization",
            "clip",
            "apply_ewm",
            "ewm_alpha",
            "ewm_adjust",
        },
        "preprocessing",
    )
    _reject_unknown(
        score,
        {"alignment", "aggregation", "direction"},
        "score",
    )
    _reject_unknown(evaluation, {"fpr"}, "evaluation")
    if root.get("baseline", "mtscid") != "mtscid":
        raise MtsCIDConfigError("baseline must be 'mtscid'")
    if "migration" in root:
        _mapping(root["migration"], "migration")
    if "run" in root:
        _mapping(root["run"], "run")
    if preprocessing.get("normalization", "train_minmax") != "train_minmax":
        raise MtsCIDConfigError(
            "preprocessing.normalization must be 'train_minmax'"
        )
    missing_value = preprocessing.get(
        "missing_values",
        (
            "interpolate",
            "backward_fill",
            "forward_fill",
            "zero_fill",
        ),
    )
    if isinstance(missing_value, (str, bytes)) or not isinstance(
        missing_value,
        Sequence,
    ):
        raise MtsCIDConfigError(
            "preprocessing.missing_values must be a sequence"
        )
    missing_policy = tuple(missing_value)
    if missing_policy != (
        "interpolate",
        "backward_fill",
        "forward_fill",
        "zero_fill",
    ):
        raise MtsCIDConfigError(
            "preprocessing.missing_values must preserve interpolate, "
            "backward_fill, forward_fill, zero_fill order"
        )
    if preprocessing.get("clip", False) is not False:
        raise MtsCIDConfigError("preprocessing.clip must be false")
    if preprocessing.get("ewm_adjust", True) is not True:
        raise MtsCIDConfigError("preprocessing.ewm_adjust must be true")
    if score.get("direction", "higher_is_more_anomalous") != (
        "higher_is_more_anomalous"
    ):
        raise MtsCIDConfigError(
            "score.direction must be 'higher_is_more_anomalous'"
        )
    return MtsCIDConfig(
        window_size=data.get(
            "window_size", model.get("window_size", 100)
        ),
        stride=data.get("stride", 1),
        test_stride=data.get("test_stride", 1),
        alignment=score.get("alignment", "last"),
        d_model=model.get("d_model"),
        temperature=model.get("temperature", 0.1),
        encoder_layers=model.get("encoder_layers", 1),
        branches_group_embedding=model.get(
            "branches_group_embedding", "False_False"
        ),
        multiscale_kernel_size=tuple(
            model.get("multiscale_kernel_size", (5,))
        ),
        multiscale_patch_size=tuple(
            model.get("multiscale_patch_size", (10, 20))
        ),
        branch1_networks=tuple(
            model.get(
                "branch1_networks",
                (
                    "fc_linear",
                    "intra_fc_transformer",
                    "multiscale_ts_attention",
                ),
            )
        ),
        branch1_match_dimension=model.get(
            "branch1_match_dimension", "first"
        ),
        branch2_networks=tuple(
            model.get(
                "branch2_networks",
                ("multiscale_conv1d", "inter_fc_transformer"),
            )
        ),
        branch2_match_dimension=model.get(
            "branch2_match_dimension", "first"
        ),
        decoder_networks=tuple(
            model.get("decoder_networks", ("linear",))
        ),
        decoder_layers=model.get("decoder_layers", 1),
        decoder_group_embedding=model.get(
            "decoder_group_embedding", "False"
        ),
        embedding_init=model.get("embedding_init", "normal"),
        memory_guided=model.get("memory_guided", "sinusoid"),
        aggregation=score.get(
            "aggregation", model.get("aggregation", "normal_mean")
        ),
        dropout=model.get("dropout", 0.3),
        batch_size=training.get("batch_size", 64),
        num_epochs=training.get("num_epochs", 20),
        patience=training.get("patience", 10),
        learning_rate=training.get("learning_rate", 2e-3),
        weight_decay=training.get("weight_decay", 5e-5),
        validation_fraction=training.get(
            "validation_fraction", 0.2
        ),
        alpha=training.get("alpha", 1.0),
        seed=training.get("seed", root.get("seed", 42)),
        apply_ewm=preprocessing.get("apply_ewm", True),
        ewm_alpha=preprocessing.get("ewm_alpha", 0.8),
        continuity=data.get("continuity", "legacy_endpoint"),
        evaluation_fpr=evaluation.get("fpr", 0.002),
    )


def load_mtscid_config(path: str | Path) -> MtsCIDConfig:
    """Load a JSON, TOML, or YAML MtsCID configuration."""

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
                raise MtsCIDConfigError(
                    "YAML configuration requires the optional yaml dependency"
                ) from exc
            document = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
        else:
            raise MtsCIDConfigError(
                "configuration must use .json, .toml, .yaml, or .yml"
            )
    except MtsCIDConfigError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise MtsCIDConfigError(
            f"failed to load MtsCID configuration: {config_path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise MtsCIDConfigError("configuration root must be a mapping")
    return config_from_mapping(document)
