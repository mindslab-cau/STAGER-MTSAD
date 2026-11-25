"""DatasetSpec adapter for complete Anomaly Transformer baseline runs.

The model-specific trainer intentionally remains independent of repository
paths.  This module is the small orchestration boundary that reproduces the
legacy ``main.py`` data flow: the shared loader prepares ``[time, feature]``
arrays, ``MyDataset`` emits ``[batch, window, feature]`` reconstruction
windows, and every score is aligned to the first point after its window.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from stager_mtsad.baselines.data import load_next_point_data
from stager_mtsad.contracts import DatasetSpec, RunResult
from stager_mtsad.utils import set_random_seed
from stager_mtsad.utils.data import MyDataset

from .runner import AnomalyTransformerRunner


class AnomalyTransformerConfigError(ValueError):
    """Raised when an Anomaly Transformer run configuration is invalid."""


class AnomalyTransformerAdapterError(RuntimeError):
    """Raised when a complete DatasetSpec run cannot be assembled."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnomalyTransformerConfigError(f"{field_name} must be a mapping")
    return dict(value)


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AnomalyTransformerConfigError(
            f"{field_name} contains unknown fields: {', '.join(unknown)}"
        )


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise AnomalyTransformerConfigError(
            f"{field_name} must be a positive integer"
        )
    return int(value)


def _non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise AnomalyTransformerConfigError(
            f"{field_name} must be a non-negative integer"
        )
    return int(value)


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnomalyTransformerConfigError(f"{field_name} must be numeric")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise AnomalyTransformerConfigError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class AnomalyTransformerConfig:
    """Validated settings for one historical Anomaly Transformer run."""

    input_dim: int | None = None
    output_dim: int | None = None
    window_size: int = 100
    stride: int = 1
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.0
    activation: str = "gelu"
    output_attention: bool = True
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    patience: int = 10
    validation_fraction: float = 0.2
    optimizer: str = "AdamW"
    association_weight: float = 0.1
    temperature: float = 50.0
    seed: int = 0
    normalize: bool = True
    ewm_alpha: float = 0.8
    evaluation_fpr: float = 0.002

    def __post_init__(self) -> None:
        for field_name in (
            "window_size",
            "stride",
            "d_model",
            "n_heads",
            "e_layers",
            "d_ff",
            "batch_size",
            "num_epochs",
            "patience",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name),
            )
        for field_name in ("input_dim", "output_dim"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _positive_integer(value, field_name),
                )
        if (
            self.input_dim is not None
            and self.output_dim is not None
            and self.input_dim != self.output_dim
        ):
            raise AnomalyTransformerConfigError(
                "output_dim must equal input_dim for per-feature reconstruction scores"
            )
        if self.d_model % self.n_heads:
            raise AnomalyTransformerConfigError(
                "d_model must be divisible by n_heads"
            )
        dropout = _finite_number(self.dropout, "dropout")
        if not 0.0 <= dropout < 1.0:
            raise AnomalyTransformerConfigError("dropout must be in [0, 1)")
        object.__setattr__(self, "dropout", dropout)
        activation = str(self.activation).lower()
        if activation not in {"relu", "gelu"}:
            raise AnomalyTransformerConfigError(
                "activation must be either 'relu' or 'gelu'"
            )
        object.__setattr__(self, "activation", activation)
        if self.output_attention is not True:
            raise AnomalyTransformerConfigError(
                "output_attention must be true for association scoring"
            )
        learning_rate = _finite_number(self.learning_rate, "learning_rate")
        if learning_rate <= 0:
            raise AnomalyTransformerConfigError(
                "learning_rate must be positive"
            )
        object.__setattr__(self, "learning_rate", learning_rate)
        validation_fraction = _finite_number(
            self.validation_fraction,
            "validation_fraction",
        )
        if validation_fraction != 0.2:
            raise AnomalyTransformerConfigError(
                "validation_fraction is fixed at 0.2 for the reported runs"
            )
        object.__setattr__(self, "validation_fraction", validation_fraction)
        optimizer = str(self.optimizer).lower()
        if optimizer not in {"adam", "adamw"}:
            raise AnomalyTransformerConfigError(
                "optimizer must be either 'Adam' or 'AdamW'"
            )
        object.__setattr__(
            self,
            "optimizer",
            "AdamW" if optimizer == "adamw" else "Adam",
        )
        association_weight = _finite_number(
            self.association_weight,
            "association_weight",
        )
        if association_weight < 0:
            raise AnomalyTransformerConfigError(
                "association_weight must be non-negative"
            )
        object.__setattr__(self, "association_weight", association_weight)
        temperature = _finite_number(self.temperature, "temperature")
        if temperature <= 0:
            raise AnomalyTransformerConfigError("temperature must be positive")
        object.__setattr__(self, "temperature", temperature)
        seed = _non_negative_integer(self.seed, "seed")
        if seed > 2**32 - 1:
            raise AnomalyTransformerConfigError(
                "seed must be in [0, 2**32 - 1]"
            )
        object.__setattr__(self, "seed", seed)
        if self.normalize is not True:
            raise AnomalyTransformerConfigError(
                "normalize must be true to preserve train-fitted min-max scaling"
            )
        ewm_alpha = _finite_number(self.ewm_alpha, "ewm_alpha")
        if not 0.0 < ewm_alpha <= 1.0:
            raise AnomalyTransformerConfigError("ewm_alpha must be in (0, 1]")
        object.__setattr__(self, "ewm_alpha", ewm_alpha)
        evaluation_fpr = _finite_number(self.evaluation_fpr, "evaluation_fpr")
        if evaluation_fpr != 0.002:
            raise AnomalyTransformerConfigError(
                "evaluation_fpr is fixed at 0.002"
            )
        object.__setattr__(self, "evaluation_fpr", evaluation_fpr)

    def as_dict(self) -> dict[str, Any]:
        """Return a portable record of the effective run configuration."""

        return {
            "window": {
                "size": self.window_size,
                "stride": self.stride,
                "score_anchor": "next_point",
            },
            "model": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "e_layers": self.e_layers,
                "d_ff": self.d_ff,
                "dropout": self.dropout,
                "activation": self.activation,
                "output_attention": self.output_attention,
            },
            "training": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "patience": self.patience,
                "validation_fraction": self.validation_fraction,
                "optimizer": self.optimizer,
                "association_weight": self.association_weight,
                "seed": self.seed,
            },
            "preprocessing": {
                "normalization": "train_minmax",
                "test_clipping": False,
                "ewm": {
                    "alpha": self.ewm_alpha,
                    "adjust": True,
                    "reset_between_splits": True,
                },
            },
            "scoring": {
                "temperature": self.temperature,
                "feature_score_orientation": "[time, feature]",
            },
            "evaluation": {"method": "label_aware_fpr", "fpr": self.evaluation_fpr},
        }


def config_from_mapping(
    document: Mapping[str, Any],
) -> AnomalyTransformerConfig:
    """Parse the packaged nested preset without silently accepting drift."""

    root = _mapping(document, "configuration")
    _reject_unknown(
        root,
        {
            "run",
            "baseline",
            "dataset",
            "preprocessing",
            "window",
            "model",
            "training",
            "scoring",
            "evaluation",
            "reproducibility",
            "seed",
        },
        "configuration",
    )
    if "run" in root:
        _mapping(root["run"], "run")
    if "dataset" in root:
        _mapping(root["dataset"], "dataset")
    if "reproducibility" in root:
        reproducibility = _mapping(root["reproducibility"], "reproducibility")
        _reject_unknown(reproducibility, {"seeds"}, "reproducibility")
    baseline = root.get("baseline")
    if baseline is not None and str(baseline).lower() not in {
        "anomaly_transformer",
        "anomalytransformer",
    }:
        raise AnomalyTransformerConfigError(
            "baseline must identify anomaly_transformer"
        )

    preprocessing = _mapping(root.get("preprocessing", {}), "preprocessing")
    _reject_unknown(
        preprocessing,
        {
            "input_stage",
            "prepared_variant",
            "feature_count",
            "feature_order",
            "normalization",
            "constant_feature_value",
            "test_clipping",
            "ewm",
            "missing_values",
        },
        "preprocessing",
    )
    if "prepared_variant" in preprocessing:
        _mapping(preprocessing["prepared_variant"], "preprocessing.prepared_variant")
    if preprocessing.get("normalization", "train_minmax") != "train_minmax":
        raise AnomalyTransformerConfigError(
            "preprocessing.normalization must be 'train_minmax'"
        )
    if preprocessing.get("constant_feature_value", 0.0) != 0.0:
        raise AnomalyTransformerConfigError(
            "preprocessing.constant_feature_value must be 0.0"
        )
    if preprocessing.get("test_clipping", False) is not False:
        raise AnomalyTransformerConfigError(
            "preprocessing.test_clipping must be false"
        )
    if preprocessing.get("missing_values", "error") != "error":
        raise AnomalyTransformerConfigError(
            "preprocessing.missing_values must be 'error'"
        )
    ewm = _mapping(preprocessing.get("ewm", {}), "preprocessing.ewm")
    _reject_unknown(
        ewm,
        {"alpha", "adjust", "reset_between_splits"},
        "preprocessing.ewm",
    )
    if ewm.get("adjust", True) is not True:
        raise AnomalyTransformerConfigError(
            "preprocessing.ewm.adjust must be true"
        )
    if ewm.get("reset_between_splits", True) is not True:
        raise AnomalyTransformerConfigError(
            "preprocessing.ewm.reset_between_splits must be true"
        )

    window = _mapping(root.get("window", {}), "window")
    _reject_unknown(
        window,
        {"size", "stride", "score_anchor", "selection_status"},
        "window",
    )
    if window.get("score_anchor", "next_point") != "next_point":
        raise AnomalyTransformerConfigError(
            "window.score_anchor must be 'next_point'"
        )

    model = _mapping(root.get("model", {}), "model")
    _reject_unknown(
        model,
        {
            "input_dim",
            "output_dim",
            "d_model",
            "n_heads",
            "e_layers",
            "d_ff",
            "dropout",
            "activation",
            "output_attention",
        },
        "model",
    )
    training = _mapping(root.get("training", {}), "training")
    _reject_unknown(
        training,
        {
            "batch_size",
            "learning_rate",
            "num_epochs",
            "patience",
            "validation_fraction",
            "optimizer",
            "association_weight",
            "seed",
        },
        "training",
    )
    scoring = _mapping(root.get("scoring", {}), "scoring")
    _reject_unknown(scoring, {"temperature"}, "scoring")
    evaluation = _mapping(root.get("evaluation", {}), "evaluation")
    _reject_unknown(evaluation, {"method", "fpr"}, "evaluation")
    if evaluation.get("method", "label_aware_fpr") != "label_aware_fpr":
        raise AnomalyTransformerConfigError(
            "evaluation.method must be 'label_aware_fpr'"
        )

    input_dim = model.get("input_dim", preprocessing.get("feature_count"))
    feature_count = preprocessing.get("feature_count")
    if input_dim is not None and feature_count is not None:
        if int(input_dim) != int(feature_count):
            raise AnomalyTransformerConfigError(
                "model.input_dim and preprocessing.feature_count must match"
            )
    return AnomalyTransformerConfig(
        input_dim=input_dim,
        output_dim=model.get("output_dim", input_dim),
        window_size=window.get("size", 100),
        stride=window.get("stride", 1),
        d_model=model.get("d_model", 512),
        n_heads=model.get("n_heads", 8),
        e_layers=model.get("e_layers", 3),
        d_ff=model.get("d_ff", 512),
        dropout=model.get("dropout", 0.0),
        activation=model.get("activation", "gelu"),
        output_attention=model.get("output_attention", True),
        batch_size=training.get("batch_size", 32),
        learning_rate=training.get("learning_rate", 1e-4),
        num_epochs=training.get("num_epochs", 50),
        patience=training.get("patience", 10),
        validation_fraction=training.get("validation_fraction", 0.2),
        optimizer=training.get("optimizer", "AdamW"),
        association_weight=training.get("association_weight", 0.1),
        temperature=scoring.get("temperature", 50.0),
        seed=training.get("seed", root.get("seed", 0)),
        normalize=True,
        ewm_alpha=ewm.get("alpha", 0.8),
        evaluation_fpr=evaluation.get("fpr", 0.002),
    )


def load_anomaly_transformer_config(
    path: str | Path,
) -> AnomalyTransformerConfig:
    """Load a JSON, TOML, or YAML Anomaly Transformer preset."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        document = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML configs require PyYAML") from exc
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise AnomalyTransformerConfigError(
            "configuration path must use .json, .toml, .yaml, or .yml"
        )
    if not isinstance(document, Mapping):
        raise AnomalyTransformerConfigError(
            "configuration root must be a mapping"
        )
    return config_from_mapping(document)


@dataclass(frozen=True, slots=True)
class PreparedAnomalyTransformerData:
    """Shared-loader products and exact score alignment for one run."""

    train: MyDataset
    test: MyDataset
    labels: np.ndarray
    timestamps: np.ndarray
    indices: np.ndarray
    feature_names: tuple[str, ...]


def prepare_dataset_spec(
    spec: DatasetSpec,
    config: AnomalyTransformerConfig,
) -> PreparedAnomalyTransformerData:
    """Prepare both splits with the retained common ``utils.data`` pipeline."""

    shared = load_next_point_data(
        spec,
        window_size=config.window_size,
        train_stride=config.stride,
        test_stride=config.stride,
        ewm_alpha=config.ewm_alpha,
    )
    train = shared.train_dataset
    test = shared.test_dataset
    feature_count = train.tag_values.shape[1]
    if config.input_dim is not None and config.input_dim != feature_count:
        raise AnomalyTransformerAdapterError(
            f"configuration expects {config.input_dim} features but the dataset "
            f"provides {feature_count}"
        )
    if config.output_dim is not None and config.output_dim != feature_count:
        raise AnomalyTransformerAdapterError(
            f"configuration output_dim is {config.output_dim} but the dataset "
            f"provides {feature_count} features"
        )
    indices = shared.target_indices.astype(np.int64, copy=False)
    timestamps = shared.target_timestamps
    aligned_labels = shared.labels.astype(np.int8, copy=False)
    if not (
        len(test) == len(indices) == len(timestamps) == len(aligned_labels)
    ):
        raise AnomalyTransformerAdapterError(
            "test windows, next-point anchors, timestamps, and labels must align"
        )
    return PreparedAnomalyTransformerData(
        train=train,
        test=test,
        labels=aligned_labels,
        timestamps=timestamps,
        indices=indices,
        feature_names=shared.feature_names,
    )


def _split_training(
    dataset: Dataset[Any],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[Dataset[Any], Dataset[Any]]:
    validation_size = int(len(dataset) * validation_fraction)
    training_size = len(dataset) - validation_size
    if validation_size < 1:
        raise AnomalyTransformerAdapterError(
            "validation_fraction=0.2 produced no validation windows"
        )
    if training_size < 1:
        raise AnomalyTransformerAdapterError(
            "validation_fraction=0.2 leaves no training windows"
        )
    return random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )


def run_anomaly_transformer(
    spec: DatasetSpec,
    config: AnomalyTransformerConfig,
    *,
    device: str | torch.device | None = "auto",
) -> RunResult:
    """Fit and score Anomaly Transformer from one portable ``DatasetSpec``."""

    set_random_seed(config.seed)
    prepared = prepare_dataset_spec(spec, config)
    training, validation = _split_training(
        prepared.train,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_loader = DataLoader(
        training,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        prepared.test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    resolved_device: str | torch.device | None
    resolved_device = None if device is None or str(device) == "auto" else device
    feature_count = prepared.train.tag_values.shape[1]
    runner = AnomalyTransformerRunner(
        window_size=config.window_size,
        input_dim=feature_count,
        output_dim=feature_count,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        activation=config.activation,
        output_attention=config.output_attention,
        association_weight=config.association_weight,
        temperature=config.temperature,
        seed=config.seed,
        device=resolved_device,
    )
    optimizer_class: type[torch.optim.Optimizer]
    optimizer_class = (
        torch.optim.AdamW
        if config.optimizer == "AdamW"
        else torch.optim.Adam
    )
    result = runner.run(
        train_loader,
        test_loader,
        num_epochs=config.num_epochs,
        valid_loader=validation_loader,
        early_stopping_patience=config.patience,
        optimizer_class=optimizer_class,
        learning_rate=config.learning_rate,
        labels=prepared.labels,
        timestamps=prepared.timestamps,
        indices=prepared.indices,
        seed=config.seed,
    )
    feature_scores = np.asarray(result.feature_scores)
    expected_shape = (len(prepared.test), feature_count)
    if feature_scores.shape != expected_shape:
        raise AnomalyTransformerAdapterError(
            "Anomaly Transformer feature scores must have shape "
            f"{expected_shape}; received {feature_scores.shape}"
        )
    metadata = {
        **dict(result.metadata),
        "dataset": spec.logical_id,
        "feature_names": list(prepared.feature_names),
        "train_array_shape": list(prepared.train.tag_values.shape),
        "test_array_shape": list(prepared.test.tag_values.shape),
        "window_batch_contract": "[batch, window, feature]",
        "target_batch_contract": "[batch, feature]",
        "validation_fraction": config.validation_fraction,
        "training_window_count": len(training),
        "validation_window_count": len(validation),
        "test_window_count": len(prepared.test),
        "evaluation_fpr": config.evaluation_fpr,
    }
    return replace(
        result,
        metadata=metadata,
        config=config.as_dict(),
    )


__all__ = [
    "AnomalyTransformerAdapterError",
    "AnomalyTransformerConfig",
    "AnomalyTransformerConfigError",
    "PreparedAnomalyTransformerData",
    "config_from_mapping",
    "load_anomaly_transformer_config",
    "prepare_dataset_spec",
    "run_anomaly_transformer",
]
