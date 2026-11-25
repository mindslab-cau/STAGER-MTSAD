"""Typed, path-free STAGER configuration."""

from __future__ import annotations

import math
from dataclasses import InitVar, asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StagerConfig:
    """Model and training settings for one in-memory run.

    Dataset locations and output locations intentionally do not belong to this
    configuration.
    """

    window_size: int = 120
    stride: int = 1
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    validation_fraction: float = 0.2
    num_workers: int = 0

    adj_func: str = "sigmoid"
    gnn_type: str = "gat"
    gnn_layers: int = 1
    gnn_hidden_dim: int | None = None
    gnn_heads: int = 4
    gnn_dropout: float = 0.1
    node_dim: int = 16
    film_hidden_ratio: float = 2.0
    use_ema_in_forward: bool = False
    feature_residual: bool = False
    film_pre_norm_type: str | None = None
    film_post_norm_type: str | None = "ln"
    dag_use_bn: bool = False
    feature_encoder_type: str = "multi_scale"
    temporal_type: str = "dag"
    conditioning_type: str | None = "film"
    tag_mixing: bool = True
    nonmix_method: str = "per_linear"
    tgnn_layers: int = 2
    tgnn_hidden_multiplier: int = 2
    shift: int = 2
    temporal_pool: str = "mean"
    ms_windows: tuple[int, ...] = (5, 10, 20)

    global_seed_dim: int | None = None
    global_mlp_hidden: int | None = None
    global_mlp_layers: int = 2
    global_mlp_init: str = "xavier_uniform"
    global_mlp_out_norm: str | None = None
    aux_target: str = "global"
    aux_classifier_hidden: int | None = None
    aux_classifier_layers: int = 2
    aux_classifier_dropout: float = 0.0
    process_to_tag: tuple[int, ...] | None = None

    lr_global: float | None = None
    lr_aux_classifier: float | None = None
    ema_alpha: float = 1e-3
    aux_euclid_weight: float = 0.0
    aux_cos_weight: float = 0.0
    aux_ortho_weight: float = 0.0
    aux_cls_weight: float = 0.0
    aux_norm_eps: float = 1e-8
    aux_cls_use_norm: bool = False
    best_state_policy: InitVar[str | None] = None

    evaluation_fpr: float = 0.002
    ewm_alpha: float = 0.8
    feature_edge_layout: str = "legacy"

    def __post_init__(self, best_state_policy: str | None) -> None:
        if self.feature_edge_layout not in ("legacy", "per_sample"):
            raise ValueError("feature_edge_layout must be 'legacy' or 'per_sample'.")
        if (
            isinstance(self.ewm_alpha, bool)
            or not isinstance(self.ewm_alpha, (int, float))
            or not math.isfinite(self.ewm_alpha)
            or not 0 < self.ewm_alpha <= 1
        ):
            raise ValueError("ewm_alpha must be a finite number in (0, 1].")
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        if self.stride < 1:
            raise ValueError("stride must be positive.")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.evaluation_fpr != 0.002:
            raise ValueError("STAGER records only evaluation_fpr=0.002.")
        if self.process_to_tag is not None and not self.process_to_tag:
            raise ValueError("process_to_tag must not be empty when supplied.")
        # Accept older presets that include this field.
        if best_state_policy not in (None, "reference", "snapshot"):
            raise ValueError("Unknown legacy best_state_policy.")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> StagerConfig:
        """Build from a flat mapping and normalize sequence-valued fields."""

        normalized = dict(values)
        if "ms_windows" in normalized:
            normalized["ms_windows"] = tuple(normalized["ms_windows"])
        if normalized.get("process_to_tag") is not None:
            normalized["process_to_tag"] = tuple(normalized["process_to_tag"])
        return cls(**normalized)

    def model_kwargs(self, *, num_features: int) -> dict[str, Any]:
        return {
            "num_features": num_features,
            "feature_dim": self.window_size,
            "adj_func": self.adj_func,
            "gnn_type": self.gnn_type,
            "feature_edge_layout": self.feature_edge_layout,
            "gnn_layers": self.gnn_layers,
            "gnn_hidden_dim": self.gnn_hidden_dim,
            "gnn_heads": self.gnn_heads,
            "gnn_dropout": self.gnn_dropout,
            "node_dim": self.node_dim,
            "film_hidden_ratio": self.film_hidden_ratio,
            "use_ema_in_forward": self.use_ema_in_forward,
            "feature_residual": self.feature_residual,
            "film_pre_norm_type": self.film_pre_norm_type,
            "film_post_norm_type": self.film_post_norm_type,
            "dag_use_bn": self.dag_use_bn,
            "feature_encoder_type": self.feature_encoder_type,
            "temporal_type": self.temporal_type,
            "conditioning_type": self.conditioning_type,
            "tag_mixing": self.tag_mixing,
            "nonmix_method": self.nonmix_method,
            "tgnn_layers": self.tgnn_layers,
            "tgnn_hidden_multiplier": self.tgnn_hidden_multiplier,
            "shift": self.shift,
            "temporal_pool": self.temporal_pool,
            "ms_windows": list(self.ms_windows),
            "global_seed_dim": self.global_seed_dim,
            "global_mlp_hidden": self.global_mlp_hidden,
            "global_mlp_layers": self.global_mlp_layers,
            "global_mlp_init": self.global_mlp_init,
            "global_mlp_out_norm": self.global_mlp_out_norm,
            "aux_target": self.aux_target,
            "aux_classifier_hidden": self.aux_classifier_hidden,
            "aux_classifier_layers": self.aux_classifier_layers,
            "aux_classifier_dropout": self.aux_classifier_dropout,
            "process_to_tag": (
                None
                if self.process_to_tag is None
                else list(self.process_to_tag)
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if self.feature_edge_layout == "legacy":
            record.pop("feature_edge_layout")
            if self.ewm_alpha == 0.8:
                record.pop("ewm_alpha")
        return record


__all__ = ["StagerConfig"]
