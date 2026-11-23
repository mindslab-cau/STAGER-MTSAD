"""Thin device-aware wrapper around the preserved MtsCID network."""

from __future__ import annotations

from typing import Any

import torch

from .config import MtsCIDConfig
from .vendor import TransformerVar


def build_model(
    config: MtsCIDConfig,
    *,
    feature_count: int,
    device: str | torch.device,
) -> TransformerVar:
    """Create MtsCID without CUDA or feature-count hardcoding."""

    resolved_device = torch.device(device)
    model = TransformerVar(
        config.model_mapping(
            feature_count=feature_count,
            device=resolved_device,
        ),
        dropout=config.dropout,
    )
    return model.to(resolved_device)


def native_anomaly_energy(
    model_output: dict[str, Any],
    inputs: torch.Tensor,
    *,
    temperature: float,
    aggregation: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute the source reconstruction-memory anomaly energy.

    Returns the per-window/per-time energy and, for ``normal_mean``, its
    exact per-feature decomposition.
    """

    from .vendor import GatheringLoss

    reconstruction = model_output["out"]
    queries = model_output["queries"]
    memory = model_output["mem"]
    reconstruction_loss = torch.nn.functional.mse_loss(
        reconstruction,
        inputs,
        reduction="none",
    )
    gathering_loss = GatheringLoss(
        reduction="none", memto_framework=True
    )
    latent_score = torch.softmax(
        gathering_loss(queries, memory) / temperature,
        dim=-1,
    )

    if aggregation != "normal_mean":
        raise ValueError(
            "the migrated MtsCID preset supports only 'normal_mean'"
        )
    feature_energy = reconstruction_loss * latent_score.unsqueeze(-1)
    energy = feature_energy.mean(dim=-1)
    return energy, feature_energy
