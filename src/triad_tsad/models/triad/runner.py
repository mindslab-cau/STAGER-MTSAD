"""Offline, in-memory TriAD runners."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from triad_tsad.contracts import RunResult
from triad_tsad.utils import set_random_seed

from .config import TriADConfig
from .data import NextPointDataset, align_to_targets, prepared_values
from .model import TriAD
from .trainer import TriADTrainer


def _training_loaders(
    dataset: Dataset,
    *,
    config: TriADConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader | None]:
    validation_size = int(len(dataset) * config.validation_fraction)
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("The configured validation split leaves no training data.")

    validation_loader = None
    training_dataset = dataset
    if validation_size:
        training_dataset, validation_dataset = random_split(
            dataset,
            [training_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return training_loader, validation_loader


@dataclass(slots=True)
class TriADExecution:
    """A completed fit/score operation and its restored best model."""

    result: RunResult
    model: TriAD
    history: dict[str, Any]


def fit_score_triad_datasets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    num_features: int,
    target_indices: Sequence[int] | np.ndarray,
    raw_test_length: int,
    labels: Sequence[Any] | np.ndarray | None = None,
    timestamps: Sequence[Any] | np.ndarray | None = None,
    indices: Sequence[Any] | np.ndarray | None = None,
    config: TriADConfig | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
    dataset_id: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> TriADExecution:
    """Fit and score already-windowed datasets.

    The caller owns preprocessing and window selection. This boundary lets the
    command-line launcher use the retained timestamp-continuity filter without
    rebuilding windows from a dense array.
    """

    resolved = config or TriADConfig()
    set_random_seed(seed)
    if isinstance(num_features, bool) or int(num_features) < 1:
        raise ValueError("num_features must be a positive integer.")
    num_features = int(num_features)
    if len(train_dataset) < 1 or len(test_dataset) < 1:
        raise ValueError("Train and test datasets must contain valid windows.")
    if resolved.process_to_tag is not None and (
        len(resolved.process_to_tag) != num_features
    ):
        raise ValueError(
            "process_to_tag length must match the prepared feature count."
        )

    targets = np.asarray(target_indices, dtype=np.int64)
    if targets.ndim != 1 or len(targets) != len(test_dataset):
        raise ValueError(
            "target_indices must contain one source index per test window."
        )
    if (
        raw_test_length < 1
        or np.any(targets < 0)
        or np.any(targets >= raw_test_length)
    ):
        raise ValueError("target_indices fall outside the raw test sequence.")

    train_loader, validation_loader = _training_loaders(
        train_dataset,
        config=resolved,
        seed=seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=resolved.batch_size,
        shuffle=False,
        num_workers=resolved.num_workers,
    )

    model = TriAD(**resolved.model_kwargs(num_features=num_features))
    trainer = TriADTrainer(device=device)
    model, history = trainer.fit(
        train_loader,
        model,
        lr=resolved.learning_rate,
        epochs=resolved.epochs,
        valid_loader=validation_loader,
        early_stopping_patience=resolved.early_stopping_patience,
        lr_global=resolved.lr_global,
        lr_aux_classifier=resolved.lr_aux_classifier,
        ema_alpha=resolved.ema_alpha,
        aux_euclid_weight=resolved.aux_euclid_weight,
        aux_cos_weight=resolved.aux_cos_weight,
        aux_ortho_weight=resolved.aux_ortho_weight,
        aux_cls_weight=resolved.aux_cls_weight,
        aux_norm_eps=resolved.aux_norm_eps,
        aux_cls_use_norm=resolved.aux_cls_use_norm,
        best_state_policy=resolved.best_state_policy,
    )
    average_loss, feature_scores = trainer.score(test_loader, model)
    if feature_scores.shape != (len(targets), num_features):
        raise ValueError(
            "TriAD score shape does not match the retained test windows."
        )

    aligned_labels = align_to_targets(
        labels,
        raw_length=raw_test_length,
        target_indices=targets,
        name="labels",
    )
    aligned_timestamps = align_to_targets(
        timestamps,
        raw_length=raw_test_length,
        target_indices=targets,
        name="timestamps",
    )
    aligned_indices = align_to_targets(
        indices,
        raw_length=raw_test_length,
        target_indices=targets,
        name="indices",
    )
    if aligned_indices is None:
        aligned_indices = targets

    diagnostic_point_scores = feature_scores.mean(axis=1)
    portable_config = resolved.as_dict()
    if dataset_id is not None:
        portable_config["dataset"] = {"logical_id": str(dataset_id)}
    metadata: dict[str, Any] = {
        "model": "TriAD",
        "alignment": "next_point",
        "score": "squared_forecast_error",
        "evaluation_score_orientation": "[feature, time]",
        "evaluation_score_source": "feature_scores_transposed",
        "point_aggregation": "diagnostic_feature_mean_only",
        "test_average_mse": average_loss,
        "training_history": history,
        "best_state_policy": resolved.best_state_policy,
        "evaluation_fpr": resolved.evaluation_fpr,
    }
    if dataset_id is not None:
        metadata["dataset"] = str(dataset_id)
    if feature_names is not None:
        names = tuple(str(name) for name in feature_names)
        if len(names) != num_features:
            raise ValueError(
                "feature_names length must match the prepared feature count."
            )
        metadata["feature_names"] = list(names)

    result = RunResult(
        point_scores=diagnostic_point_scores,
        feature_scores=feature_scores,
        labels=aligned_labels,
        timestamps=aligned_timestamps,
        indices=aligned_indices,
        metadata=metadata,
        seed=seed,
        config=portable_config,
    )
    return TriADExecution(result=result, model=model, history=history)


def run_triad(
    train_values: Any,
    test_values: Any,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
    timestamps: Sequence[Any] | np.ndarray | None = None,
    indices: Sequence[Any] | np.ndarray | None = None,
    config: TriADConfig | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> RunResult:
    """Fit and score prepared arrays without reading or writing files.

    Inputs must already have their feature order, normalization, smoothing, and
    discontinuity handling fixed. Scores align to the next point after each
    input window.
    """

    resolved = config or TriADConfig()
    train_array = prepared_values(train_values, name="train_values")
    test_array = prepared_values(test_values, name="test_values")
    if train_array.shape[1] != test_array.shape[1]:
        raise ValueError("Train and test feature counts must match.")

    train_dataset = NextPointDataset(
        train_array,
        window_size=resolved.window_size,
        stride=resolved.stride,
    )
    test_dataset = NextPointDataset(
        test_array,
        window_size=resolved.window_size,
        stride=resolved.stride,
    )
    execution = fit_score_triad_datasets(
        train_dataset,
        test_dataset,
        num_features=train_array.shape[1],
        target_indices=test_dataset.target_indices,
        raw_test_length=len(test_array),
        labels=labels,
        timestamps=timestamps,
        indices=indices,
        config=resolved,
        seed=seed,
        device=device,
    )
    return execution.result


__all__ = ["TriADExecution", "fit_score_triad_datasets", "run_triad"]
