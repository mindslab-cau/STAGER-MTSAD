"""DatasetSpec adapter used by the standalone FuSAGNet baseline command."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from triad_tsad.baselines.data import load_next_point_data
from triad_tsad.contracts import DatasetSpec, RunResult

from .config import FuSAGNetConfig
from .runner import FuSAGNetRunner


def run_fusagnet(
    spec: DatasetSpec,
    config: FuSAGNetConfig,
    *,
    seed: int | None = None,
    device: str = "auto",
) -> RunResult:
    """Preprocess, train, and score FuSAGNet with the legacy next-point anchor."""

    if seed is not None:
        config = replace(
            config,
            training=replace(config.training, seed=int(seed)),
        )
    prepared = load_next_point_data(
        spec,
        window_size=config.model.window_size,
        train_stride=1,
        test_stride=1,
        ewm_alpha=0.8,
    )
    config.layout.validate_observed_order(prepared.feature_names)
    runner = FuSAGNetRunner(config, device=device)
    result = runner.fit_score(
        prepared.train_values,
        prepared.test_values,
        feature_order=prepared.feature_names,
        train_valid_starts=prepared.train_valid_starts,
        test_valid_starts=prepared.test_valid_starts,
    )
    if not np.array_equal(result.indices, prepared.target_indices):
        raise RuntimeError(
            "FuSAGNet target indices diverged from the shared data.py alignment"
        )
    metadata = {
        **dict(result.metadata),
        "feature_names": list(prepared.feature_names),
        "input_shape": "[batch, window, feature]",
        "feature_score_shape": "[time, feature]",
        "evaluator_score_shape": "[feature, time]",
        "evaluation_reduction": "per_feature_threshold_then_or",
        "preprocessing": "utils.data.load_dataset",
        "expected_period_seconds": prepared.expected_period_seconds,
    }
    return replace(
        result,
        labels=prepared.labels,
        timestamps=prepared.target_timestamps,
        metadata=metadata,
    )


__all__ = ["run_fusagnet"]
