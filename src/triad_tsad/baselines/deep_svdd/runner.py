"""DatasetSpec-to-RunResult adapter for time-series Deep SVDD."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from triad_tsad.contracts import DatasetSpec, RunResult

from .config import DeepSVDDConfig
from .data import (
    LegacyNextWindowDataset,
    PreparedDataset,
    prepare_dataset,
)
from .trainer import DeepSVDDTrainer


class DeepSVDDRunnerError(RuntimeError):
    """Raised when a complete Deep SVDD run cannot be produced."""


def _test_auc(
    labels: np.ndarray | None,
    scores: np.ndarray,
) -> dict[str, float]:
    if labels is None or len(np.unique(labels)) < 2:
        return {}
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while (
            stop < len(scores)
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive = np.asarray(labels) == 1
    positive_count = int(np.sum(positive))
    negative_count = len(labels) - positive_count
    auc = (
        np.sum(ranks[positive])
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return {"test_auc": float(auc)}


class DeepSVDDRunner:
    """Train and score the migrated Deep SVDD time-series baseline."""

    def __init__(
        self,
        config: DeepSVDDConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.trainer = DeepSVDDTrainer(config, device=device)
        self.prepared: PreparedDataset | None = None
        self.feature_count: int | None = None
        self.fitted_spec: DatasetSpec | None = None

    def _prepare(self, spec: DatasetSpec) -> PreparedDataset:
        return prepare_dataset(
            spec,
            normalize=self.config.normalize,
            ewm_alpha=self.config.ewm_alpha,
        )

    def _window_dataset(
        self,
        prepared: Any,
        spec: DatasetSpec,
    ) -> LegacyNextWindowDataset:
        return LegacyNextWindowDataset(
            prepared,
            window_size=self.config.window_size,
            stride=self.config.stride,
            predict_size=self.config.predict_size,
            expected_period_seconds=spec.schema.get(
                "expected_period_seconds"
            ),
        )

    def fit(self, spec: DatasetSpec) -> DeepSVDDRunner:
        """Fit a fresh model from the training split."""

        prepared = self._prepare(spec)
        feature_count = prepared.train.values.shape[1]
        train_dataset = self._window_dataset(prepared.train, spec)
        self.trainer.fit(
            train_dataset,
            window_size=self.config.window_size,
            feature_count=feature_count,
        )
        self.prepared = prepared
        self.feature_count = feature_count
        self.fitted_spec = spec
        return self

    def score(self, spec: DatasetSpec) -> RunResult:
        """Score test windows and align each score to the next raw row."""

        if (
            self.prepared is None
            or self.feature_count is None
            or self.fitted_spec is None
        ):
            raise DeepSVDDRunnerError(
                "fit must be called before score"
            )
        if spec != self.fitted_spec:
            raise DeepSVDDRunnerError(
                "score must use the DatasetSpec fitted by this runner"
            )
        test_dataset = self._window_dataset(self.prepared.test, spec)
        scores, anchors = self.trainer.score(test_dataset)
        split = self.prepared.test
        labels = None if split.labels is None else split.labels[anchors]
        timestamps = (
            None
            if split.timestamps is None
            else split.timestamps[anchors]
        )
        indices = split.source_indices[anchors]
        history = self.trainer.history
        if history is None:
            raise DeepSVDDRunnerError(
                "training history is unavailable"
            )
        metrics = _test_auc(labels, scores)
        return RunResult(
            point_scores=scores,
            labels=labels,
            timestamps=timestamps,
            indices=indices,
            metrics=metrics,
            metadata={
                "baseline": "deep_svdd",
                "objective": self.config.objective,
                "score_anchor": (
                    "first_point_after_input_window"
                ),
                "score_definition": (
                    "squared_distance_to_center"
                    if self.config.objective == "one-class"
                    else (
                        "squared_distance_to_center_minus_"
                        "squared_radius"
                    )
                ),
                "score_direction": "higher_is_more_anomalous",
                "radius": history.radius,
                "pretrained": self.config.pretrain,
                "source_license": "MIT",
                "evaluation_fpr": self.config.evaluation_fpr,
                "feature_names": list(split.feature_names),
            },
            seed=self.config.seed,
            config=self.config.as_dict(),
        )


def run_deep_svdd(
    spec: DatasetSpec,
    config: DeepSVDDConfig,
    *,
    device: str | torch.device = "cpu",
) -> RunResult:
    """Fit and score Deep SVDD without writing artifacts or using a network."""

    runner = DeepSVDDRunner(config, device=device)
    runner.fit(spec)
    return runner.score(spec)
