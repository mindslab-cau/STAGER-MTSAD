"""Unified execution path for KNN and OCSVM experiments."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from triad_tsad.contracts import DatasetSpec, RunResult
from triad_tsad.evaluation import (
    evaluate_ensemble_best_modified,
    save_run_artifacts,
)

from .config import ClassicalRunConfig
from .data import PreparedClassicalData, load_classical_csv
from .models import ClassicalEstimator, build_estimator

EstimatorFactory = Callable[[str, Mapping[str, Any]], ClassicalEstimator]
DataLoader = Callable[..., PreparedClassicalData]


class ClassicalRunError(RuntimeError):
    """Raised when a classical model emits an invalid run result."""


def _score_statistics(scores: np.ndarray) -> dict[str, float]:
    return {
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }


def _run_prepared(
    dataset: DatasetSpec,
    config: ClassicalRunConfig,
    prepared: PreparedClassicalData,
    output_dir: str | Path,
    *,
    seed: int,
    estimator_factory: EstimatorFactory,
) -> RunResult:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ClassicalRunError("seed must be a non-negative integer")
    np.random.seed(seed)

    stage_count = 3 if prepared.labels is not None else 2
    with tqdm(
        total=stage_count,
        desc=f"{config.model.upper()} run",
        unit="stage",
        dynamic_ncols=True,
    ) as progress:
        estimator = estimator_factory(config.model, config.model_params)
        progress.set_postfix(stage="fit")
        estimator.fit(prepared.train_values)
        progress.update()
        progress.set_postfix(stage="score")
        raw_scores = np.asarray(
            estimator.decision_function(prepared.test_values),
            dtype=np.float64,
        )
        progress.update()
        if raw_scores.ndim != 1:
            raw_scores = np.squeeze(raw_scores)
        if raw_scores.ndim != 1:
            raise ClassicalRunError(
                "decision_function must return one score per row"
            )
        if len(raw_scores) != len(prepared.test_values):
            raise ClassicalRunError(
                "decision_function score count does not match the test rows"
            )
        if not np.all(np.isfinite(raw_scores)):
            raise ClassicalRunError(
                "decision_function returned non-finite scores"
            )

        labels = (
            None
            if prepared.labels is None
            else np.asarray(prepared.labels)
        )
        timestamps = np.asarray(prepared.timestamps)
        indices = np.asarray(prepared.indices)
        metrics = _score_statistics(raw_scores)
        evaluation_metrics = None
        evaluation_output = None
        if labels is not None:
            progress.set_postfix(stage="evaluate")
            figure_dir = Path(output_dir) / "bestf1_figures"
            figure_dir.mkdir(parents=True, exist_ok=True)
            evaluation_metrics, evaluation_output = (
                evaluate_ensemble_best_modified(
                    attack_label=labels,
                    predictions=[raw_scores],
                    fpr_threshold=config.fpr,
                    fig_dir=figure_dir,
                    is_brief=False,
                )
            )
            metrics.update(evaluation_metrics)
            progress.update()

    portable_config = config.as_dict()
    portable_config["dataset"] = {"logical_id": dataset.logical_id}
    metadata: dict[str, Any] = {
        "model": config.model,
        "dataset": dataset.logical_id,
        "score_semantics": RunResult.SCORE_SEMANTICS,
        "feature_names": list(prepared.feature_names),
        "train_shape": list(prepared.train_values.shape),
        "test_shape": list(prepared.test_values.shape),
    }
    if evaluation_metrics is not None:
        metadata["evaluation"] = {
            "protocol": config.evaluation_protocol,
            "label_aware": True,
            "fpr": config.fpr,
            "selection_metric": "eTaf1",
        }

    result = RunResult(
        point_scores=raw_scores,
        labels=labels,
        timestamps=timestamps,
        indices=indices,
        metrics=metrics,
        metadata=metadata,
        seed=seed,
        config=portable_config,
    )
    save_run_artifacts(
        result,
        output_dir,
        predictions=(
            None
            if evaluation_output is None
            else evaluation_output["combined_predictions"]
        ),
        thresholds=(
            None if evaluation_output is None else evaluation_output["thresholds"]
        ),
    )
    return result


def run_classical_baseline(
    dataset: DatasetSpec,
    config: ClassicalRunConfig,
    output_dir: str | Path,
    *,
    seed: int = 0,
    estimator_factory: EstimatorFactory = build_estimator,
    data_loader: DataLoader = load_classical_csv,
) -> RunResult:
    """Fit, score, evaluate, and serialize one KNN or OCSVM run."""

    if config.parameter_grid:
        raise ClassicalRunError(
            "run_classical_baseline accepts one parameter set; use "
            "run_classical_sweep for parameter_grid configurations"
        )
    prepared = data_loader(dataset, ewm_alpha=config.ewm_alpha)
    return _run_prepared(
        dataset,
        config,
        prepared,
        output_dir,
        seed=seed,
        estimator_factory=estimator_factory,
    )


def _slug(value: Any) -> str:
    text = str(value).lower()
    safe = "".join(
        character if character.isalnum() else "-"
        for character in text
    ).strip("-")
    return safe or "value"


def _parameter_sets(
    base: Mapping[str, Any],
    grid: Mapping[str, tuple[Any, ...]],
) -> list[dict[str, Any]]:
    if not grid:
        return [dict(base)]
    keys = list(grid)
    return [
        {
            **base,
            **dict(zip(keys, values, strict=True)),
        }
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def run_classical_sweep(
    dataset: DatasetSpec,
    config: ClassicalRunConfig,
    output_dir: str | Path,
    *,
    seed: int = 0,
    estimator_factory: EstimatorFactory = build_estimator,
    data_loader: DataLoader = load_classical_csv,
) -> tuple[RunResult, ...]:
    """Expand a model parameter grid through the same single-run pipeline."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    parameter_sets = _parameter_sets(
        config.model_params,
        config.parameter_grid,
    )
    prepared = data_loader(dataset, ewm_alpha=config.ewm_alpha)
    grid_keys = list(config.parameter_grid)
    results: list[RunResult] = []
    summary: list[dict[str, Any]] = []

    for run_index, params in enumerate(parameter_sets):
        if grid_keys:
            parts = [f"{key}-{_slug(params[key])}" for key in grid_keys]
            run_name = "__".join(parts)
        else:
            run_name = "default"
        run_config = replace(
            config,
            model_params=params,
            parameter_grid={},
        )
        result = _run_prepared(
            dataset,
            run_config,
            prepared,
            destination / run_name,
            seed=seed,
            estimator_factory=estimator_factory,
        )
        results.append(result)
        summary.append(
            {
                "run_index": run_index,
                "run_name": run_name,
                "model_params": params,
                "metrics": dict(result.metrics),
            }
        )

    (destination / "sweep_summary.json").write_text(
        json.dumps(
            {
                "model": config.model,
                "dataset": dataset.logical_id,
                "parameter_grid": {
                    key: list(values)
                    for key, values in config.parameter_grid.items()
                },
                "fpr": config.fpr,
                "label_aware": True,
                "runs": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(results)
