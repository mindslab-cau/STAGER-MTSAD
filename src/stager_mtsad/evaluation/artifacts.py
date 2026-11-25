"""Portable artifact serialization for baseline runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from stager_mtsad.contracts import RunResult


class ArtifactError(ValueError):
    """Raised when a run artifact cannot be serialized safely."""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ArtifactError(f"unsupported JSON artifact value: {type(value).__name__}")


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(document), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_run_artifacts(
    result: RunResult,
    output_dir: str | Path,
    *,
    predictions: Any | None = None,
    thresholds: Any | None = None,
) -> None:
    """Save the stable artifact set for one run.

    The caller must always provide ``output_dir``. Dataset file paths are not
    inferred or embedded by this serializer.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # ``scores.npy`` is the primary historical evaluator input: one row per
    # sensor/score stream and one column per aligned time point. Never replace
    # it with the feature mean for multivariate neural baselines.
    np.save(destination / "scores.npy", result.score_streams)
    np.save(
        destination / "point_scores.npy",
        np.asarray(result.point_scores),
    )
    if result.feature_scores is not None:
        np.save(
            destination / "feature_scores.npy",
            np.asarray(result.feature_scores),
        )
    if result.labels is not None:
        np.save(destination / "labels.npy", np.asarray(result.labels))
    if result.timestamps is not None:
        np.save(
            destination / "timestamps.npy",
            np.asarray(result.timestamps),
            allow_pickle=True,
        )
    if result.indices is not None:
        np.save(destination / "indices.npy", np.asarray(result.indices))
    if predictions is not None:
        np.save(destination / "predictions.npy", np.asarray(predictions))
    if thresholds is not None:
        np.save(destination / "thresholds.npy", np.asarray(thresholds))

    _write_json(destination / "metrics.json", dict(result.metrics))
    _write_json(destination / "config.json", dict(result.config))
    _write_json(destination / "seed.json", {"seed": result.seed})
    _write_json(destination / "metadata.json", dict(result.metadata))
