"""Separate command-line launcher for baseline experiments.

The STAGER launcher intentionally remains in :mod:`stager_mtsad.cli`.  This
module supplies a small orchestration boundary for heterogeneous baseline
runners while retaining one evaluation and artifact contract.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from stager_mtsad.config import load_dataset_manifest
from stager_mtsad.contracts import DatasetSpec, RunResult


EVALUATION_FPR = 0.002


class BaselineCLIError(RuntimeError):
    """Raised when a baseline run cannot be configured or launched safely."""


class BaselineAdapterUnavailable(BaselineCLIError):
    """Raised when a model lacks a complete ``DatasetSpec`` run adapter."""


@dataclass(frozen=True, slots=True)
class BaselinePlan:
    """Resolved inputs for one or more seed runs."""

    baseline: str
    dataset: DatasetSpec
    config: Any
    seeds: tuple[int, ...]
    output_dir: Path
    device: str | None
    seed_mode: str = "config_seed"
    max_epochs: int | None = None


ConfigParser = Callable[[Mapping[str, Any]], Any]
ModelRunner = Callable[[DatasetSpec, Any, int, str | None], RunResult]
ThresholdRunner = Callable[..., tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
ArtifactWriter = Callable[..., None]


def _read_document(path: Path) -> dict[str, Any]:
    """Read a portable config without importing model dependencies."""

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            with path.open("rb") as stream:
                document = tomllib.load(stream)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise BaselineCLIError(
                    "YAML baseline configs require PyYAML"
                ) from exc
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise BaselineCLIError(
                "baseline config must use .json, .toml, .yaml, or .yml"
            )
    except BaselineCLIError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise BaselineCLIError(f"failed to read baseline config: {path}") from exc
    if not isinstance(document, Mapping):
        raise BaselineCLIError("baseline config root must be a mapping")
    return dict(document)


def _canonical_baseline(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise BaselineCLIError("baseline must be a non-empty string")
    try:
        from stager_mtsad.baselines.registry import get_baseline_info

        return get_baseline_info(name).name
    except KeyError as exc:
        raise BaselineCLIError(str(exc)) from exc


def _optional_baseline(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        return _canonical_baseline(value)
    except BaselineCLIError as exc:
        raise BaselineCLIError(f"invalid {field_name}: {exc}") from exc


def _dataset_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("logical_id")
    if not isinstance(value, str) or not value.strip():
        raise BaselineCLIError(
            f"{field_name} must be a dataset ID or a mapping with logical_id"
        )
    return value.strip()


def _seed_tuple(value: Any, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise BaselineCLIError(f"{field_name} must be a non-empty sequence")
    if not value:
        raise BaselineCLIError(f"{field_name} must be a non-empty sequence")
    seeds: list[int] = []
    for seed in value:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**32 - 1
        ):
            raise BaselineCLIError(
                f"every {field_name} value must be an integer in "
                "[0, 2**32 - 1]"
            )
        seeds.append(int(seed))
    if len(set(seeds)) != len(seeds):
        raise BaselineCLIError(f"{field_name} must not contain duplicates")
    return tuple(seeds)


def _positive_cli_int(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if resolved < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return resolved


def _seed_iter(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    from stager_mtsad.run_control import positive_count

    try:
        return positive_count(value, field_name)
    except ValueError as exc:
        raise BaselineCLIError(str(exc)) from exc


def _orchestration_document(
    document: Mapping[str, Any],
    cli_baseline: str,
) -> tuple[
    dict[str, Any],
    str | None,
    tuple[int, ...] | None,
    int | None,
]:
    """Remove run metadata before handing config to a typed parser.

    Older baseline presets sometimes put ``baseline`` or ``dataset`` at the
    top level.  They are accepted as orchestration metadata too, which lets the
    CLI consume both wrapped and existing presets without weakening the typed
    model parsers.
    """

    payload = dict(document)
    raw_run = payload.pop("run", {})
    if raw_run is None:
        raw_run = {}
    if not isinstance(raw_run, Mapping):
        raise BaselineCLIError("run must be a mapping")
    # Only these three fields control orchestration. Model-specific parsers
    # receive the remaining configuration.

    top_baseline = payload.pop("baseline", None)
    run_baseline = raw_run.get("baseline")
    declared = [
        value
        for value in (
            _optional_baseline(top_baseline, "baseline"),
            _optional_baseline(run_baseline, "run.baseline"),
        )
        if value is not None
    ]
    if any(value != cli_baseline for value in declared):
        raise BaselineCLIError(
            "--baseline conflicts with the baseline declared in the config"
        )

    top_dataset = _dataset_id(payload.pop("dataset", None), "dataset")
    run_dataset = _dataset_id(raw_run.get("dataset"), "run.dataset")
    if top_dataset is not None and run_dataset is not None:
        if top_dataset != run_dataset:
            raise BaselineCLIError(
                "dataset and run.dataset must not specify different values"
            )
    selected_dataset = run_dataset or top_dataset

    top_seed = payload.pop("seed", None)
    top_seeds = payload.pop("seeds", None)
    explicit_candidates = [
        value
        for value in (
            _seed_tuple(
                None if top_seed is None else [top_seed],
                "seed",
            ),
            _seed_tuple(top_seeds, "seeds"),
            _seed_tuple(
                None
                if raw_run.get("seed") is None
                else [raw_run.get("seed")],
                "run.seed",
            ),
            _seed_tuple(raw_run.get("seeds"), "run.seeds"),
        )
        if value is not None
    ]
    configured_seeds = explicit_candidates[0] if explicit_candidates else None
    if any(value != configured_seeds for value in explicit_candidates[1:]):
        raise BaselineCLIError(
            "seed/seeds fields must not specify different values"
        )
    seed_iter_candidates = [
        value
        for value in (
            _seed_iter(payload.pop("seed_iter", None), "seed_iter"),
            _seed_iter(raw_run.get("seed_iter"), "run.seed_iter"),
        )
        if value is not None
    ]
    configured_seed_iter = (
        seed_iter_candidates[0] if seed_iter_candidates else None
    )
    if any(
        value != configured_seed_iter for value in seed_iter_candidates[1:]
    ):
        raise BaselineCLIError(
            "seed_iter and run.seed_iter must not specify different values"
        )
    if configured_seeds is not None and configured_seed_iter is not None:
        raise BaselineCLIError(
            "explicit seed values and seed_iter are mutually exclusive"
        )
    return (
        payload,
        selected_dataset,
        configured_seeds,
        configured_seed_iter,
    )


def _parse_classical(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.classical.config import config_from_mapping

    return config_from_mapping(document)


def _parse_tsmixer(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.tsmixer.config import config_from_mapping

    return config_from_mapping(document)


def _parse_usad(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.usad.config import config_from_mapping

    return config_from_mapping(document)


def _parse_fusagnet(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.fusagnet.config import FuSAGNetConfig

    unknown = sorted(
        set(document) - {"layout", "model", "training", "false_positive_rate"}
    )
    if unknown:
        raise ValueError(
            f"configuration contains unknown fields: {', '.join(unknown)}"
        )
    return FuSAGNetConfig.from_mapping(document)


def _parse_deep_svdd(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.deep_svdd.config import config_from_mapping

    return config_from_mapping(document)


def _parse_mtscid(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.mtscid.config import config_from_mapping

    return config_from_mapping(document)


def _parse_mshtrans(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.mshtrans.config import MSHTransConfig

    unknown = sorted(
        set(document)
        - {
            "model",
            "training",
            "stride",
            "train_stride",
            "test_stride",
            "false_positive_rate",
        }
    )
    if unknown:
        raise ValueError(
            f"configuration contains unknown fields: {', '.join(unknown)}"
        )
    return MSHTransConfig.from_mapping(document)


def _parse_anomaly_transformer(document: Mapping[str, Any]) -> Any:
    from stager_mtsad.baselines.anomaly_transformer.cli_adapter import (
        config_from_mapping,
    )

    return config_from_mapping(document)


_CONFIG_PARSERS: dict[str, ConfigParser] = {
    "knn": _parse_classical,
    "ocsvm": _parse_classical,
    "tsmixer": _parse_tsmixer,
    "anomaly_transformer": _parse_anomaly_transformer,
    "usad": _parse_usad,
    "fusagnet": _parse_fusagnet,
    "deep_svdd": _parse_deep_svdd,
    "mtscid": _parse_mtscid,
    "mshtrans": _parse_mshtrans,
}


def _config_seed(config: Any) -> int:
    seed = getattr(config, "seed", None)
    if seed is None:
        seed = getattr(getattr(config, "training", None), "seed", None)
    if seed is None:
        return 0
    resolved = _seed_tuple([seed], "config seed")
    assert resolved is not None
    return resolved[0]


def _seeded_config(config: Any, seed: int) -> Any:
    """Override either a direct or nested dataclass seed when available."""

    if is_dataclass(config) and hasattr(config, "seed"):
        return replace(config, seed=seed)
    training = getattr(config, "training", None)
    if (
        is_dataclass(config)
        and is_dataclass(training)
        and hasattr(training, "seed")
    ):
        return replace(config, training=replace(training, seed=seed))
    return config


def _torch_device(requested: str | None) -> str:
    """Select the visible CUDA device by default, falling back to CPU."""

    if requested is not None:
        return requested
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise BaselineCLIError(
            "the selected baseline requires a compatible PyTorch installation"
        ) from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_plan(
    *,
    baseline: str,
    config_path: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    seed_overrides: Sequence[int] | None = None,
    seed_iter_override: int | None = None,
    max_epochs: int | None = None,
    device: str | None = None,
    config_document: Mapping[str, Any] | None = None,
) -> BaselinePlan:
    """Resolve and validate config, manifest, dataset, and seed selection."""

    canonical = _canonical_baseline(baseline)
    document = (
        _read_document(Path(config_path))
        if config_document is None else config_document
    )
    (
        payload,
        dataset_id,
        configured_seeds,
        configured_seed_iter,
    ) = _orchestration_document(
        document,
        canonical,
    )
    try:
        parser = _CONFIG_PARSERS[canonical]
    except KeyError as exc:  # Defensive: registry and launcher must stay aligned.
        raise BaselineCLIError(
            f"no typed config parser is registered for {canonical}"
        ) from exc
    try:
        typed_config = parser(payload)
    except BaselineCLIError:
        raise
    except (TypeError, ValueError) as exc:
        raise BaselineCLIError(
            f"invalid {canonical} configuration: {exc}"
        ) from exc

    if canonical in {"knn", "ocsvm"}:
        configured_model = getattr(typed_config, "model", None)
        if configured_model != canonical:
            raise BaselineCLIError(
                f"--baseline {canonical!r} conflicts with config model "
                f"{configured_model!r}"
            )
        if typed_config.parameter_grid:
            raise BaselineAdapterUnavailable(
                "stager-baseline-run does not expand classical parameter_grid "
                "configurations; select one parameter set first"
            )

    effective_max_epochs = None
    if max_epochs is not None:
        from stager_mtsad.run_control import cap_epoch_loops, positive_count

        try:
            resolved_max_epochs = positive_count(max_epochs, "max_epochs")
        except ValueError as exc:
            raise BaselineCLIError(str(exc)) from exc
        if canonical not in {"knn", "ocsvm"}:
            typed_config = cap_epoch_loops(typed_config, resolved_max_epochs)
            effective_max_epochs = resolved_max_epochs

    try:
        datasets = load_dataset_manifest(dataset_manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        raise BaselineCLIError(
            f"invalid dataset manifest: {exc}"
        ) from exc
    if dataset_id is None:
        if len(datasets) != 1:
            raise BaselineCLIError(
                "the config must select run.dataset when the manifest contains "
                "multiple datasets"
            )
        dataset_id = next(iter(datasets))
    try:
        dataset = datasets[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(datasets))
        raise BaselineCLIError(
            f"dataset {dataset_id!r} is absent from the manifest; "
            f"available: {available}"
        ) from exc

    override_seeds = _seed_tuple(seed_overrides, "--seed")
    if override_seeds is not None and seed_iter_override is not None:
        raise BaselineCLIError("--seed and --seed-iter are mutually exclusive")
    from stager_mtsad.run_control import generate_unique_seeds

    if override_seeds is not None:
        seeds = override_seeds
        seed_mode = "cli_seed"
    elif seed_iter_override is not None:
        try:
            seeds = generate_unique_seeds(seed_iter_override)
        except ValueError as exc:
            raise BaselineCLIError(str(exc)) from exc
        seed_mode = "cli_seed_iter"
    elif configured_seeds is not None:
        seeds = configured_seeds
        seed_mode = "config_seed"
    elif configured_seed_iter is not None:
        try:
            seeds = generate_unique_seeds(configured_seed_iter)
        except ValueError as exc:
            raise BaselineCLIError(str(exc)) from exc
        seed_mode = "config_seed_iter"
    else:
        seeds = (_config_seed(typed_config),)
        seed_mode = "model_config_seed"
    return BaselinePlan(
        baseline=canonical,
        dataset=dataset,
        config=typed_config,
        seeds=seeds,
        output_dir=Path(output_dir),
        device=device,
        seed_mode=seed_mode,
        max_epochs=effective_max_epochs,
    )


def _run_classical(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    del device
    import numpy as np

    from stager_mtsad.baselines.classical.data import load_classical_csv
    from stager_mtsad.baselines.classical.models import build_estimator

    np.random.seed(seed)
    prepared = load_classical_csv(dataset, ewm_alpha=config.ewm_alpha)
    estimator = build_estimator(config.model, config.model_params)
    estimator.fit(prepared.train_values)
    raw_scores = np.asarray(
        estimator.decision_function(prepared.test_values),
        dtype=np.float64,
    )
    scores = raw_scores.reshape(-1) if raw_scores.size else raw_scores
    if scores.ndim != 1 or len(scores) != len(prepared.test_values):
        raise BaselineCLIError(
            "classical decision_function must return one score per test row"
        )
    if not np.all(np.isfinite(scores)):
        raise BaselineCLIError("classical decision_function returned non-finite scores")
    portable_config = config.as_dict()
    portable_config["dataset"] = {"logical_id": dataset.logical_id}
    return RunResult(
        point_scores=scores,
        labels=prepared.labels,
        timestamps=prepared.timestamps,
        indices=prepared.indices,
        metrics={
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        },
        metadata={
            "model": config.model,
            "dataset": dataset.logical_id,
            "score_semantics": RunResult.SCORE_SEMANTICS,
            "feature_names": list(prepared.feature_names),
            "train_shape": list(prepared.train_values.shape),
            "test_shape": list(prepared.test_values.shape),
        },
        seed=seed,
        config=portable_config,
    )


def _run_tsmixer(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    from stager_mtsad.baselines.tsmixer.runner import run_tsmixer

    return run_tsmixer(
        dataset,
        _seeded_config(config, seed),
        device=_torch_device(device),
    )


def _run_anomaly_transformer(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    from stager_mtsad.baselines.anomaly_transformer.cli_adapter import (
        run_anomaly_transformer,
    )

    return run_anomaly_transformer(
        dataset,
        _seeded_config(config, seed),
        device=device or "auto",
    )


def _run_usad(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    from stager_mtsad.baselines.usad.runner import run_usad

    return run_usad(
        dataset,
        _seeded_config(config, seed),
        device=_torch_device(device),
    )


def _run_fusagnet(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    from stager_mtsad.baselines.fusagnet.cli_adapter import run_fusagnet

    return run_fusagnet(
        dataset,
        config,
        seed=seed,
        device=device or "auto",
    )


def _run_deep_svdd(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    from stager_mtsad.baselines.deep_svdd.runner import run_deep_svdd

    return run_deep_svdd(
        dataset,
        _seeded_config(config, seed),
        device=_torch_device(device),
    )


def _run_mtscid(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    from stager_mtsad.baselines.mtscid.runner import run_mtscid

    return run_mtscid(
        dataset,
        _seeded_config(config, seed),
        device=_torch_device(device),
    )


def _run_mshtrans(
    dataset: DatasetSpec,
    config: Any,
    seed: int,
    device: str | None,
) -> RunResult:
    from stager_mtsad.baselines.mshtrans.cli_adapter import run_mshtrans

    return run_mshtrans(
        dataset,
        _seeded_config(config, seed),
        device=device or "auto",
    )


_MODEL_RUNNERS: dict[str, ModelRunner] = {
    "knn": _run_classical,
    "ocsvm": _run_classical,
    "tsmixer": _run_tsmixer,
    "anomaly_transformer": _run_anomaly_transformer,
    "usad": _run_usad,
    "fusagnet": _run_fusagnet,
    "deep_svdd": _run_deep_svdd,
    "mtscid": _run_mtscid,
    "mshtrans": _run_mshtrans,
}


def _default_threshold_runner() -> ThresholdRunner:
    from stager_mtsad.evaluation.evaluate import run_thresholding

    return run_thresholding


def _default_artifact_writer() -> ArtifactWriter:
    from stager_mtsad.evaluation.artifacts import save_run_artifacts

    return save_run_artifacts


def evaluate_and_save(
    result: RunResult,
    output_dir: str | Path,
    *,
    threshold_runner: ThresholdRunner | None = None,
    artifact_writer: ArtifactWriter | None = None,
) -> RunResult:
    """Evaluate feature streams at FPR 0.002 and write canonical artifacts."""

    if not isinstance(result, RunResult):
        raise BaselineCLIError("model runner must return RunResult")
    if result.labels is None:
        raise BaselineCLIError(
            "common baseline evaluation requires aligned binary labels"
        )
    destination = Path(output_dir)
    threshold_runner = threshold_runner or _default_threshold_runner()
    artifact_writer = artifact_writer or _default_artifact_writer()
    evaluation, thresholds, predictions = threshold_runner(
        test_losses=result.score_streams,
        train_losses=None,
        test_labels=result.labels,
        threshold_methods=["bestf1"],
        threshold_config={"fpr_threshold": EVALUATION_FPR},
        threshold_dir=str(destination / "thresholding"),
    )
    result_key = f"bestf1_result_fpr{EVALUATION_FPR}"
    artifact_key = f"bestf1_fpr{EVALUATION_FPR}"
    if result_key not in evaluation:
        raise BaselineCLIError(
            f"common evaluator did not return {result_key!r}"
        )
    if artifact_key not in thresholds or artifact_key not in predictions:
        raise BaselineCLIError(
            "common evaluator did not return thresholds and predictions"
        )
    common_metrics = {
        str(name): float(value)
        for name, value in evaluation[result_key].items()
    }
    evaluated = replace(
        result,
        metrics={**dict(result.metrics), **common_metrics},
    )
    artifact_writer(
        evaluated,
        destination,
        predictions=predictions[artifact_key],
        thresholds=thresholds[artifact_key],
    )
    return evaluated


def execute_plan(
    plan: BaselinePlan,
    *,
    model_runner: ModelRunner | None = None,
    threshold_runner: ThresholdRunner | None = None,
    artifact_writer: ArtifactWriter | None = None,
    sweep_run: bool = False,
) -> tuple[RunResult, ...]:
    """Execute each seed through one model, evaluator, and artifact boundary."""

    if model_runner is None:
        try:
            model_runner = _MODEL_RUNNERS[plan.baseline]
        except KeyError as exc:
            raise BaselineCLIError(
                f"no model runner is registered for {plan.baseline}"
            ) from exc
    occupied = [
        plan.output_dir / f"seed-{seed}"
        for seed in plan.seeds
        if (plan.output_dir / f"seed-{seed}").exists()
        and any((plan.output_dir / f"seed-{seed}").iterdir())
    ]
    if occupied:
        joined = ", ".join(str(path) for path in occupied)
        raise BaselineCLIError(
            "refusing to overwrite existing seed output directories: "
            f"{joined}"
        )
    results: list[RunResult] = []
    for seed in plan.seeds:
        seed_dir = plan.output_dir / f"seed-{seed}"
        result = model_runner(
            plan.dataset,
            plan.config,
            seed,
            plan.device,
        )
        if result.seed != seed:
            raise BaselineCLIError(
                f"model runner returned seed {result.seed}, expected {seed}"
            )
        results.append(
            evaluate_and_save(
                result,
                seed_dir,
                threshold_runner=threshold_runner,
                artifact_writer=artifact_writer,
            )
        )
        if sweep_run:
            from stager_mtsad.sweep import record_completed_seed

            record_completed_seed(seed_dir, seed)
    return tuple(results)


def _summary(plan: BaselinePlan, results: Sequence[RunResult]) -> dict[str, Any]:
    common_metric_names = (
        set.intersection(*(set(result.metrics) for result in results))
        if results
        else set()
    )
    aggregate_metrics = {}
    for name in sorted(common_metric_names):
        values = [float(result.metrics[name]) for result in results]
        aggregate_metrics[str(name)] = {
            "mean": fmean(values),
            "std": pstdev(values),
            "count": len(values),
        }
    return {
        "baseline": plan.baseline,
        "dataset": plan.dataset.logical_id,
        "device": plan.device or "auto",
        "seed_mode": plan.seed_mode,
        "seeds": list(plan.seeds),
        "max_epochs": plan.max_epochs,
        "aggregate_metrics": aggregate_metrics,
        "runs": [
            {
                "seed": result.seed,
                "metrics": {
                    str(name): float(value)
                    for name, value in result.metrics.items()
                },
            }
            for result in results
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    from stager_mtsad.run_control import add_sweep_options

    parser = argparse.ArgumentParser(
        prog="stager-baseline-run",
        description=(
            "Train and evaluate one anomaly detection baseline."
        ),
    )
    parser.add_argument("--baseline", required=True, help="Baseline name or alias.")
    parser.add_argument("--config", type=Path, required=True, help="Baseline preset or sweep YAML.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Override configured seeds; repeat for multiple runs.",
    )
    seed_group.add_argument(
        "--seed-iter",
        type=_positive_cli_int,
        help=(
            "Run this many times with unique OS-generated 32-bit seeds. "
            "Mutually exclusive with --seed."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=_positive_cli_int,
        help=(
            "CLI-only epoch cap for neural baseline smoke tests; classical "
            "baselines are unchanged."
        ),
    )
    parser.add_argument(
        "--device",
        help=(
            "PyTorch device such as cpu, cuda, or cuda:0. By default the "
            "visible CUDA device is used when available."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved plan without training or writing files.",
    )
    parser.add_argument("--sweep-run", action="store_true", help=argparse.SUPPRESS)
    add_sweep_options(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        baseline = _canonical_baseline(args.baseline)
        document = _read_document(args.config)
        if "base_config" in document:
            from stager_mtsad.sweep import run_launcher_sweep

            return run_launcher_sweep(args, document, baseline=baseline)
        if args.plan or args.resume or args.keep_going:
            parser.error("--plan, --resume, and --keep-going require a sweep config")
        plan = build_plan(
            baseline=baseline,
            config_path=args.config,
            dataset_manifest_path=args.dataset_manifest,
            output_dir=args.output_dir,
            seed_overrides=args.seed,
            seed_iter_override=args.seed_iter,
            max_epochs=args.max_epochs,
            device=args.device,
            config_document=document,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "baseline": plan.baseline,
                        "dataset": plan.dataset.logical_id,
                        "seed_mode": plan.seed_mode,
                        "seeds": list(plan.seeds),
                        "max_epochs": plan.max_epochs,
                        "device": plan.device or "auto",
                        "output_dir": str(plan.output_dir),
                        "dry_run": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        results = execute_plan(plan, sweep_run=args.sweep_run)
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        summary = _summary(plan, results)
        (plan.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (plan.output_dir / "seeds.json").write_text(
            json.dumps(
                {
                    "seed_mode": plan.seed_mode,
                    "seeds": list(plan.seeds),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaselineCLIError as exc:
        raise SystemExit(f"stager-baseline-run: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
