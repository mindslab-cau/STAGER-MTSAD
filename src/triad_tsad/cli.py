"""Command-line launcher for STAGER experiments."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


_STAGER_MODEL_NAMES = frozenset({"stager", "triad"})


class TriADRunError(RuntimeError):
    """Raised when a STAGER experiment cannot be launched safely."""


@dataclass(frozen=True, slots=True)
class TriADPreset:
    """Validated path-free experiment settings."""

    config: Any
    dataset_id: str
    seeds: tuple[int, ...]
    seed_iter: int | None = None


@dataclass(slots=True)
class PreparedTriADData:
    """Legacy-compatible train/test datasets and aligned metadata."""

    train_dataset: Any
    test_dataset: Any
    labels: Any
    feature_names: tuple[str, ...]
    expected_period_seconds: int


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TriADRunError(f"{name} must be a mapping.")
    return dict(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise TriADRunError(
            "STAGER YAML configs require PyYAML. Install the 'yaml' extra."
        ) from exc

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TriADRunError(f"Failed to read config: {path}") from exc
    except yaml.YAMLError as exc:
        raise TriADRunError(f"Invalid YAML config: {path}") from exc
    return _mapping(document, "config root")


def _seed_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        values = value
    else:
        raise TriADRunError(f"{name} must be an integer or a sequence.")
    if not values:
        raise TriADRunError(f"{name} must not be empty.")
    seeds: list[int] = []
    for seed in values:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**32 - 1
        ):
            raise TriADRunError(
                f"Every {name} value must be an integer in [0, 2**32 - 1]."
            )
        seeds.append(int(seed))
    if len(set(seeds)) != len(seeds):
        raise TriADRunError(f"{name} must not contain duplicates.")
    return tuple(seeds)


def _positive_cli_int(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if resolved < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return resolved


def load_triad_preset(path: str | Path) -> TriADPreset:
    """Load a STAGER preset without resolving dataset paths."""

    return triad_preset_from_mapping(_load_yaml(Path(path)))


def triad_preset_from_mapping(document: Mapping[str, Any]) -> TriADPreset:
    """Validate a preset mapping, including a config expanded by a sweep."""

    from triad_tsad.models.triad.config import TriADConfig

    model = _mapping(document.get("model"), "model")
    model_name = str(model.get("name", "")).lower()
    if model_name not in _STAGER_MODEL_NAMES:
        raise TriADRunError("model.name must be STAGER (or legacy TriAD).")

    dataset = _mapping(document.get("dataset"), "dataset")
    dataset_id = dataset.get("logical_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise TriADRunError("dataset.logical_id must be a non-empty string.")

    triad_values = _mapping(document.get(model_name), model_name)
    try:
        config = TriADConfig.from_mapping(triad_values)
    except (TypeError, ValueError) as exc:
        raise TriADRunError(f"Invalid STAGER settings: {exc}") from exc

    explicit_fields = [
        (name, document[name])
        for name in ("seed", "seeds")
        if name in document
    ]
    if len(explicit_fields) > 1:
        raise TriADRunError("seed and seeds are mutually exclusive.")
    raw_seed_iter = document.get("seed_iter")
    if explicit_fields and raw_seed_iter is not None:
        raise TriADRunError(
            "Explicit seed values and seed_iter are mutually exclusive."
        )
    if not explicit_fields and raw_seed_iter is None:
        raise TriADRunError("Config must define seed, seeds, or seed_iter.")
    seeds = (
        _seed_tuple(explicit_fields[0][1], explicit_fields[0][0])
        if explicit_fields
        else ()
    )
    seed_iter = None
    if raw_seed_iter is not None:
        from triad_tsad.run_control import positive_count

        try:
            seed_iter = positive_count(raw_seed_iter, "seed_iter")
        except ValueError as exc:
            raise TriADRunError(str(exc)) from exc
    return TriADPreset(
        config=config,
        dataset_id=dataset_id,
        seeds=seeds,
        seed_iter=seed_iter,
    )


def _resolve_seeds(
    preset: TriADPreset,
    *,
    seed_overrides: Sequence[int] | None,
    seed_iter_override: int | None,
) -> tuple[tuple[int, ...], str]:
    from triad_tsad.run_control import generate_unique_seeds

    if seed_overrides is not None:
        return _seed_tuple(seed_overrides, "--seed"), "cli_seed"
    try:
        if seed_iter_override is not None:
            return generate_unique_seeds(seed_iter_override), "cli_seed_iter"
        if preset.seed_iter is not None:
            return generate_unique_seeds(preset.seed_iter), "config_seed_iter"
    except ValueError as exc:
        raise TriADRunError(str(exc)) from exc
    return preset.seeds, "config_seed"


def _positive_int(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TriADRunError(f"{name} must be a positive integer.")
    return int(value)


def load_triad_data(spec: Any, config: Any) -> PreparedTriADData:
    """Load SWaT, WADI, or HAI through the retained preprocessing boundary."""

    from triad_tsad.utils.data import load_dataset
    from triad_tsad.utils.dataset_paths import configure_dataset_spec

    train_key, test_key = configure_dataset_spec(spec)
    expected_period = _positive_int(
        spec.schema.get("expected_period_seconds"),
        "schema.expected_period_seconds",
        10,
    )
    common = {
        "window_size": config.window_size,
        "stride": config.stride,
        "time_lag": 1,
        "predict_size": 1,
        "normalize_data": True,
        "apply_ewm": True,
        "ewm_alpha": config.ewm_alpha,
        "downsampled": True,
        "downsample_interval": expected_period,
    }
    train_dataset, _, _, normalization_stats = load_dataset(
        train_key,
        **common,
    )
    if normalization_stats is None:
        raise TriADRunError(
            "Training preprocessing did not return normalization statistics."
        )
    test_dataset, labels, _, _ = load_dataset(
        test_key,
        normalization_stats=normalization_stats,
        **common,
    )
    if labels is None:
        raise TriADRunError("The selected test dataset did not provide labels.")
    if len(train_dataset) < 1 or len(test_dataset) < 1:
        raise TriADRunError(
            "Timestamp continuity filtering left no train or test windows."
        )

    train_names = tuple(train_dataset.feature_names)
    test_names = tuple(test_dataset.feature_names)
    if not train_names or train_names != test_names:
        raise TriADRunError(
            "Train and test feature names must be non-empty and identical."
        )
    expected_count = spec.schema.get("expected_feature_count")
    if expected_count is not None:
        expected_count = _positive_int(
            expected_count,
            "schema.expected_feature_count",
            len(train_names),
        )
        if len(train_names) != expected_count:
            raise TriADRunError(
                "Prepared feature count does not match the dataset manifest: "
                f"expected {expected_count}, found {len(train_names)}."
            )
    feature_order = spec.schema.get(
        "feature_order",
        spec.schema.get("feature_names"),
    )
    if feature_order is not None:
        if (
            not isinstance(feature_order, Sequence)
            or isinstance(feature_order, (str, bytes, bytearray))
        ):
            raise TriADRunError("schema.feature_order must be a sequence.")
        declared_order = tuple(str(value) for value in feature_order)
        if train_names != declared_order:
            raise TriADRunError(
                "Prepared feature order does not match the dataset manifest."
            )
    if len(labels) != len(test_dataset):
        raise TriADRunError(
            "Aligned test labels do not match the retained test windows."
        )
    return PreparedTriADData(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        labels=labels,
        feature_names=train_names,
        expected_period_seconds=expected_period,
    )


def _data_summary(
    preset: TriADPreset,
    prepared: PreparedTriADData,
) -> dict[str, Any]:
    train_rows = len(prepared.train_dataset.tag_values)
    test_rows = len(prepared.test_dataset.tag_values)
    train_candidates = max(0, train_rows - preset.config.window_size)
    test_candidates = max(0, test_rows - preset.config.window_size)
    return {
        "dataset": preset.dataset_id,
        "feature_count": len(prepared.feature_names),
        "expected_period_seconds": prepared.expected_period_seconds,
        "window_size": preset.config.window_size,
        "stride": preset.config.stride,
        "train": {
            "rows": train_rows,
            "candidate_windows": train_candidates,
            "valid_windows": len(prepared.train_dataset),
            "continuity_rejections": (
                train_candidates - len(prepared.train_dataset)
            ),
        },
        "test": {
            "rows": test_rows,
            "candidate_windows": test_candidates,
            "valid_windows": len(prepared.test_dataset),
            "continuity_rejections": (
                test_candidates - len(prepared.test_dataset)
            ),
        },
    }


def _selected_evaluation(
    result: Any,
    output_dir: Path,
    *,
    fpr: float,
) -> tuple[dict[str, float], Any, Any]:
    from triad_tsad.evaluation.evaluate import run_thresholding

    if result.labels is None:
        raise TriADRunError("Evaluation requires aligned binary test labels.")
    evaluation_results, threshold_dict, prediction_dict = run_thresholding(
        test_losses=result.score_streams,
        train_losses=None,
        test_labels=result.labels,
        threshold_methods=["bestf1"],
        threshold_config={"fpr_threshold": fpr},
        threshold_dir=str(output_dir),
    )
    result_key = f"bestf1_result_fpr{fpr}"
    artifact_key = f"bestf1_fpr{fpr}"
    if result_key not in evaluation_results:
        raise TriADRunError(
            f"Historical evaluator did not return {result_key!r}."
        )
    if artifact_key not in threshold_dict or artifact_key not in prediction_dict:
        raise TriADRunError(
            "Historical evaluator did not return thresholds and predictions."
        )
    metrics = {
        str(name): float(value)
        for name, value in evaluation_results[result_key].items()
    }
    return (
        metrics,
        threshold_dict[artifact_key],
        prediction_dict[artifact_key],
    )


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {
        str(name): tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def _run_one_seed(
    preset: TriADPreset,
    prepared: PreparedTriADData,
    output_dir: Path,
    *,
    seed: int,
    device: str | None,
) -> Any:
    import numpy as np
    import torch

    from triad_tsad.evaluation.artifacts import save_run_artifacts
    from triad_tsad.models.triad.runner import fit_score_triad_datasets

    execution = fit_score_triad_datasets(
        prepared.train_dataset,
        prepared.test_dataset,
        num_features=len(prepared.feature_names),
        target_indices=prepared.test_dataset.get_target_indices(),
        raw_test_length=len(prepared.test_dataset.tag_values),
        labels=prepared.labels,
        timestamps=prepared.test_dataset.ts,
        indices=np.arange(
            len(prepared.test_dataset.tag_values),
            dtype=np.int64,
        ),
        config=preset.config,
        seed=seed,
        device=device,
        dataset_id=preset.dataset_id,
        feature_names=prepared.feature_names,
    )
    metrics, thresholds, predictions = _selected_evaluation(
        execution.result,
        output_dir / "thresholding",
        fpr=preset.config.evaluation_fpr,
    )
    result = replace(execution.result, metrics=metrics)
    save_run_artifacts(
        result,
        output_dir,
        predictions=predictions,
        thresholds=thresholds,
    )
    torch.save(
        _cpu_state_dict(execution.model),
        output_dir / "best_model.pt",
    )
    return result


def _mean_std(values: Sequence[float]) -> dict[str, float | int | None]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": None if len(array) < 2 else float(array.std(ddof=1)),
        "n": len(array),
    }


def _summary(
    preset: TriADPreset,
    results: Sequence[Any],
    *,
    seed_mode: str = "config_seed",
    max_epochs: int | None = None,
) -> dict[str, Any]:
    metric_names = sorted(
        set.intersection(*(set(result.metrics) for result in results))
    )
    return {
        "dataset": preset.dataset_id,
        "seed_mode": seed_mode,
        "seeds": [result.seed for result in results],
        "max_epochs": max_epochs,
        "effective_epochs": preset.config.epochs,
        "metrics": {
            name: _mean_std([float(result.metrics[name]) for result in results])
            for name in metric_names
            if all(math.isfinite(float(result.metrics[name])) for result in results)
        },
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    from triad_tsad.run_control import add_sweep_options

    parser = argparse.ArgumentParser(
        prog="stager-run",
        description=(
            "Train and evaluate STAGER on SWaT, WADI, or HAI."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="STAGER preset or sweep YAML.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="Local dataset manifest containing prepared CSV paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the run or sweep.",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Run only this seed; repeat to run multiple overrides.",
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
        help="CLI-only cap for smoke tests; the preset file is not modified.",
    )
    parser.add_argument(
        "--device",
        help=(
            "PyTorch device. By default CUDA is used when available; "
            "CUDA_VISIBLE_DEVICES controls the visible physical GPU."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate data without training or evaluation.",
    )
    parser.add_argument("--sweep-run", action="store_true", help=argparse.SUPPRESS)
    add_sweep_options(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    document = _load_yaml(args.config)
    if "base_config" in document:
        from triad_tsad.sweep import run_launcher_sweep

        return run_launcher_sweep(args, document)
    if args.plan or args.resume or args.keep_going:
        parser.error("--plan, --resume, and --keep-going require a sweep config")
    preset = triad_preset_from_mapping(document)
    if args.max_epochs is not None:
        from triad_tsad.run_control import cap_epoch_loops

        preset = replace(
            preset,
            config=cap_epoch_loops(preset.config, args.max_epochs),
        )
    seeds, seed_mode = _resolve_seeds(
        preset,
        seed_overrides=args.seed,
        seed_iter_override=args.seed_iter,
    )

    from triad_tsad.config import load_dataset_manifest

    datasets = load_dataset_manifest(args.dataset_manifest)
    if preset.dataset_id not in datasets:
        available = ", ".join(sorted(datasets))
        raise TriADRunError(
            f"Dataset {preset.dataset_id!r} is absent from the manifest. "
            f"Available datasets: {available}"
        )
    prepared = load_triad_data(datasets[preset.dataset_id], preset.config)
    data_summary = _data_summary(preset, prepared)
    data_summary.update(
        {
            "seeds": list(seeds),
            "seed_mode": seed_mode,
            "max_epochs": args.max_epochs,
            "effective_epochs": preset.config.epochs,
        }
    )
    print(json.dumps(data_summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    output_dir = args.output_dir
    occupied = [
        output_dir / f"seed_{seed}"
        for seed in seeds
        if (output_dir / f"seed_{seed}").exists()
        and any((output_dir / f"seed_{seed}").iterdir())
    ]
    if occupied:
        joined = ", ".join(str(path) for path in occupied)
        raise TriADRunError(
            "Refusing to overwrite existing seed output directories: "
            f"{joined}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "seeds.json",
        {"seed_mode": seed_mode, "seeds": list(seeds)},
    )
    results = []
    for position, seed in enumerate(seeds, start=1):
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[STAGER] seed {seed} ({position}/{len(seeds)}) -> {seed_dir}",
            flush=True,
        )
        results.append(
            _run_one_seed(
                preset,
                prepared,
                seed_dir,
                seed=seed,
                device=args.device,
            )
        )
        if args.sweep_run:
            from triad_tsad.sweep import record_completed_seed

            record_completed_seed(seed_dir, seed)
    _write_json(
        output_dir / "summary.json",
        _summary(
            preset,
            results,
            seed_mode=seed_mode,
            max_epochs=args.max_epochs,
        ),
    )
    return 0


__all__ = [
    "PreparedTriADData",
    "TriADPreset",
    "TriADRunError",
    "build_parser",
    "load_triad_data",
    "load_triad_preset",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
