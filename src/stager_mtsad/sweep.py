"""Local grid sweeps around the STAGER and baseline launchers."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import itertools
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stager_mtsad import __version__
from stager_mtsad.cli import _STAGER_MODEL_NAMES, _seed_tuple, stager_preset_from_mapping
from stager_mtsad.run_control import add_sweep_options, generate_unique_seeds, positive_count


class SweepError(ValueError):
    """An invalid sweep or a conflicting output directory."""


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        result = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SweepError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise SweepError(f"Config must be a mapping: {path}")
    return result


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise SweepError(f"{field} must be a simple directory name, received {value!r}")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _json(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
    temporary.replace(path)


@dataclass(frozen=True)
class Combination:
    name: str
    parameters: dict[str, Any]
    document: dict[str, Any]


@dataclass(frozen=True)
class SweepSpec:
    name: str
    model: str
    dataset: str
    base: dict[str, Any]
    document: dict[str, Any]
    combinations: tuple[Combination, ...]
    seeds: tuple[int, ...] | None
    seed_iter: int | None


def _identity(document: Mapping[str, Any]) -> tuple[str, str]:
    model = document.get("model")
    model_name = str(model.get("name", "")).lower() if isinstance(model, Mapping) else ""
    if model_name in _STAGER_MODEL_NAMES:
        dataset = document.get("dataset")
        if not isinstance(dataset, Mapping):
            raise SweepError("dataset must be a mapping with logical_id")
        return model_name, _name(dataset.get("logical_id"), "dataset")
    from stager_mtsad.baseline_cli import _canonical_baseline, _orchestration_document

    run = document.get("run", {})
    if not isinstance(run, Mapping):
        raise SweepError("run must be a mapping")
    canonical = _canonical_baseline(run.get("baseline", document.get("baseline")))
    _, dataset, _, _ = _orchestration_document(document, canonical)
    return canonical, _name(dataset, "dataset")


def _assign(document: dict[str, Any], path: str, value: Any, model: str) -> None:
    parts = path.split(".")
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts):
        raise SweepError(f"Invalid parameter path: {path!r}")
    if model in _STAGER_MODEL_NAMES and (parts[0] != model or len(parts) != 2):
        raise SweepError(f"STAGER parameter paths must use {model}.<parameter>")
    if parts[0] in {"run", "dataset", "baseline"} or {"seed", "seeds", "seed_iter"}.intersection(parts):
        raise SweepError(f"Use sweep seed fields instead of sweeping {path!r}")
    target = document
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise SweepError(f"Parameter parent does not exist: {path!r}")
        target = target[part]
    target[parts[-1]] = copy.deepcopy(value)


def _validate(document: Mapping[str, Any], model: str) -> None:
    if model in _STAGER_MODEL_NAMES:
        stager_preset_from_mapping(document)
    else:
        from stager_mtsad.baseline_cli import _CONFIG_PARSERS, _orchestration_document

        payload, _, _, _ = _orchestration_document(document, model)
        config = _CONFIG_PARSERS[model](payload)
        if model in {"knn", "ocsvm"} and config.parameter_grid:
            raise SweepError("Move classical parameter_grid into sweep parameters")
        if model in {"knn", "ocsvm"} and config.model != model:
            raise SweepError("Baseline name conflicts with model")


def _with_seeds(document: Mapping[str, Any], model: str, seeds: Sequence[int]) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    for field in ("seed", "seeds", "seed_iter"):
        result.pop(field, None)
        if isinstance(result.get("run"), dict):
            result["run"].pop(field, None)
    if model in _STAGER_MODEL_NAMES:
        result["seeds"] = list(seeds)
    else:
        result.setdefault("run", {})["seeds"] = list(seeds)
    return result


def load_sweep(path: str | Path, *, document: dict[str, Any] | None = None) -> SweepSpec:
    path = Path(path).resolve()
    if document is None:
        document = read_yaml(path)
    unknown = set(document) - {"name", "base_config", "parameters", "seeds", "seed_iter"}
    if unknown:
        raise SweepError(f"Unknown sweep fields: {sorted(unknown)}")
    name = _name(document.get("name", path.stem), "name")
    base_path = document.get("base_config")
    if not isinstance(base_path, str) or not base_path.strip():
        raise SweepError("base_config must be a YAML path relative to the sweep file")
    base = read_yaml(path.parent / base_path)
    model, dataset = _identity(base)
    seeds = _seed_tuple(document["seeds"], "seeds") if "seeds" in document else None
    seed_iter = positive_count(document["seed_iter"], "seed_iter") if "seed_iter" in document else None
    if seeds is not None and seed_iter is not None:
        raise SweepError("seeds and seed_iter are mutually exclusive")
    if seeds is None and seed_iter is None:
        if model in _STAGER_MODEL_NAMES:
            preset = stager_preset_from_mapping(base)
            seeds, seed_iter = preset.seeds or None, preset.seed_iter
        else:
            from stager_mtsad.baseline_cli import _orchestration_document

            _, _, seeds, seed_iter = _orchestration_document(base, model)
            if seeds is None and seed_iter is None:
                raise SweepError("Define seeds or seed_iter for the sweep")
    parameters = document.get("parameters", {})
    if not isinstance(parameters, dict):
        raise SweepError("parameters must be a mapping of paths to value/values")
    fixed, axes = {}, {}
    for key, definition in parameters.items():
        if not isinstance(key, str) or not isinstance(definition, dict):
            raise SweepError("Each parameter requires {value: ...} or {values: [...]}")
        if set(definition) == {"value"}:
            fixed[key] = definition["value"]
        elif set(definition) == {"values"}:
            candidates = definition["values"]
            if not isinstance(candidates, list) or not candidates:
                raise SweepError(f"{key}.values must be a non-empty list")
            if len({_json(value) for value in candidates}) != len(candidates):
                raise SweepError(f"Duplicate candidates in {key}")
            axes[key] = candidates
        else:
            raise SweepError(f"{key} requires exactly one of value or values")
    paths = list(parameters)
    if any(other.startswith(key + ".") for key in paths for other in paths if other != key):
        raise SweepError("Parameter paths must not overlap")
    combinations = []
    for positions in itertools.product(*(range(len(values)) for values in axes.values())):
        selected = dict(fixed)
        selected.update({key: axes[key][index] for key, index in zip(axes, positions)})
        expanded = _with_seeds(base, model, seeds or (0,))
        for key, value in selected.items():
            _assign(expanded, key, value, model)
        if _identity(expanded) != (model, dataset):
            raise SweepError("Sweep parameters cannot change model or dataset identity")
        try:
            _validate(expanded, model)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SweepError(f"Invalid combination {selected}: {exc}") from exc
        folder = "__".join(
            f"{_axis_label(key, model)}-{index + 1}"
            for key, index in zip(axes, positions)
        ) or "fixed"
        if len(folder.encode()) > 240:
            raise SweepError("Combination directory name is too long; use fewer axes")
        combinations.append(Combination(folder, selected, expanded))
    return SweepSpec(name, model, dataset, base, document, tuple(combinations), seeds, seed_iter)


def _axis_label(path: str, model: str) -> str:
    if model in _STAGER_MODEL_NAMES:
        path = path.removeprefix(f"{model}.")
    return path.replace(".", "-")


def _seed_dir(root: Path, model: str, seed: int) -> Path:
    # Preserve each existing launcher's public layout.
    return root / (f"seed_{seed}" if model in _STAGER_MODEL_NAMES else f"seed-{seed}")


def _completed(path: Path, seed: int) -> bool:
    marker = path / "completed.json"
    if not marker.exists():
        return False
    if json.loads(marker.read_text()) != {"seed": seed} or not (path / "statistics.csv").is_file():
        raise SweepError(f"Invalid completed run: {path}")
    return True


def _tables(spec: SweepSpec, root: Path, seeds: Sequence[int]) -> None:
    # The launcher refreshes after every completed seed; the parent refreshes
    # when its subprocess exits. Serialize both writers before atomic replaces.
    with (root / "tables.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _collect_tables(spec, root, seeds)


def _collect_tables(spec: SweepSpec, root: Path, seeds: Sequence[int]) -> None:
    all_runs, all_summaries = [], []
    for combo in spec.combinations:
        directory = root / combo.name
        runs = []
        metrics_by_name: dict[str, list[float]] = {}
        for seed in seeds:
            seed_dir = _seed_dir(directory, spec.model, seed)
            if not _completed(seed_dir, seed):
                continue
            with (seed_dir / "statistics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1 or int(rows[0]["seed"]) != seed:
                raise SweepError(f"Invalid statistics CSV: {seed_dir}")
            runs.append({"combination": combo.name, **combo.parameters, **rows[0]})
            for name, value in json.loads((seed_dir / "metrics.json").read_text()).items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise SweepError(f"Non-finite metric {name!r}: {seed_dir}")
                metrics_by_name.setdefault(name, []).append(float(value))
        summary: dict[str, Any] = {
            "combination": combo.name, **combo.parameters,
            "completed_seeds": len(runs), "expected_seeds": len(seeds),
            "status": "completed" if len(runs) == len(seeds) else "incomplete",
        }
        metric_rows = []
        for name, values in sorted(metrics_by_name.items()):
            row = {
                "metric": name, "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values), "max": max(values), "n": len(values),
            }
            metric_rows.append(row)
            summary.update({f"{name}.{key}": row[key] for key in ("mean", "std", "min", "max", "n")})
        write_csv(directory / "runs.csv", runs)
        write_csv(directory / "summary.csv", metric_rows)
        _write_json(directory / "sweep_summary.json", summary)
        all_runs.extend(runs)
        all_summaries.append(summary)
    write_csv(root / "runs.csv", all_runs)
    write_csv(root / "summary.csv", all_summaries)


def refresh_sweep_tables(combination_dir: str | Path) -> bool:
    """Refresh a launcher's owning sweep from completed artifacts, if present.

    Read the saved plan so all configured seeds remain in the denominator,
    including during resume. Single-run output directories are left alone.
    This never starts a model, changes seeds, or rewrites seed artifacts.
    """

    directory = Path(combination_dir).resolve()
    root = directory.parent
    record_path = root / "sweep.json"
    if not record_path.is_file():
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    combinations = tuple(
        Combination(combo["name"], combo["parameters"], {})
        for combo in record["combinations"]
    )
    if directory.name not in {combo.name for combo in combinations}:
        return False
    seeds = _seed_tuple(record["seeds"], "saved sweep seeds")
    spec = SweepSpec(
        record["name"], record["model"], record["dataset"],
        record["base_config"], record["sweep_config"], combinations, seeds, None,
    )
    _tables(spec, root, seeds)
    return True


def record_completed_seed(seed_dir: Path, seed: int) -> None:
    """Write sweep-only statistics after a launcher saves its usual artifacts."""

    row: dict[str, Any] = {"seed": seed}

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}.{key}", item)
        else:
            row[prefix] = value

    for name in ("metrics", "config", "metadata"):
        payload = json.loads((seed_dir / f"{name}.json").read_text(encoding="utf-8"))
        flatten(name, payload)
    write_csv(seed_dir / "statistics.csv", [row])
    _write_json(seed_dir / "completed.json", {"seed": seed})
    refresh_sweep_tables(seed_dir.parent)


def _command(spec: SweepSpec, directory: Path, manifest: Path, seeds: Sequence[int], args: Any) -> list[str]:
    is_stager = spec.model in _STAGER_MODEL_NAMES
    module = "stager_mtsad.cli" if is_stager else "stager_mtsad.baseline_cli"
    command = [sys.executable, "-m", module, "--sweep-run"]
    if not is_stager:
        command.extend(["--baseline", spec.model])
    command.extend(["--config", str(directory / "config.yaml"), "--dataset-manifest", str(manifest), "--output-dir", str(directory)])
    for seed in seeds:
        command.extend(["--seed", str(seed)])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.max_epochs is not None:
        command.extend(["--max-epochs", str(args.max_epochs)])
    return command


def run_launcher_sweep(
    args: argparse.Namespace,
    document: dict[str, Any],
    *,
    baseline: str | None = None,
) -> int:
    """Run a sweep selected through a model launcher's config argument."""

    try:
        if args.seed is not None or args.seed_iter is not None:
            raise SweepError("Set seeds or seed_iter in the sweep YAML")
        if args.dry_run:
            raise SweepError("Use --plan to preview a sweep")
        if args.sweep_run:
            raise SweepError("A sweep child must receive a single-run preset")
        spec = load_sweep(args.config, document=document)
        if baseline is None:
            if spec.model not in _STAGER_MODEL_NAMES:
                raise SweepError(f"Use stager-baseline-run --baseline {spec.model} for this sweep")
        elif spec.model != baseline:
            raise SweepError(f"Sweep model {spec.model!r} conflicts with --baseline {baseline!r}")
        return run_sweep(args, spec=spec, output_dir=args.output_dir)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"stager-sweep: {exc}") from exc


def run_sweep(
    args: Any,
    *,
    launch: Any = subprocess.run,
    spec: SweepSpec | None = None,
    output_dir: Path | None = None,
) -> int:
    import yaml
    from stager_mtsad.config import load_dataset_manifest

    if spec is None:
        spec = load_sweep(args.config)
    manifest = args.dataset_manifest.resolve()
    if spec.dataset not in load_dataset_manifest(manifest):
        raise SweepError(f"Dataset {spec.dataset!r} is absent from the manifest")
    root = (
        args.output_root.resolve() / spec.model / spec.dataset / spec.name
        if output_dir is None else output_dir.resolve()
    )
    combinations = [
        {"name": combo.name, "parameters": combo.parameters}
        for combo in spec.combinations
    ]
    signature_inputs = {
        "stager-mtsad": __version__,
        "sweep": spec.document, "base": spec.base,
        "combinations": combinations,
        "manifest": {"path": str(manifest), "contents": manifest.read_text()},
        "max_epochs": args.max_epochs, "device": args.device,
    }
    if spec.model in _STAGER_MODEL_NAMES:
        signature_inputs["stager_evaluation_epoch"] = "last"
    signature = hashlib.sha256(_json(signature_inputs).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "sweep.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SweepError(f"Another sweep is running in {root}") from exc
        record_path = root / "sweep.json"
        if record_path.is_file():
            previous = json.loads(record_path.read_text(encoding="utf-8"))
        elif any(path.name not in {lock_path.name, "sweep.json.tmp"} for path in root.iterdir()):
            raise SweepError(f"Refusing to use an unrelated output directory: {root}")
        else:
            previous = None
        if previous is not None and previous["signature"] != signature:
            raise SweepError("Config/manifest/options changed; use a different sweep name")
        seeds = tuple(previous["seeds"]) if previous else spec.seeds or generate_unique_seeds(spec.seed_iter)
        existing_runs = any(
            _seed_dir(root / combo.name, spec.model, seed).exists()
            for combo in spec.combinations for seed in seeds
        )
        if existing_runs and not args.resume and not args.plan:
            raise SweepError("Sweep already has run outputs; use --resume or a different name")
        record = {
            "signature": signature, "name": spec.name, "model": spec.model,
            "dataset": spec.dataset, "seeds": list(seeds),
            "combinations": combinations,
            "combination_count": len(spec.combinations),
            "run_count": len(spec.combinations) * len(seeds),
            "source_config": str(args.config.resolve()),
            "sweep_config": spec.document,
            "base_config": spec.base,
            "dataset_manifest": str(manifest),
            "max_epochs": args.max_epochs, "device": args.device,
        }
        _write_json(record_path, record)
        _write_json(root / "seeds.json", {"seeds": list(seeds), "seed_mode": "sweep_seed_iter" if spec.seed_iter else "sweep_seeds"})
        for combo in spec.combinations:
            directory = root / combo.name
            directory.mkdir(parents=True, exist_ok=True)
            effective = _with_seeds(combo.document, spec.model, seeds)
            (directory / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
            _write_json(directory / "parameters.json", combo.parameters)
            full_command = _command(spec, directory, manifest, seeds, args)
            (directory / "command.txt").write_text(shlex.join(full_command) + "\n", encoding="utf-8")
        _tables(spec, root, seeds)
        failed = False
        for combo in spec.combinations:
            directory = root / combo.name
            pending = [seed for seed in seeds if not _completed(_seed_dir(directory, spec.model, seed), seed)]
            command = _command(spec, directory, manifest, pending, args)
            if args.plan or not pending:
                continue
            for seed in pending:
                incomplete = _seed_dir(directory, spec.model, seed)
                if incomplete.exists():
                    archive = directory / "incomplete" / f"{incomplete.name}-{uuid.uuid4().hex[:8]}"
                    archive.parent.mkdir(exist_ok=True)
                    shutil.move(str(incomplete), archive)
            print(f"[sweep] {combo.name}: {len(pending)} seeds -> {directory}", flush=True)
            with (directory / "run.log").open("a", encoding="utf-8") as log:
                log.write(shlex.join(command) + "\n")
                log.flush()
                try:
                    completed = launch(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        check=False, pass_fds=(lock.fileno(),),
                    )
                    returncode = completed.returncode
                except OSError as exc:
                    log.write(str(exc) + "\n")
                    returncode = 1
            _tables(spec, root, seeds)
            if returncode or any(not _completed(_seed_dir(directory, spec.model, seed), seed) for seed in seeds):
                failed = True
                print(f"[sweep] failed: see {directory / 'run.log'}", file=sys.stderr)
                if not args.keep_going:
                    break
        print(f"{len(spec.combinations)} combinations x {len(seeds)} seeds = {record['run_count']} runs -> {root}", flush=True)
        return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stager-sweep", description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Sweep YAML; its filename is the default sweep name.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device")
    parser.add_argument("--max-epochs", type=lambda value: positive_count(int(value), "max_epochs"))
    add_sweep_options(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_sweep(build_parser().parse_args(argv))
    except (SweepError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"stager-sweep: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
