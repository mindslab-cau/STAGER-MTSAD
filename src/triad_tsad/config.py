"""Load portable dataset specifications from external manifest files."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import ContractError, DatasetSpec


class ManifestError(ValueError):
    """Raised when a dataset manifest cannot be loaded or validated."""


def _read_manifest(path: Path) -> Mapping[str, Any]:
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
                raise ManifestError(
                    "YAML manifests require the optional 'yaml' dependency"
                ) from exc
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise ManifestError(
                "dataset manifest must use a .json, .toml, .yaml, or .yml extension"
            )
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"failed to read dataset manifest: {path}") from exc

    if not isinstance(document, Mapping):
        raise ManifestError("dataset manifest root must be a mapping")
    return document


def _column_value(
    entry: Mapping[str, Any],
    short_name: str,
    explicit_name: str,
) -> Any:
    if short_name in entry and explicit_name in entry:
        if entry[short_name] != entry[explicit_name]:
            raise ManifestError(
                f"{short_name} and {explicit_name} must not specify different values"
            )
    return entry.get(short_name, entry.get(explicit_name))


def _resolve_path(raw_path: Any, manifest_directory: Path, field_name: str) -> Path:
    if isinstance(raw_path, str) and not raw_path.strip():
        raise ManifestError(f"{field_name} must not be empty")
    try:
        path = Path(raw_path)
    except TypeError as exc:
        raise ManifestError(f"{field_name} must be a path-like value") from exc
    if not path.is_absolute():
        path = manifest_directory / path
    return path


def _dataset_spec(
    logical_id: Any,
    entry: Mapping[str, Any],
    manifest_directory: Path,
) -> DatasetSpec:
    if "train_path" not in entry or "test_path" not in entry:
        raise ManifestError(
            f"dataset {logical_id!r} must define train_path and test_path"
        )

    try:
        return DatasetSpec(
            logical_id=logical_id,
            train_path=_resolve_path(
                entry["train_path"], manifest_directory, "train_path"
            ),
            test_path=_resolve_path(
                entry["test_path"], manifest_directory, "test_path"
            ),
            timestamp=_column_value(entry, "timestamp", "timestamp_column"),
            label=_column_value(entry, "label", "label_column"),
            schema=entry.get("schema", {}),
            metadata=entry.get("metadata", {}),
        )
    except ContractError as exc:
        raise ManifestError(f"invalid dataset {logical_id!r}: {exc}") from exc


def load_dataset_manifest(path: str | Path) -> dict[str, DatasetSpec]:
    """Load dataset locations without reading any dataset contents.

    Relative train and test paths are interpreted relative to the manifest.
    The manifest must contain a ``datasets`` mapping or list. Mapping keys act
    as logical IDs; list entries must each provide ``logical_id``.
    """

    manifest_path = Path(path)
    document = _read_manifest(manifest_path)
    if "datasets" not in document:
        raise ManifestError("dataset manifest must contain a datasets section")

    raw_datasets = document["datasets"]
    manifest_directory = manifest_path.parent
    result: dict[str, DatasetSpec] = {}

    if isinstance(raw_datasets, Mapping):
        entries: list[tuple[Any, Any]] = list(raw_datasets.items())
    elif isinstance(raw_datasets, Sequence) and not isinstance(
        raw_datasets, (str, bytes, bytearray)
    ):
        entries = []
        for position, entry in enumerate(raw_datasets):
            if not isinstance(entry, Mapping):
                raise ManifestError(
                    f"datasets entry {position} must be a mapping"
                )
            if "logical_id" not in entry:
                raise ManifestError(
                    f"datasets entry {position} must define logical_id"
                )
            entries.append((entry["logical_id"], entry))
    else:
        raise ManifestError("datasets must be a mapping or list")

    for logical_id, raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ManifestError(f"dataset {logical_id!r} must be a mapping")
        if "logical_id" in raw_entry and raw_entry["logical_id"] != logical_id:
            raise ManifestError(
                f"dataset key {logical_id!r} conflicts with its logical_id"
            )
        spec = _dataset_spec(logical_id, raw_entry, manifest_directory)
        if spec.logical_id in result:
            raise ManifestError(f"duplicate dataset logical_id: {spec.logical_id!r}")
        result[spec.logical_id] = spec

    if not result:
        raise ManifestError("dataset manifest must define at least one dataset")
    return result
