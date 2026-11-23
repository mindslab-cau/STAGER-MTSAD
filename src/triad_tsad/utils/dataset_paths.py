"""Runtime dataset-path registration without embedded machine paths."""

from __future__ import annotations

from pathlib import Path

from triad_tsad.contracts import DatasetSpec


# This dictionary intentionally starts empty. ``configure_dataset_spec`` fills
# it from the manifest supplied with ``--dataset-manifest`` at run time.
# Users should edit configs/datasets.yaml, not this dictionary.
DATASET_PATHS: dict[str, str] = {}

# Left: public logical ID used in configs and commands.
# Right: retained internal key expected by the legacy-compatible data loader.
PUBLIC_DATASET_ALIASES = {
    "swat": "SWaTogskip_down_10",
    "swat_test": "SWaTogskip_Test_down_10",
    "wadi": "WADImodeskip_down_10",
    "wadi_test": "WADImodeskip_Test_down_10",
    "hai": "HAI_down_10",
    "hai_test": "HAI_Test_down_10",
}

_FAMILY_KEYS = {
    "swat": ("SWaTogskip_down_10", "SWaTogskip_Test_down_10"),
    "wadi": ("WADImodeskip_down_10", "WADImodeskip_Test_down_10"),
    "hai": ("HAI_down_10", "HAI_Test_down_10"),
}


def _family(logical_id: str) -> str:
    normalized = "".join(
        character for character in str(logical_id).lower() if character.isalnum()
    )
    if normalized.startswith("swat"):
        return "swat"
    if normalized.startswith("wadi"):
        return "wadi"
    if normalized.startswith("hai"):
        return "hai"
    raise ValueError(
        f"STAGER supports SWaT, WADI, and HAI dataset specs; received {logical_id!r}"
    )


def configure_dataset_spec(spec: DatasetSpec) -> tuple[str, str]:
    """Register one portable manifest entry for the legacy-compatible loader."""

    family = _family(spec.logical_id)
    train_key, test_key = _FAMILY_KEYS[family]
    DATASET_PATHS[train_key] = str(Path(spec.train_path))
    DATASET_PATHS[test_key] = str(Path(spec.test_path))
    return train_key, test_key


def resolve_dataset_name(name: str, test: bool = False) -> str:
    """Resolve a public logical name to the retained internal dataset key."""

    key = str(name)
    alias = key.lower()
    if test and not alias.endswith("_test"):
        alias = f"{alias}_test"
    return PUBLIC_DATASET_ALIASES.get(alias, key)


__all__ = [
    "DATASET_PATHS",
    "PUBLIC_DATASET_ALIASES",
    "configure_dataset_spec",
    "resolve_dataset_name",
]
