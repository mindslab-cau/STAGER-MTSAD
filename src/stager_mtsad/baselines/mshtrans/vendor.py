"""Lazy loader for the bundled MSHTrans source tree."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class MSHTransDependencyError(RuntimeError):
    """Raised when bundled source or optional dependencies are unavailable."""


def _default_vendor_root() -> Path:
    candidate = Path(__file__).resolve().parent / "upstream"
    if (candidate / "networks").is_dir():
        return candidate
    raise MSHTransDependencyError(
        "bundled MSHTrans source was not found; pass vendor_root pointing to "
        "a directory that contains networks/"
    )


def resolve_vendor_root(vendor_root: str | Path | None = None) -> Path:
    """Resolve a portable source root without machine-specific defaults."""

    root = _default_vendor_root() if vendor_root is None else Path(vendor_root)
    root = root.expanduser().resolve()
    if not (root / "networks").is_dir():
        raise MSHTransDependencyError(
            f"MSHTrans vendor root does not contain networks/: {root}"
        )
    return root


def _load_networks_package(root: Path) -> ModuleType:
    package_directory = root / "networks"
    existing = sys.modules.get("networks")
    if existing is not None:
        existing_paths = {
            Path(path).resolve()
            for path in getattr(existing, "__path__", ())
        }
        if package_directory not in existing_paths:
            raise MSHTransDependencyError(
                "cannot load bundled MSHTrans source because another top-level "
                "'networks' package is already imported"
            )
        return existing

    spec = importlib.util.spec_from_file_location(
        "networks",
        package_directory / "__init__.py",
        submodule_search_locations=[str(package_directory)],
    )
    if spec is None or spec.loader is None:
        raise MSHTransDependencyError(
            "could not construct an import specification for MSHTrans"
        )
    package = importlib.util.module_from_spec(spec)
    sys.modules["networks"] = package
    try:
        spec.loader.exec_module(package)
    except Exception:
        sys.modules.pop("networks", None)
        raise
    return package


def load_external_model_class(
    vendor_root: str | Path | None = None,
) -> type[Any]:
    """Load the external model only after optional dependencies are requested."""

    root = resolve_vendor_root(vendor_root)
    _load_networks_package(root)

    try:
        module = importlib.import_module("networks.MSHTrans")
    except (ImportError, OSError) as exc:
        raise MSHTransDependencyError(
            "MSHTrans execution requires compatible installations of PyTorch, "
            "PyTorch Geometric, einops, and tqdm"
        ) from exc

    model_class = getattr(module, "MSHTrans", None)
    if not isinstance(model_class, type):
        raise MSHTransDependencyError(
            "bundled MSHTrans source does not expose MSHTrans"
        )
    return model_class
