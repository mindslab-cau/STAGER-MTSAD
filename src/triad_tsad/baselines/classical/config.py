"""Configuration objects for the unified KNN and OCSVM runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEGACY_FPR = 0.002
EVALUATION_PROTOCOL = "evaluate_ensemble_best_modified"


class ClassicalConfigError(ValueError):
    """Raised when a classical baseline configuration is invalid."""


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ClassicalConfigError(f"{field_name} must be a mapping")
    return dict(value)


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ClassicalConfigError(
            f"{field_name} contains unknown fields: {', '.join(unknown)}"
        )


@dataclass(frozen=True, slots=True)
class ClassicalRunConfig:
    """One model configuration, optionally expanded by a parameter grid."""

    model: str
    model_params: Mapping[str, Any] = field(default_factory=dict)
    parameter_grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    ewm_alpha: float = 0.8
    fpr: float = LEGACY_FPR
    evaluation_protocol: str = EVALUATION_PROTOCOL
    require_etapr: bool = True

    def __post_init__(self) -> None:
        normalized_model = self.model.lower().replace("-", "").replace("_", "")
        if normalized_model == "knn":
            normalized_model = "knn"
        elif normalized_model in {"ocsvm", "oneclasssvm"}:
            normalized_model = "ocsvm"
        else:
            raise ClassicalConfigError("model must be 'knn' or 'ocsvm'")
        object.__setattr__(self, "model", normalized_model)

        params = _mapping(self.model_params, "model_params")
        grid = _mapping(self.parameter_grid, "parameter_grid")
        normalized_grid: dict[str, tuple[Any, ...]] = {}
        for key, values in grid.items():
            if not isinstance(key, str) or not key:
                raise ClassicalConfigError(
                    "parameter_grid keys must be non-empty strings"
                )
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ClassicalConfigError(
                    f"parameter_grid[{key!r}] must be a sequence"
                )
            if not values:
                raise ClassicalConfigError(
                    f"parameter_grid[{key!r}] must not be empty"
                )
            normalized_grid[key] = tuple(values)
        if set(params) & set(normalized_grid):
            overlap = ", ".join(sorted(set(params) & set(normalized_grid)))
            raise ClassicalConfigError(
                f"parameters cannot appear in both model_params and grid: {overlap}"
            )

        if not 0.0 < float(self.ewm_alpha) <= 1.0:
            raise ClassicalConfigError("ewm_alpha must be in (0, 1]")
        if abs(float(self.fpr) - LEGACY_FPR) > 1e-12:
            raise ClassicalConfigError(
                f"classical baseline FPR is fixed to {LEGACY_FPR}"
            )
        if self.evaluation_protocol != EVALUATION_PROTOCOL:
            raise ClassicalConfigError(
                f"evaluation_protocol must be {EVALUATION_PROTOCOL!r}"
            )
        if self.require_etapr is not True:
            raise ClassicalConfigError(
                "require_etapr must be true for the historical evaluator"
            )

        object.__setattr__(self, "model_params", params)
        object.__setattr__(self, "parameter_grid", normalized_grid)
        object.__setattr__(self, "ewm_alpha", float(self.ewm_alpha))
        object.__setattr__(self, "fpr", float(self.fpr))

    def as_dict(self) -> dict[str, Any]:
        """Return the portable configuration recorded in run artifacts."""

        return {
            "model": self.model,
            "model_params": dict(self.model_params),
            "parameter_grid": {
                key: list(values) for key, values in self.parameter_grid.items()
            },
            "preprocessing": {
                "normalization": "train_minmax",
                "ewm_alpha": self.ewm_alpha,
                "window_size": 1,
            },
            "evaluation": {
                "protocol": self.evaluation_protocol,
                "fpr": self.fpr,
                "label_aware": True,
                "require_etapr": self.require_etapr,
            },
        }


def config_from_mapping(document: Mapping[str, Any]) -> ClassicalRunConfig:
    """Build a validated configuration from its serialized representation."""

    root = _mapping(document, "configuration")
    preprocessing = _mapping(root.get("preprocessing", {}), "preprocessing")
    evaluation = _mapping(root.get("evaluation", {}), "evaluation")
    _reject_unknown(
        root,
        {
            "model",
            "model_params",
            "parameter_grid",
            "preprocessing",
            "evaluation",
            "run",
        },
        "configuration",
    )
    _reject_unknown(
        preprocessing,
        {"normalization", "ewm_alpha", "window_size"},
        "preprocessing",
    )
    _reject_unknown(
        evaluation,
        {"protocol", "fpr", "label_aware", "require_etapr"},
        "evaluation",
    )
    if preprocessing.get("normalization", "train_minmax") != "train_minmax":
        raise ClassicalConfigError(
            "preprocessing.normalization must be 'train_minmax'"
        )
    if "run" in root:
        _mapping(root["run"], "run")
    if preprocessing.get("window_size", 1) != 1:
        raise ClassicalConfigError("preprocessing.window_size must be 1")
    if evaluation.get("label_aware", True) is not True:
        raise ClassicalConfigError("evaluation.label_aware must be true")
    return ClassicalRunConfig(
        model=root.get("model", ""),
        model_params=root.get("model_params", {}),
        parameter_grid=root.get("parameter_grid", {}),
        ewm_alpha=preprocessing.get("ewm_alpha", 0.8),
        fpr=evaluation.get("fpr", LEGACY_FPR),
        evaluation_protocol=evaluation.get("protocol", EVALUATION_PROTOCOL),
        require_etapr=evaluation.get("require_etapr", True),
    )


def load_classical_config(path: str | Path) -> ClassicalRunConfig:
    """Load a JSON, TOML, or YAML classical baseline configuration."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            import json

            document = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            import tomllib

            with config_path.open("rb") as stream:
                document = tomllib.load(stream)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ClassicalConfigError(
                    "YAML configuration requires the optional yaml dependency"
                ) from exc
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raise ClassicalConfigError(
                "configuration must use .json, .toml, .yaml, or .yml"
            )
    except ClassicalConfigError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ClassicalConfigError(
            f"failed to load classical configuration: {config_path}"
        ) from exc

    if not isinstance(document, Mapping):
        raise ClassicalConfigError("configuration root must be a mapping")
    return config_from_mapping(document)
