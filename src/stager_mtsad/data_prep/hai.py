"""Reproduce the HAI 22.04 block aggregation used for the paper."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from .prepare import (
    PreparationError,
    PreparationResult,
    _json_write,
    _normalize_source_path,
    _output_paths,
    _output_summary,
    _pd,
    _publish_files,
    _source_summary,
)

_TIMESTAMP = "timestamp"
_LABEL = "Attack"
_FEATURE_COUNT = 86
_BLOCK_SIZE = 10


def _downsample(paths: Sequence[Path], feature_names: list[str]):
    pd = _pd()
    parts = []
    for path in paths:
        part = pd.read_csv(path)
        if list(part.columns) != [_TIMESTAMP, *feature_names, _LABEL]:
            raise PreparationError(f"{path.name} does not have the HAI 22.04 column order")
        if not part[_LABEL].isin([0, 1]).all():
            raise PreparationError(f"{path.name} must contain binary Attack labels")
        parts.append(part)

    rows = pd.concat(parts, ignore_index=True)
    usable = len(rows) - len(rows) % _BLOCK_SIZE
    if not usable:
        raise PreparationError("HAI split contains no complete block of ten rows")

    grouped = rows.iloc[:usable].groupby(rows.index[:usable] // _BLOCK_SIZE)
    prepared = grouped[feature_names].median()
    prepared.insert(0, _TIMESTAMP, grouped[_TIMESTAMP].first())
    prepared[_LABEL] = (grouped[_LABEL].sum() > _BLOCK_SIZE // 2).astype(int)
    return prepared


def prepare_hai(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> PreparationResult:
    """Read the ten HAI 22.04 recordings and write prepared train/test CSVs."""

    raw_dir = Path(raw_dir).expanduser()
    sources = {
        split: tuple(
            _normalize_source_path(raw_dir / f"{split}{index}.csv", f"{split}{index}")
            for index in range(1, count + 1)
        )
        for split, count in (("train", 6), ("test", 4))
    }
    first = _pd().read_csv(sources["train"][0], nrows=0)
    feature_names = list(first.columns[1:-1])
    if (
        len(feature_names) != _FEATURE_COUNT
        or first.columns[0] != _TIMESTAMP
        or first.columns[-1] != _LABEL
    ):
        raise PreparationError("expected timestamp, 86 HAI features, and Attack")

    destination, targets = _output_paths(
        output_dir, (*sources["train"], *sources["test"]), force=force
    )
    with tempfile.TemporaryDirectory(prefix=".stager-prepare-", dir=destination) as directory:
        temporary = {key: Path(directory) / path.name for key, path in targets.items()}
        summaries = {}
        for split, paths in sources.items():
            prepared = _downsample(paths, feature_names)
            prepared.to_csv(temporary[split], index=False)
            summaries[split] = _output_summary(temporary[split], prepared, _LABEL)

        _json_write(temporary["dataset_manifest"], {"datasets": {"hai": {
            "train_path": targets["train"].name,
            "test_path": targets["test"].name,
            "timestamp": _TIMESTAMP,
            "label": _LABEL,
            "schema": {
                "format": "prepared_csv",
                "timestamp_fields": [_TIMESTAMP],
                "label_field": _LABEL,
                "normal_value": 0,
                "anomaly_value": 1,
                "expected_period_seconds": _BLOCK_SIZE,
                "expected_feature_count": _FEATURE_COUNT,
                "feature_order": feature_names,
                "drop_fields": [_LABEL],
            },
        }}})
        _json_write(temporary["preparation"], {
            "dataset": "hai",
            "protocol": "hai_22.04_numbered_recordings_then_10_row_blocks",
            "block_rows": _BLOCK_SIZE,
            "sources": {
                split: [_source_summary(path) for path in paths]
                for split, paths in sources.items()
            },
            "outputs": summaries,
        })
        _publish_files(temporary, targets)

    return PreparationResult(
        dataset="hai",
        output_dir=destination,
        train_path=targets["train"],
        test_path=targets["test"],
        dataset_manifest_path=targets["dataset_manifest"],
        preparation_path=targets["preparation"],
    )
