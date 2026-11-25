#!/usr/bin/env python3
"""Reproduce the sparse-vector prediction-versus-ground-truth figures."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


LOG = logging.getLogger("stager_pred_gt_plot")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "input_manifest.json"
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "inputs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "generated"


@dataclass(frozen=True)
class AlignmentConfig:
    anomaly_threshold: float = 0.0
    pad_value: float = 0.0
    align_segment: int = 0


@dataclass(frozen=True)
class PlotStyleConfig:
    pred_bar_height: float = 0.95
    gt_bar_height: float = 0.55
    pred_color: str = "#1f77b4"
    gt_color: str = "#d62728"
    false_alarm_color: str = "#FF8A00"
    false_alarm_bar_height: float = 0.95
    false_alarm_linestyle: Any = (0, (14.0, 7.0, 2.0, 7.0))
    false_alarm_legend_linestyle: Any = (0, (4.0, 2.0, 1.0, 2.0))
    false_alarm_linewidth: float = 0.7
    pred_alpha: float = 0.85
    false_alarm_alpha: float = 0.95
    gt_alpha: float = 0.40
    pred_dilate_radius: int = 20
    gt_dilate_radius: int | None = None
    figure_size: tuple[float, float] = (16.0, 2.8)
    dpi: int = 300
    output_format: str = "pdf"
    x_label: str = "Time index"
    axis_fontsize: int = 24
    title_fontsize: int = 12
    tick_fontsize: int = 8
    tight_pad: float = 0.1
    bbox_pad_inches: float = 0.02
    plot_binary: bool = True
    binary_threshold: float = 0.5
    show_xticks: bool = False
    show_legend: bool = True
    stack_show_title: bool = False
    stack_title_fontsize: int = 28
    stack_title_y: float = 0.95
    stack_row_height: float = 1.8
    stack_legend_fontsize: int = 24
    stack_legend_ncol: int = 3
    stack_legend_handletextpad: float = 0.1
    stack_legend_columnspacing: float = 0.8
    stack_legend_handlelength: float = 1.8
    stack_legend_bbox_axes: tuple[float, float] = (0.5, 1.0)
    stack_legend_loc: str = "lower center"
    stack_legend_borderaxespad: float = 0.1
    stack_tight_layout_top: float = 0.90
    stack_model_fontsize: int = 25
    stack_model_labelpad: float = 10.0
    stack_left_margin: float = 0.15
    stack_model_label_mode: str = "ylabel"
    stack_model_inset_x: float = 0.008
    stack_model_inset_y: float = 0.92
    stack_model_inset_bbox: bool = True
    legend_label_gt: str = ": Ground Truth Anomaly"
    legend_label_tp: str = ": True Positive"
    legend_label_fp: str = ": False Positive"
    font_family: str = "Liberation Serif"
    font_sans_family: str = "Liberation Sans"
    font_mono_family: str = "Liberation Mono"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    prediction_dilation_radius: int
    alignment_segment: int


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str


@dataclass(frozen=True)
class PlotManifest:
    datasets: tuple[DatasetSpec, ...]
    models: tuple[ModelSpec, ...]
    subset_models: tuple[str, ...]
    files: Mapping[str, Mapping[str, Mapping[str, str]]]


@dataclass(frozen=True)
class PlotTask:
    model: str
    display_name: str
    dataset: str
    prediction_path: Path
    gt_path: Path


@dataclass(frozen=True)
class AlignedData:
    gt: np.ndarray
    pred: np.ndarray
    left_pad: int
    right_pad: int
    original_gt_len: int
    original_pred_len: int
    first_anomaly_gt: int
    target_length: int
    global_left_anchor: int


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError(f"{field_name} must match [a-z0-9_-]+: {value!r}")
    return value


def load_manifest(path: Path) -> PlotManifest:
    """Load and validate the portable input manifest."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if raw.get("schema_version") != 1:
        raise ValueError("input manifest schema_version must be 1")

    datasets: list[DatasetSpec] = []
    for entry in raw.get("datasets", []):
        dataset_id = _require_identifier(entry.get("id"), "dataset id")
        radius = int(entry.get("prediction_dilation_radius"))
        segment = int(entry.get("alignment_segment"))
        if radius < 0 or segment < 0:
            raise ValueError("dilation radius and alignment segment must be non-negative")
        datasets.append(DatasetSpec(dataset_id, radius, segment))

    models: list[ModelSpec] = []
    for entry in raw.get("models", []):
        model_id = _require_identifier(entry.get("id"), "model id")
        display_name = entry.get("display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"display_name is missing for model {model_id}")
        models.append(ModelSpec(model_id, display_name.strip()))

    if not datasets or not models:
        raise ValueError("input manifest must declare datasets and models")
    if len({item.dataset_id for item in datasets}) != len(datasets):
        raise ValueError("input manifest contains duplicate dataset ids")
    if len({item.model_id for item in models}) != len(models):
        raise ValueError("input manifest contains duplicate model ids")

    subset_models = tuple(
        _require_identifier(value, "subset model id")
        for value in raw.get("subset_models", [])
    )
    model_ids = {item.model_id for item in models}
    unknown_subset = set(subset_models) - model_ids
    if unknown_subset:
        raise ValueError(f"unknown subset model ids: {sorted(unknown_subset)}")

    files = raw.get("files")
    if not isinstance(files, dict):
        raise ValueError("input manifest files must be an object")

    return PlotManifest(tuple(datasets), tuple(models), subset_models, files)


def _resolve_input_path(input_root: Path, relative_value: object) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"invalid input path: {relative_value!r}")
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"input paths must be relative to --input-root: {relative_value}")
    return input_root / relative_path


def build_tasks(manifest: PlotManifest, input_root: Path) -> dict[str, list[PlotTask]]:
    """Resolve every manifest entry into an ordered plotting task."""
    tasks_by_dataset: dict[str, list[PlotTask]] = {}
    for dataset in manifest.datasets:
        dataset_files = manifest.files.get(dataset.dataset_id)
        if not isinstance(dataset_files, dict):
            raise ValueError(f"missing files entry for dataset {dataset.dataset_id}")

        tasks: list[PlotTask] = []
        for model in manifest.models:
            pair = dataset_files.get(model.model_id)
            if not isinstance(pair, dict):
                raise ValueError(
                    f"missing files entry for {dataset.dataset_id}/{model.model_id}"
                )
            prediction_path = _resolve_input_path(input_root, pair.get("prediction"))
            gt_path = _resolve_input_path(input_root, pair.get("ground_truth"))
            model_dir = input_root / dataset.dataset_id / "stager"
            if (
                model.model_id == "stager"
                and prediction_path.parent == gt_path.parent == model_dir
                and not prediction_path.exists()
                and not gt_path.exists()
            ):
                legacy_dir = model_dir.with_name("triad")
                legacy_prediction = legacy_dir / prediction_path.name
                legacy_gt = legacy_dir / gt_path.name
                if legacy_prediction.is_file() and legacy_gt.is_file():
                    prediction_path, gt_path = legacy_prediction, legacy_gt
            tasks.append(
                PlotTask(
                    model=model.model_id,
                    display_name=model.display_name,
                    dataset=dataset.dataset_id,
                    prediction_path=prediction_path,
                    gt_path=gt_path,
                )
            )
        tasks_by_dataset[dataset.dataset_id] = tasks
    return tasks_by_dataset


def find_missing_inputs(tasks_by_dataset: Mapping[str, Sequence[PlotTask]]) -> list[Path]:
    missing: list[Path] = []
    for tasks in tasks_by_dataset.values():
        for task in tasks:
            for path in (task.prediction_path, task.gt_path):
                if not path.is_file():
                    missing.append(path)
    return missing


def load_1d_float_array(path: Path) -> np.ndarray:
    """Load a numeric NumPy sequence without permitting pickled objects."""
    array = np.load(path, allow_pickle=False)
    if array.dtype.kind not in "buif":
        raise TypeError(f"{path} must contain a numeric or boolean array, got {array.dtype}")
    array = np.asarray(array).squeeze()
    if array.ndim == 0:
        raise ValueError(f"{path} is scalar after squeeze; expected a time sequence")
    return array.reshape(-1).astype(float)


def first_anomaly_index(array: np.ndarray, threshold: float) -> int:
    indices = np.flatnonzero(array > threshold)
    return int(indices[0]) if indices.size else -1


def last_anomaly_index(array: np.ndarray, threshold: float) -> int:
    indices = np.flatnonzero(array > threshold)
    return int(indices[-1]) if indices.size else -1


def find_anomaly_segment_starts(array: np.ndarray, threshold: float) -> list[int]:
    binary = (array > threshold).astype(int)
    difference = np.diff(np.r_[0, binary])
    return np.flatnonzero(difference == 1).tolist()


def find_anomaly_segments(array: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    binary = (array > threshold).astype(int)
    difference = np.diff(np.r_[0, binary, 0])
    starts = np.flatnonzero(difference == 1)
    ends = np.flatnonzero(difference == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def compute_dataset_global_alignment(
    tasks: Sequence[PlotTask],
    config: AlignmentConfig,
) -> list[AlignedData]:
    """Align all model sequences for one dataset using ground-truth segments."""
    gt_arrays: list[np.ndarray] = []
    pred_arrays: list[np.ndarray] = []
    anchor_positions: list[int] = []
    last_anomalies: list[int] = []

    for task in tasks:
        gt = load_1d_float_array(task.gt_path)
        pred = load_1d_float_array(task.prediction_path)
        gt_arrays.append(gt)
        pred_arrays.append(pred)

        starts = find_anomaly_segment_starts(gt, config.anomaly_threshold)
        if len(starts) > config.align_segment:
            anchor_positions.append(starts[config.align_segment])
        elif starts:
            anchor_positions.append(starts[0])
        else:
            anchor_positions.append(0)
        last_anomalies.append(last_anomaly_index(gt, config.anomaly_threshold))

    global_left_anchor = max(anchor_positions) if anchor_positions else 0
    left_pads = [global_left_anchor - position for position in anchor_positions]

    tails = [
        len(gt_arrays[index]) - last_index - 1
        if last_index >= 0
        else len(gt_arrays[index])
        for index, last_index in enumerate(last_anomalies)
    ]
    global_tail = max(tails) if tails else 0

    target_length = 0
    for index, gt in enumerate(gt_arrays):
        from_data = left_pads[index] + len(gt)
        if last_anomalies[index] >= 0:
            from_tail = (
                left_pads[index] + last_anomalies[index] + 1 + global_tail
            )
        else:
            from_tail = from_data
        target_length = max(target_length, from_data, from_tail)

    aligned_results: list[AlignedData] = []
    for index, (gt, pred) in enumerate(zip(gt_arrays, pred_arrays)):
        left_pad = left_pads[index]
        right_pad_gt = max(0, target_length - left_pad - len(gt))
        right_pad_pred = max(0, target_length - left_pad - len(pred))
        aligned_gt = np.concatenate(
            [
                np.full(left_pad, config.pad_value),
                gt,
                np.full(right_pad_gt, config.pad_value),
            ]
        )[:target_length]
        aligned_pred = np.concatenate(
            [
                np.full(left_pad, config.pad_value),
                pred,
                np.full(right_pad_pred, config.pad_value),
            ]
        )[:target_length]
        aligned_results.append(
            AlignedData(
                gt=aligned_gt,
                pred=aligned_pred,
                left_pad=left_pad,
                right_pad=right_pad_gt,
                original_gt_len=len(gt),
                original_pred_len=len(pred),
                first_anomaly_gt=anchor_positions[index],
                target_length=target_length,
                global_left_anchor=global_left_anchor,
            )
        )

    if config.align_segment > 0:
        first_segment_bounds = [
            segments[0] if (segments := find_anomaly_segments(
                aligned.gt, config.anomaly_threshold
            )) else None
            for aligned in aligned_results
        ]
        valid_starts = [bounds[0] for bounds in first_segment_bounds if bounds is not None]
        if valid_starts:
            reference_start = min(valid_starts)
            for aligned, bounds in zip(aligned_results, first_segment_bounds):
                if bounds is None or bounds[0] == reference_start:
                    continue
                source_start, source_end = bounds
                segment_length = source_end - source_start + 1
                target_start = reference_start
                target_end = target_start + segment_length
                gt_segment = aligned.gt[source_start : source_end + 1].copy()
                pred_segment = aligned.pred[source_start : source_end + 1].copy()
                aligned.gt[source_start : source_end + 1] = 0.0
                aligned.pred[source_start : source_end + 1] = 0.0
                aligned.gt[target_start:target_end] = gt_segment
                aligned.pred[target_start:target_end] = pred_segment

    return aligned_results


def apply_plot_style(style: PlotStyleConfig) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                style.font_family,
                "Liberation Serif",
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.sans-serif": [
                style.font_sans_family,
                "Liberation Sans",
                "Nimbus Sans",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.monospace": [
                style.font_mono_family,
                "Liberation Mono",
                "Nimbus Mono PS",
                "Courier New",
                "DejaVu Sans Mono",
            ],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": style.title_fontsize,
            "axes.labelsize": style.axis_fontsize,
            "xtick.labelsize": style.tick_fontsize,
            "ytick.labelsize": style.tick_fontsize,
        }
    )


def to_bar_values(array: np.ndarray, style: PlotStyleConfig) -> np.ndarray:
    if style.plot_binary:
        return (array > style.binary_threshold).astype(float)
    return array.astype(float)


def dilate_binary(array: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(array) == 0:
        return array
    kernel = np.ones(2 * radius + 1)
    return (np.convolve(array, kernel, mode="same") > 0).astype(float)


def prepare_plot_bars_split(
    aligned: AlignedData,
    style: PlotStyleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_binary = to_bar_values(aligned.gt, style)
    gt_radius = (
        style.pred_dilate_radius
        if style.gt_dilate_radius is None
        else style.gt_dilate_radius
    )
    if gt_radius > 0:
        gt_binary = dilate_binary(gt_binary, gt_radius)
    gt_bar = gt_binary * style.gt_bar_height
    gt_anomaly = gt_binary > 0.5

    pred_binary = to_bar_values(aligned.pred, style)
    if style.pred_dilate_radius > 0:
        pred_binary = dilate_binary(pred_binary, style.pred_dilate_radius)
    pred_on = pred_binary > 0.5
    true_positive = (
        (pred_on & gt_anomaly).astype(float) * style.pred_bar_height
    )
    false_positive = (
        (pred_on & ~gt_anomaly).astype(float) * style.false_alarm_bar_height
    )
    return gt_bar, true_positive, false_positive


def _legend_handles(style: PlotStyleConfig) -> list[Any]:
    return [
        Patch(
            facecolor=style.gt_color,
            alpha=style.gt_alpha,
            edgecolor="none",
            label=style.legend_label_gt,
        ),
        Patch(
            facecolor=style.pred_color,
            alpha=style.pred_alpha,
            edgecolor="none",
            label=style.legend_label_tp,
        ),
        Line2D(
            [0],
            [0],
            color=style.false_alarm_color,
            alpha=style.false_alarm_alpha,
            linestyle=style.false_alarm_legend_linestyle,
            linewidth=max(2.0, style.false_alarm_linewidth),
            solid_capstyle="butt",
            dash_capstyle="butt",
            label=style.legend_label_fp,
        ),
    ]


def _draw_false_positive(
    axis: Any,
    false_positive_bar: np.ndarray,
    style: PlotStyleConfig,
) -> None:
    indices = np.flatnonzero(false_positive_bar > 0)
    if indices.size == 0:
        return
    segments = [
        [(float(index) + 0.5, 0.0), (float(index) + 0.5, float(height))]
        for index, height in zip(indices, false_positive_bar[indices])
    ]
    collection = LineCollection(
        segments,
        colors=style.false_alarm_color,
        linestyles=style.false_alarm_linestyle,
        linewidths=style.false_alarm_linewidth,
        alpha=style.false_alarm_alpha,
        zorder=3,
    )
    collection.set_capstyle("butt")
    axis.add_collection(collection)


def positive_bar_runs(bar: np.ndarray) -> list[tuple[float, float, float]]:
    """Return contiguous positive intervals as start, width, and height."""
    if bar.size == 0:
        return []
    active = bar > 0
    if not np.any(active):
        return []
    difference = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(difference == 1)
    ends = np.flatnonzero(difference == -1)
    return [
        (float(start), float(end - start), float(bar[start]))
        for start, end in zip(starts.tolist(), ends.tolist())
    ]


def _draw_bar_runs(
    axis: Any,
    bar: np.ndarray,
    *,
    facecolor: str,
    alpha: float,
    zorder: int,
) -> None:
    runs = positive_bar_runs(bar)
    if not runs:
        return
    by_height: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for start, width, height in runs:
        by_height[height].append((start, width))
    for height, ranges in by_height.items():
        axis.broken_barh(
            ranges,
            (0.0, height),
            facecolors=facecolor,
            alpha=alpha,
            edgecolors="none",
            linewidths=0,
            zorder=zorder,
        )


def draw_sparse_bars(
    axis: Any,
    gt_bar: np.ndarray,
    true_positive_bar: np.ndarray,
    false_positive_bar: np.ndarray,
    style: PlotStyleConfig,
) -> None:
    _draw_bar_runs(
        axis,
        gt_bar,
        facecolor=style.gt_color,
        alpha=style.gt_alpha,
        zorder=1,
    )
    _draw_bar_runs(
        axis,
        true_positive_bar,
        facecolor=style.pred_color,
        alpha=style.pred_alpha,
        zorder=2,
    )
    _draw_false_positive(axis, false_positive_bar, style)


def render_stacked_figure(
    dataset: str,
    rows: Sequence[tuple[PlotTask, AlignedData]],
    output_path: Path,
    style: PlotStyleConfig,
) -> Path:
    """Render one sparse-vector stacked PDF."""
    if not rows:
        raise ValueError(f"no rows supplied for dataset {dataset}")

    apply_plot_style(style)
    target_length = rows[0][1].target_length
    row_count = len(rows)
    figure_height = max(2.0, style.stack_row_height * row_count)
    figure, axes = plt.subplots(
        row_count,
        1,
        figsize=(style.figure_size[0], figure_height),
        dpi=style.dpi,
        sharex=True,
    )
    if row_count == 1:
        axes = [axes]

    model_counts: dict[str, int] = defaultdict(int)
    model_totals: dict[str, int] = defaultdict(int)
    for task, _ in rows:
        model_totals[task.model] += 1

    for index, (task, aligned) in enumerate(rows):
        axis = axes[index]
        model_counts[task.model] += 1
        label = task.display_name
        if model_totals[task.model] > 1:
            label = f"{label} #{model_counts[task.model]}"

        gt_bar, true_positive_bar, false_positive_bar = prepare_plot_bars_split(
            aligned, style
        )
        draw_sparse_bars(
            axis, gt_bar, true_positive_bar, false_positive_bar, style
        )

        if style.stack_model_label_mode == "inset":
            text_options: dict[str, Any] = {
                "transform": axis.transAxes,
                "fontsize": style.stack_model_fontsize,
                "ha": "left",
                "va": "top",
                "zorder": 10,
            }
            if style.stack_model_inset_bbox:
                text_options["bbox"] = {
                    "boxstyle": "square,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.55",
                    "linewidth": 0.6,
                    "alpha": 0.92,
                }
            axis.text(
                style.stack_model_inset_x,
                style.stack_model_inset_y,
                label,
                **text_options,
            )
            axis.set_ylabel("")
        else:
            axis.set_ylabel(
                label,
                rotation=90,
                ha="center",
                va="center",
                fontsize=style.stack_model_fontsize,
                labelpad=style.stack_model_labelpad,
            )

        axis.set_xlim(0, target_length)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([])
        if index == row_count - 1:
            axis.set_xlabel(style.x_label, fontsize=style.axis_fontsize)
            if not style.show_xticks:
                axis.set_xticks([])
        else:
            axis.set_xticks([])

    layout_top = style.stack_tight_layout_top if style.stack_show_title else 1.0
    figure.tight_layout(
        pad=style.tight_pad,
        rect=[style.stack_left_margin, 0, 1, layout_top],
    )

    if style.stack_show_title:
        title_x = (style.stack_left_margin + 1.0) / 2.0
        figure.suptitle(
            dataset.upper(),
            fontsize=style.stack_title_fontsize,
            x=title_x,
            y=style.stack_title_y,
            ha="center",
            fontweight="bold",
        )

    if style.show_legend:
        handles = _legend_handles(style)
        axes[0].legend(
            handles,
            [handle.get_label() for handle in handles],
            loc=style.stack_legend_loc,
            bbox_to_anchor=style.stack_legend_bbox_axes,
            bbox_transform=axes[0].transAxes,
            ncol=style.stack_legend_ncol,
            fontsize=style.stack_legend_fontsize,
            handlelength=style.stack_legend_handlelength,
            handletextpad=style.stack_legend_handletextpad,
            columnspacing=style.stack_legend_columnspacing,
            borderaxespad=style.stack_legend_borderaxespad,
            frameon=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format=style.output_format,
        dpi=style.dpi,
        bbox_inches="tight",
        pad_inches=style.bbox_pad_inches,
    )
    plt.close(figure)
    return output_path


def align_all_inputs(
    manifest: PlotManifest,
    tasks_by_dataset: Mapping[str, Sequence[PlotTask]],
) -> dict[str, list[AlignedData]]:
    """Load and align all declared inputs once."""
    aligned_by_dataset: dict[str, list[AlignedData]] = {}
    for dataset in manifest.datasets:
        config = AlignmentConfig(align_segment=dataset.alignment_segment)
        aligned = compute_dataset_global_alignment(
            tasks_by_dataset[dataset.dataset_id], config
        )
        aligned_by_dataset[dataset.dataset_id] = aligned
        LOG.info(
            "%s: %d models, target length %d, global anchor %d",
            dataset.dataset_id,
            len(aligned),
            aligned[0].target_length,
            aligned[0].global_left_anchor,
        )
    return aligned_by_dataset


def generate_figures(
    manifest: PlotManifest,
    tasks_by_dataset: Mapping[str, Sequence[PlotTask]],
    aligned_by_dataset: Mapping[str, Sequence[AlignedData]],
    output_root: Path,
) -> list[Path]:
    """Generate the historical set of eight sparse-vector PDFs."""
    outputs: list[Path] = []
    base_style = PlotStyleConfig()

    for dataset in manifest.datasets:
        dataset_id = dataset.dataset_id
        tasks = list(tasks_by_dataset[dataset_id])
        aligned = list(aligned_by_dataset[dataset_id])
        aligned_by_model = {
            task.model: aligned_item
            for task, aligned_item in zip(tasks, aligned)
        }
        task_by_model = {task.model: task for task in tasks}
        full_rows = list(zip(tasks, aligned))
        subset_rows = [
            (task_by_model[model], aligned_by_model[model])
            for model in manifest.subset_models
        ]

        dataset_style = replace(
            base_style,
            pred_dilate_radius=dataset.prediction_dilation_radius,
        )
        compact_style = replace(
            dataset_style,
            stack_row_height=round(dataset_style.stack_row_height * 0.7, 4),
            stack_model_label_mode="inset",
            stack_left_margin=0.02,
        )
        dataset_output = output_root / dataset_id

        outputs.append(
            render_stacked_figure(
                dataset_id,
                full_rows,
                dataset_output / f"{dataset_id}_stacked.pdf",
                compact_style,
            )
        )
        outputs.append(
            render_stacked_figure(
                dataset_id,
                full_rows,
                dataset_output / f"{dataset_id}_stacked_nolegend.pdf",
                replace(compact_style, show_legend=False),
            )
        )
        outputs.append(
            render_stacked_figure(
                dataset_id,
                subset_rows,
                dataset_output / f"{dataset_id}_stacked_sub_compact.pdf",
                compact_style,
            )
        )
        outputs.append(
            render_stacked_figure(
                dataset_id,
                subset_rows,
                dataset_output / f"{dataset_id}_stacked_sub.pdf",
                dataset_style,
            )
        )

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce sparse-vector prediction-versus-ground-truth PDFs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="input manifest (default: input_manifest.json next to this script)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="input array directory (default: inputs/ next to this script)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="generated PDF directory (default: generated/ next to this script)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and align all inputs without writing PDFs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    tasks_by_dataset = build_tasks(manifest, args.input_root)
    missing = find_missing_inputs(tasks_by_dataset)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing {len(missing)} required input files:\n{formatted}")

    aligned_by_dataset = align_all_inputs(manifest, tasks_by_dataset)
    if args.check_only:
        LOG.info("All inputs are valid; no figures were written.")
        return 0

    outputs = generate_figures(
        manifest, tasks_by_dataset, aligned_by_dataset, args.output_root
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
