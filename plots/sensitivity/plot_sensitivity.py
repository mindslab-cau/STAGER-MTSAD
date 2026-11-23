from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "generated"
SAVE_FORMATS = ("pdf",)
METRICS = ("F1", "PA-F1", "eTaF1")
COLORMAP = "Blues"

DATASET_FILES = {
    "swat": INPUT_DIR / "swat.csv",
    "wadi": INPUT_DIR / "wadi.csv",
}
DATASET_DISPLAY_NAMES = {
    "swat": "SWaT",
    "wadi": "WADI",
}
MSHTRANS_COLOR_MINIMA = {
    "swat": {"F1": 0.8273, "PA-F1": 0.9095, "eTaF1": 0.5500},
    "wadi": {"F1": 0.5311, "PA-F1": 0.8286, "eTaF1": 0.6676},
}


@dataclass(frozen=True)
class FontConfig:
    title: float = 12
    title_pad: float = 1.0
    axis_label: float = 12
    axis_label_pad: float = 4.0
    tick: float = 8
    tick_pad: float = 4.0
    annotation: float = 10
    colorbar: float = 9


SINGLE_FONT = FontConfig()
COMBINED_FONT = FontConfig(
    title=11,
    title_pad=6.0,
    axis_label=10,
    tick=8,
    annotation=9,
    colorbar=10,
)
NO_COLORBAR_FONT = FontConfig(
    title=11,
    title_pad=2.0,
    axis_label=10,
    axis_label_pad=2.0,
    tick=8,
    tick_pad=3.0,
    annotation=9,
    colorbar=10,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the MSHTrans sensitivity figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: plots/sensitivity/generated).",
    )
    return parser.parse_args()


def configure_font() -> str:
    for font_path in font_manager.findSystemFonts(fontext="ttf"):
        try:
            font_manager.fontManager.addfont(font_path)
        except (OSError, RuntimeError):
            continue
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        mpl.rcParams["font.family"] = family
        return family
    mpl.rcParams["font.family"] = "serif"
    return "serif"


def load_metric_grid(
    csv_path: Path,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    required = {"classification", "euclid distance", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    data = frame.loc[
        frame["classification"].astype(str).str.lower() != "min"
    ].copy()
    for column in ("classification", "euclid distance", metric):
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data.duplicated(["classification", "euclid distance"]).any():
        raise ValueError(f"{csv_path} contains duplicate sensitivity coordinates")

    pivot = (
        data.pivot(
            index="classification",
            columns="euclid distance",
            values=metric,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    expected = np.arange(0.0, 0.5, 0.1)
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    if not np.allclose(x_values, expected) or not np.allclose(y_values, expected):
        raise ValueError(
            f"{csv_path} must contain a complete 5x5 grid from 0.0 to 0.4"
        )
    if pivot.isna().any().any():
        raise ValueError(f"{csv_path} contains an incomplete sensitivity grid")
    return x_values, y_values, pivot.to_numpy(dtype=float)


def load_all_data() -> dict[
    str,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    return {
        dataset: {
            metric: load_metric_grid(csv_path, metric) for metric in METRICS
        }
        for dataset, csv_path in DATASET_FILES.items()
    }


def compute_edges(values: np.ndarray) -> np.ndarray:
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2
    edges[0] = values[0] - (values[1] - values[0]) / 2
    edges[-1] = values[-1] + (values[-1] - values[-2]) / 2
    return edges


def format_tick(value: float) -> str:
    return f"{value:g}"


def format_metric_name(metric: str) -> str:
    return metric.lower().replace("-", "_")


def text_color(value: float, vmin: float, vmax: float) -> str:
    rgba = plt.get_cmap(COLORMAP)(Normalize(vmin=vmin, vmax=vmax)(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if luminance > 0.55 else "white"


def draw_heatmap(
    axis,
    x_values: np.ndarray,
    y_values: np.ndarray,
    scores: np.ndarray,
    *,
    dataset: str,
    metric: str,
    font: FontConfig,
    title: str | None,
):
    vmin = MSHTRANS_COLOR_MINIMA[dataset][metric]
    vmax = float(np.nanmax(scores))
    x_edges = compute_edges(x_values)
    y_edges = compute_edges(y_values)
    mesh = axis.pcolormesh(
        x_edges,
        y_edges,
        scores,
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        edgecolors="white",
        linewidth=1.0,
    )
    if title is not None:
        axis.set_title(title, fontsize=font.title, pad=font.title_pad)
    axis.set_xlabel(
        r"Reg. ($\alpha$)",
        fontsize=font.axis_label,
        labelpad=font.axis_label_pad,
    )
    axis.set_ylabel(
        r"Cls. ($\beta$)",
        fontsize=font.axis_label,
        labelpad=font.axis_label_pad,
    )
    axis.set_xticks(x_values)
    axis.set_yticks(y_values)
    axis.set_xticklabels([format_tick(value) for value in x_values])
    axis.set_yticklabels([format_tick(value) for value in y_values])
    axis.tick_params(labelsize=font.tick, pad=font.tick_pad)
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])
    axis.set_aspect("equal")

    for row_index, row_value in enumerate(y_values):
        for column_index, column_value in enumerate(x_values):
            score = scores[row_index, column_index]
            axis.text(
                column_value,
                row_value,
                f"{score * 100:.2f}",
                ha="center",
                va="center",
                fontsize=font.annotation,
                color=text_color(score, vmin, vmax),
            )
    return mesh


def save_figure(
    figure,
    stem: str,
    output_dir: Path,
    *,
    pad_inches: float = 0.0,
) -> None:
    for extension in SAVE_FORMATS:
        destination = output_dir / extension / "2d" / f"{stem}.{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight", pad_inches=pad_inches)
        print(f"Saved {destination}")


def add_shared_colorbar(
    figure,
    axes,
    font: FontConfig,
    *,
    pad: float,
    width: float,
    height_ratio: float,
) -> None:
    figure.canvas.draw()
    positions = [axis.get_position() for axis in np.atleast_1d(axes)]
    bottom = min(position.y0 for position in positions)
    top = max(position.y1 for position in positions)
    right = max(position.x1 for position in positions)
    total_height = top - bottom
    colorbar_height = total_height * height_ratio
    colorbar_bottom = bottom + (total_height - colorbar_height) / 2
    colorbar_axis = figure.add_axes(
        [right + pad, colorbar_bottom, width, colorbar_height]
    )
    scalar_mappable = mpl.cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap=COLORMAP,
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=font.colorbar, length=0)


def plot_single_metrics(all_data, output_dir: Path) -> None:
    for dataset, metric_data in all_data.items():
        for metric in METRICS:
            x_values, y_values, scores = metric_data[metric]
            figure, axis = plt.subplots(figsize=(3.2, 2.8))
            figure.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.90)
            draw_heatmap(
                axis,
                x_values,
                y_values,
                scores,
                dataset=dataset,
                metric=metric,
                font=SINGLE_FONT,
                title=DATASET_DISPLAY_NAMES[dataset],
            )
            save_figure(
                figure,
                f"{dataset}_{format_metric_name(metric)}_2d",
                output_dir,
            )
            plt.close(figure)


def plot_wadi_colorbars(all_data, output_dir: Path) -> None:
    dataset = "wadi"
    for metric in METRICS:
        x_values, y_values, scores = all_data[dataset][metric]
        figure, axis = plt.subplots(figsize=(3.8, 2.8))
        figure.subplots_adjust(left=0.08, right=0.94, bottom=0.12, top=0.90)
        mesh = draw_heatmap(
            axis,
            x_values,
            y_values,
            scores,
            dataset=dataset,
            metric=metric,
            font=SINGLE_FONT,
            title=DATASET_DISPLAY_NAMES[dataset],
        )
        colorbar = figure.colorbar(mesh, ax=axis, shrink=0.92, pad=0.02)
        colorbar.set_ticks(
            [MSHTRANS_COLOR_MINIMA[dataset][metric], float(np.nanmax(scores))]
        )
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.ax.tick_params(labelsize=SINGLE_FONT.colorbar, length=0)
        save_figure(
            figure,
            f"{dataset}_{format_metric_name(metric)}_colorbar_2d",
            output_dir,
        )
        plt.close(figure)


def plot_combined(all_data, output_dir: Path) -> None:
    datasets = list(all_data)
    figure, axes = plt.subplots(len(datasets), len(METRICS), figsize=(9.6, 6.0))
    axes = np.atleast_2d(axes)
    figure.subplots_adjust(
        left=0.06,
        right=0.92,
        bottom=0.08,
        top=0.92,
        wspace=0.22,
        hspace=0.35,
    )
    for row, dataset in enumerate(datasets):
        for column, metric in enumerate(METRICS):
            x_values, y_values, scores = all_data[dataset][metric]
            draw_heatmap(
                axes[row, column],
                x_values,
                y_values,
                scores,
                dataset=dataset,
                metric=metric,
                font=COMBINED_FONT,
                title=f"{DATASET_DISPLAY_NAMES[dataset]} | {metric}",
            )
    add_shared_colorbar(
        figure,
        axes.ravel(),
        COMBINED_FONT,
        pad=0.02,
        width=0.018,
        height_ratio=1.0,
    )
    save_figure(figure, "combined_aux_2d", output_dir)
    plt.close(figure)


def plot_combined_no_colorbar(all_data, output_dir: Path) -> None:
    datasets = list(all_data)
    figure, axes = plt.subplots(len(datasets), len(METRICS), figsize=(9.6, 6.0))
    axes = np.atleast_2d(axes)
    figure.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.96,
        wspace=-0.04,
        hspace=0.22,
    )
    for row, dataset in enumerate(datasets):
        for column, metric in enumerate(METRICS):
            x_values, y_values, scores = all_data[dataset][metric]
            draw_heatmap(
                axes[row, column],
                x_values,
                y_values,
                scores,
                dataset=dataset,
                metric=metric,
                font=NO_COLORBAR_FONT,
                title=f"{DATASET_DISPLAY_NAMES[dataset]} | {metric}",
            )
    save_figure(
        figure,
        "combined_aux_2d_no_colorbar",
        output_dir,
        pad_inches=0.02,
    )
    plt.close(figure)


def plot_combined_wide(all_data, output_dir: Path) -> None:
    datasets = list(all_data)
    figure, axes = plt.subplots(
        1,
        len(datasets) * len(METRICS),
        figsize=(16.0, 3.2),
    )
    axes = np.atleast_1d(axes)
    figure.subplots_adjust(
        left=0.04,
        right=0.94,
        bottom=0.10,
        top=0.90,
        wspace=0.28,
    )
    axis_index = 0
    for dataset in datasets:
        for metric in METRICS:
            x_values, y_values, scores = all_data[dataset][metric]
            draw_heatmap(
                axes[axis_index],
                x_values,
                y_values,
                scores,
                dataset=dataset,
                metric=metric,
                font=COMBINED_FONT,
                title=f"{DATASET_DISPLAY_NAMES[dataset]} | {metric}",
            )
            axis_index += 1
    add_shared_colorbar(
        figure,
        axes,
        COMBINED_FONT,
        pad=0.02,
        width=0.012,
        height_ratio=1.0,
    )
    save_figure(figure, "combined_aux_wide_2d", output_dir)
    plt.close(figure)


def plot_combined_transposed(all_data, output_dir: Path) -> None:
    datasets = list(all_data)
    figure = plt.figure(figsize=(4.8, 7.5))
    grid = figure.add_gridspec(
        len(METRICS) + 1,
        len(datasets),
        height_ratios=[1.0] * len(METRICS) + [0.045],
        left=0.10,
        right=0.98,
        bottom=0.045,
        top=0.98,
        wspace=0.30,
        hspace=0.50,
    )
    for row, metric in enumerate(METRICS):
        for column, dataset in enumerate(datasets):
            axis = figure.add_subplot(grid[row, column])
            x_values, y_values, scores = all_data[dataset][metric]
            draw_heatmap(
                axis,
                x_values,
                y_values,
                scores,
                dataset=dataset,
                metric=metric,
                font=COMBINED_FONT,
                title=f"{DATASET_DISPLAY_NAMES[dataset]} | {metric}",
            )

    colorbar_axis = figure.add_subplot(grid[-1, :])
    scalar_mappable = mpl.cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap=COLORMAP,
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=COMBINED_FONT.colorbar, length=0, pad=2)
    save_figure(figure, "combined_aux_2d_transposed", output_dir)
    plt.close(figure)


def plot_standalone_colorbar(output_dir: Path) -> None:
    bar_height = 6.0 * (0.92 - 0.08) / (2.0 + 0.35)
    figure_width = 0.85
    bar_width = 0.16
    left_margin = 0.12
    vertical_margin = 0.12
    figure_height = bar_height + 2 * vertical_margin
    figure = plt.figure(figsize=(figure_width, figure_height))
    colorbar_axis = figure.add_axes(
        [
            left_margin / figure_width,
            vertical_margin / figure_height,
            bar_width / figure_width,
            bar_height / figure_height,
        ]
    )
    scalar_mappable = mpl.cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap=COLORMAP,
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=10, length=0, pad=3)
    save_figure(figure, "colorbar_2d", output_dir)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    font_family = configure_font()
    print(f"Using font family: {font_family}")
    all_data = load_all_data()
    plot_single_metrics(all_data, output_dir)
    plot_combined_wide(all_data, output_dir)
    plot_combined_no_colorbar(all_data, output_dir)


if __name__ == "__main__":
    main()
