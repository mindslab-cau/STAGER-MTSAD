#!/usr/bin/env python3
"""Generate inference-time versus detection-quality trade-off figures."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


_MPL_CACHE = tempfile.TemporaryDirectory(prefix="triad-matplotlib-")
os.environ["MPLCONFIGDIR"] = _MPL_CACHE.name

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
DEFAULT_TIME_CSV = INPUT_DIR / "inference_time.csv"
DEFAULT_METRIC_CSV = INPUT_DIR / "RQ1_maintable.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "generated"

DATASETS = ("SWaT", "WADI")
METRICS = ("F1", "PA-F1", "eTaF1")

TITLE_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 10
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 10

X_AXIS_LABEL = "Inference Time (ms)"
Y_AXIS_MIN = 0.0
Y_AXIS_MAX = 1.0
X_AXIS_PADDING_RATIO = 0.08

SINGLE_TITLE_TEMPLATE = ""
ROW_TITLE_TEMPLATE = "{metric}"
GRID_TITLE_TEMPLATE = ""

DATASET_LABEL_X = 0.5
DATASET_LABEL_Y = 0.965
DATASET_LABEL_FONT_SIZE = 10
DATASET_LABEL_BOX_STYLE = "square,pad=0.25"
DATASET_LABEL_FACE_COLOR = "white"
DATASET_LABEL_EDGE_COLOR = "black"
DATASET_LABEL_EDGE_WIDTH = 0.7

GRID_FIGSIZE = (8.0, 8.0 * 7.0 / 11.0)
ROW_FIGSIZE = (8.0, 8.0 * 3.5 / 11.8)
SINGLE_FIGSIZE = (GRID_FIGSIZE[0] / 3, GRID_FIGSIZE[1] / 2)

SINGLE_LEFT = 0.17
SINGLE_RIGHT = 0.98
SINGLE_BOTTOM = 0.18
SINGLE_TOP = 0.97

ROW_LEFT = 0.07
ROW_RIGHT = 0.995
ROW_BOTTOM = 0.16
ROW_TOP = 0.92
ROW_WSPACE = 0.14

GRID_LEFT = 0.07
GRID_RIGHT = 0.995
GRID_BOTTOM = 0.10
GRID_TOP = 0.95
GRID_WSPACE = 0.14
GRID_HSPACE = 0.18
DATASET_ROWS_HSPACE = 0.096
COMPACT_AXIS_LABEL_PAD = 2.0
COMPACT_TICK_PAD = 3.0
PANEL_LEGEND_COLUMNS = 2

SAVE_BBOX = "tight"
SINGLE_PAD_INCHES = 0.02
COMBINED_PAD_INCHES = 0.0

OUR_METHOD = "TriAD"

DEFAULT_MARKER_SIZE = 90
MODEL_MARKER_SIZE_MULTIPLIERS = {
    "USAD": 1.0,
    "TSMixer": 1.0,
    "AnomalyTrans": 1.0,
    "FuSAGNet": 0.8,
    "MtsCID": 1.0,
    "MSHTrans": 1.0,
    OUR_METHOD: 2.0,
}
MARKER_EDGE_COLOR = "black"
DEFAULT_MARKER_EDGE_WIDTH = 0.7
MODEL_MARKER_EDGE_WIDTHS = {
    "USAD": DEFAULT_MARKER_EDGE_WIDTH,
    "TSMixer": DEFAULT_MARKER_EDGE_WIDTH,
    "AnomalyTrans": DEFAULT_MARKER_EDGE_WIDTH,
    "FuSAGNet": DEFAULT_MARKER_EDGE_WIDTH,
    "MtsCID": DEFAULT_MARKER_EDGE_WIDTH,
    "MSHTrans": DEFAULT_MARKER_EDGE_WIDTH,
    OUR_METHOD: 1.0,
}
GRID_ALPHA = 0.22
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.7

LEGEND_LOCATION = "lower right"
GRID_LEGEND_LOCATION = "lower center"
GRID_LEGEND_ROW_INDEX = 0
GRID_LEGEND_COL_INDEX = 1
LEGEND_COLUMNS = 2
LEGEND_COLUMN_SPACING = 1.0
LEGEND_HANDLE_TEXT_PAD = 0.1
SHOW_SINGLE_LEGEND = False

MODEL_NAME_MAP = {
    "USAD": "USAD",
    "TSMIXER": "TSMixer",
    "TSMixer": "TSMixer",
    "AnomalyTransformer": "AnomalyTrans",
    "AnomalyTrans": "AnomalyTrans",
    "FuSAGNet": "FuSAGNet",
    "MtsCID": "MtsCID",
    "MSHTrans": "MSHTrans",
    "MSHTRANS": "MSHTrans",
    OUR_METHOD: OUR_METHOD,
}

MODEL_DISPLAY_NAMES = {
    "USAD": "USAD",
    "TSMixer": "TSMixer",
    "AnomalyTrans": "AT",
    "FuSAGNet": "FuSAGNet",
    "MtsCID": "MtsCID",
    "MSHTrans": "MSHTrans",
    OUR_METHOD: OUR_METHOD,
}

MODEL_MARKERS = {
    "USAD": "o",
    "TSMixer": "s",
    "AnomalyTrans": "^",
    "FuSAGNet": "D",
    "MtsCID": "P",
    "MSHTrans": "X",
    OUR_METHOD: "*",
}

MODEL_ORDER = (
    "USAD",
    "TSMixer",
    "AnomalyTrans",
    "FuSAGNet",
    "MtsCID",
    "MSHTrans",
    OUR_METHOD,
)

MODEL_COLORS = {
    "USAD": "#90CAF9",
    "TSMixer": "#F5BCDB",
    "AnomalyTrans": "#F27457",
    "FuSAGNet": "#48B5A3",
    "MtsCID": "#6863F2",
    "MSHTrans": "#585934",
    OUR_METHOD: "#F2C572",
}


def select_serif_font() -> str:
    """Select the preferred serif font with portable fallbacks."""
    for font_path in font_manager.findSystemFonts(fontext="ttf"):
        try:
            font_manager.fontManager.addfont(font_path)
        except (OSError, RuntimeError):
            continue
    for candidate in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        return candidate
    return "DejaVu Serif"


def configure_plot_style() -> None:
    mpl.rcParams["font.family"] = select_serif_font()
    mpl.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
    mpl.rcParams["axes.labelsize"] = AXIS_LABEL_FONT_SIZE
    mpl.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
    mpl.rcParams["ytick.labelsize"] = TICK_FONT_SIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def normalize_model_name(model_name: str) -> str:
    normalized = MODEL_NAME_MAP.get(str(model_name).strip())
    if normalized is None:
        raise KeyError(f"Unknown model name: {model_name!r}")
    return normalized


def load_inference_time_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=1)
    frame = frame.rename(columns={frame.columns[0]: "model_name"})
    missing_datasets = [name for name in DATASETS if name not in frame.columns]
    if missing_datasets:
        raise ValueError(f"Missing inference-time columns: {missing_datasets}")
    frame["canonical_model"] = frame["model_name"].map(normalize_model_name)
    long_frame = frame.melt(
        id_vars=["model_name", "canonical_model"],
        value_vars=list(DATASETS),
        var_name="dataset",
        value_name="time_ms",
    )
    long_frame["time_ms"] = pd.to_numeric(long_frame["time_ms"], errors="raise")
    return long_frame


def load_metric_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={frame.columns[0]: "dataset", "Model": "metric"})
    frame["dataset"] = frame["dataset"].ffill()
    long_frame = frame.melt(
        id_vars=["dataset", "metric"],
        var_name="model_name",
        value_name="score",
    )
    long_frame = long_frame[long_frame["metric"].isin(METRICS)].copy()
    long_frame["canonical_model"] = long_frame["model_name"].map(MODEL_NAME_MAP)
    long_frame = long_frame[long_frame["canonical_model"].notna()].copy()
    long_frame["score"] = pd.to_numeric(long_frame["score"], errors="raise")
    return long_frame


def build_plot_frame(time_csv: Path, metric_csv: Path) -> pd.DataFrame:
    inference_frame = load_inference_time_frame(time_csv)
    metric_frame = load_metric_frame(metric_csv)
    merged = metric_frame.merge(
        inference_frame[["dataset", "canonical_model", "time_ms"]],
        on=["dataset", "canonical_model"],
        how="inner",
        validate="many_to_one",
    )
    merged["display_name"] = merged["canonical_model"].map(MODEL_DISPLAY_NAMES)
    merged["model_order"] = merged["canonical_model"].map(
        {model_name: index for index, model_name in enumerate(MODEL_ORDER)}
    )
    expected_rows = len(DATASETS) * len(METRICS) * len(MODEL_ORDER)
    if len(merged) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} plotted records, received {len(merged)}"
        )
    return merged.sort_values(
        ["dataset", "metric", "model_order"]
    ).reset_index(drop=True)


def resolve_x_limits(plot_frame: pd.DataFrame) -> tuple[float, float]:
    min_time = float(plot_frame["time_ms"].min())
    max_time = float(plot_frame["time_ms"].max())
    span = max(max_time - min_time, 0.1)
    padding = span * X_AXIS_PADDING_RATIO
    return max(0.0, min_time - padding), max_time + padding


def legend_handles(
    model_names: list[str], *, full_anomaly_transformer_name: bool = False
) -> list[Line2D]:
    display_names = dict(MODEL_DISPLAY_NAMES)
    if full_anomaly_transformer_name:
        display_names["AnomalyTrans"] = "AnomalyTransformer"
    handles: list[Line2D] = []
    for model_name in model_names:
        marker_size = DEFAULT_MARKER_SIZE * MODEL_MARKER_SIZE_MULTIPLIERS[model_name]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS[model_name],
                color="none",
                label=display_names[model_name],
                markerfacecolor=MODEL_COLORS[model_name],
                markeredgecolor=MARKER_EDGE_COLOR,
                markeredgewidth=MODEL_MARKER_EDGE_WIDTHS[model_name],
                markersize=marker_size**0.5,
            )
        )
    return handles


def available_model_names(plot_frame: pd.DataFrame) -> list[str]:
    available = set(plot_frame["canonical_model"].dropna())
    return [model_name for model_name in MODEL_ORDER if model_name in available]


def formatted_title(template: str, dataset: str, metric: str) -> str:
    return template.format(dataset=dataset, metric=metric) if template.strip() else ""


def style_axis(
    ax: plt.Axes,
    dataset: str,
    metric: str,
    x_limits: tuple[float, float],
    title_template: str,
    show_xlabel: bool,
    show_ylabel: bool,
    show_dataset_label: bool,
) -> None:
    title = formatted_title(title_template, dataset, metric)
    if title:
        ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(X_AXIS_LABEL)
    if show_ylabel:
        ax.set_ylabel(metric)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlim(*x_limits)
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.grid(
        True,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )
    if show_dataset_label:
        ax.text(
            DATASET_LABEL_X,
            DATASET_LABEL_Y,
            dataset,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=DATASET_LABEL_FONT_SIZE,
            bbox={
                "boxstyle": DATASET_LABEL_BOX_STYLE,
                "facecolor": DATASET_LABEL_FACE_COLOR,
                "edgecolor": DATASET_LABEL_EDGE_COLOR,
                "linewidth": DATASET_LABEL_EDGE_WIDTH,
            },
            zorder=5,
        )


def scatter_metric(
    ax: plt.Axes,
    dataset_metric_frame: pd.DataFrame,
    dataset: str,
    metric: str,
    x_limits: tuple[float, float],
    title_template: str,
    show_xlabel: bool,
    show_ylabel: bool,
    add_legend: bool,
    show_dataset_label: bool = False,
    legend_location: str | None = None,
) -> None:
    style_axis(
        ax,
        dataset,
        metric,
        x_limits,
        title_template,
        show_xlabel,
        show_ylabel,
        show_dataset_label,
    )
    available_models: list[str] = []
    for model_name in MODEL_ORDER:
        row = dataset_metric_frame[
            dataset_metric_frame["canonical_model"] == model_name
        ]
        if row.empty:
            continue
        available_models.append(model_name)
        marker_size = DEFAULT_MARKER_SIZE * MODEL_MARKER_SIZE_MULTIPLIERS[model_name]
        ax.scatter(
            row["time_ms"],
            row["score"],
            s=marker_size,
            marker=MODEL_MARKERS[model_name],
            c=MODEL_COLORS[model_name],
            edgecolors=MARKER_EDGE_COLOR,
            linewidths=MODEL_MARKER_EDGE_WIDTHS[model_name],
            zorder=3,
        )
    if add_legend and available_models:
        ax.legend(
            handles=legend_handles(available_models),
            loc=legend_location or LEGEND_LOCATION,
            ncol=LEGEND_COLUMNS,
            frameon=True,
            columnspacing=LEGEND_COLUMN_SPACING,
            handletextpad=LEGEND_HANDLE_TEXT_PAD,
        )


def save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, pad_inches: float
) -> Path:
    output_path = output_dir / f"{stem}.pdf"
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches=SAVE_BBOX,
        pad_inches=pad_inches,
    )
    plt.close(fig)
    return output_path


def create_single_figures(
    plot_frame: pd.DataFrame,
    x_limits: tuple[float, float],
    output_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    for dataset in DATASETS:
        dataset_frame = plot_frame[plot_frame["dataset"] == dataset]
        for metric in METRICS:
            metric_frame = dataset_frame[dataset_frame["metric"] == metric]
            fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
            fig.subplots_adjust(
                left=SINGLE_LEFT,
                right=SINGLE_RIGHT,
                bottom=SINGLE_BOTTOM,
                top=SINGLE_TOP,
            )
            scatter_metric(
                ax,
                metric_frame,
                dataset,
                metric,
                x_limits,
                SINGLE_TITLE_TEMPLATE,
                True,
                True,
                SHOW_SINGLE_LEGEND,
            )
            stem = f"{dataset.lower()}_{metric.lower().replace('-', '_')}"
            outputs.append(save_figure(fig, output_dir, stem, SINGLE_PAD_INCHES))
    return outputs


def create_row_figures(
    plot_frame: pd.DataFrame,
    x_limits: tuple[float, float],
    output_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    for dataset in DATASETS:
        dataset_frame = plot_frame[plot_frame["dataset"] == dataset]
        fig, axes = plt.subplots(
            1,
            len(METRICS),
            figsize=ROW_FIGSIZE,
            sharex=True,
            sharey=True,
        )
        fig.subplots_adjust(
            left=ROW_LEFT,
            right=ROW_RIGHT,
            bottom=ROW_BOTTOM,
            top=ROW_TOP,
            wspace=ROW_WSPACE,
        )
        for index, metric in enumerate(METRICS):
            metric_frame = dataset_frame[dataset_frame["metric"] == metric]
            scatter_metric(
                axes[index],
                metric_frame,
                dataset,
                metric,
                x_limits,
                ROW_TITLE_TEMPLATE,
                True,
                True,
                index == len(METRICS) - 1,
            )
        outputs.append(
            save_figure(
                fig,
                output_dir,
                f"{dataset.lower()}_all_metrics",
                COMBINED_PAD_INCHES,
            )
        )
    return outputs


def create_grid_figure(
    plot_frame: pd.DataFrame,
    x_limits: tuple[float, float],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(DATASETS),
        len(METRICS),
        figsize=GRID_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        wspace=GRID_WSPACE,
        hspace=GRID_HSPACE,
    )
    for row_index, dataset in enumerate(DATASETS):
        dataset_frame = plot_frame[plot_frame["dataset"] == dataset]
        for col_index, metric in enumerate(METRICS):
            metric_frame = dataset_frame[dataset_frame["metric"] == metric]
            scatter_metric(
                axes[row_index, col_index],
                metric_frame,
                dataset,
                metric,
                x_limits,
                GRID_TITLE_TEMPLATE,
                row_index == len(DATASETS) - 1,
                True,
                row_index == GRID_LEGEND_ROW_INDEX
                and col_index == GRID_LEGEND_COL_INDEX,
                show_dataset_label=True,
                legend_location=GRID_LEGEND_LOCATION,
            )
    return save_figure(
        fig,
        output_dir,
        "all_datasets_all_metrics",
        COMBINED_PAD_INCHES,
    )


def create_paper_grid_figure(
    plot_frame: pd.DataFrame,
    x_limits: tuple[float, float],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(DATASETS),
        len(METRICS),
        figsize=GRID_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=0.97,
        wspace=GRID_WSPACE,
        hspace=DATASET_ROWS_HSPACE,
    )

    for row_index, dataset in enumerate(DATASETS):
        for col_index, metric in enumerate(METRICS):
            axis = axes[row_index, col_index]
            metric_frame = plot_frame[
                (plot_frame["dataset"] == dataset)
                & (plot_frame["metric"] == metric)
            ]
            scatter_metric(
                axis,
                metric_frame,
                dataset,
                metric,
                x_limits,
                "",
                row_index == len(DATASETS) - 1,
                True,
                False,
                show_dataset_label=True,
            )
            axis.xaxis.labelpad = COMPACT_AXIS_LABEL_PAD
            axis.yaxis.labelpad = COMPACT_AXIS_LABEL_PAD
            axis.tick_params(pad=COMPACT_TICK_PAD)
            if row_index < len(DATASETS) - 1:
                axis.tick_params(axis="x", labelbottom=False)
            if col_index > 0:
                axis.tick_params(axis="y", labelleft=False)

    available_models = available_model_names(plot_frame)
    if available_models:
        axes[DATASETS.index("SWaT"), METRICS.index("PA-F1")].legend(
            handles=legend_handles(available_models),
            loc="lower center",
            ncol=PANEL_LEGEND_COLUMNS,
            frameon=True,
            fontsize=8.5,
            columnspacing=0.7,
            handletextpad=LEGEND_HANDLE_TEXT_PAD,
            labelspacing=0.45,
            borderpad=0.35,
        )

    return save_figure(
        fig,
        output_dir,
        "all_datasets_all_metrics_2x3_legend_swat_pa_f1",
        SINGLE_PAD_INCHES,
    )


def create_one_row_grid_figure(
    plot_frame: pd.DataFrame,
    x_limits: tuple[float, float],
    output_dir: Path,
) -> Path:
    panel_specs = [(dataset, metric) for dataset in DATASETS for metric in METRICS]
    original_rows = len(DATASETS)
    original_cols = len(METRICS)
    panel_cols = len(panel_specs)
    one_row_figsize = (
        GRID_FIGSIZE[0]
        * (panel_cols + GRID_WSPACE * (panel_cols - 1))
        / (original_cols + GRID_WSPACE * (original_cols - 1)),
        GRID_FIGSIZE[1] / (original_rows + GRID_HSPACE * (original_rows - 1)),
    )
    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=one_row_figsize,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        wspace=GRID_WSPACE,
        hspace=GRID_HSPACE,
    )
    for index, (dataset, metric) in enumerate(panel_specs):
        dataset_frame = plot_frame[plot_frame["dataset"] == dataset]
        metric_frame = dataset_frame[dataset_frame["metric"] == metric]
        scatter_metric(
            axes[index],
            metric_frame,
            dataset,
            metric,
            x_limits,
            GRID_TITLE_TEMPLATE,
            True,
            True,
            False,
            show_dataset_label=True,
        )
    available_models = available_model_names(plot_frame)
    fig.legend(
        handles=legend_handles(
            available_models,
            full_anomaly_transformer_name=True,
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(available_models),
        frameon=True,
        columnspacing=LEGEND_COLUMN_SPACING,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )
    return save_figure(
        fig,
        output_dir,
        "all_datasets_all_metrics_one_row",
        COMBINED_PAD_INCHES,
    )


def create_legend_figure(plot_frame: pd.DataFrame, output_dir: Path) -> Path:
    available_models = available_model_names(plot_frame)
    fig = plt.figure(figsize=(12.0, 0.7))
    fig.legend(
        handles=legend_handles(
            available_models,
            full_anomaly_transformer_name=True,
        ),
        loc="center",
        ncol=len(available_models),
        frameon=True,
        columnspacing=LEGEND_COLUMN_SPACING,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )
    return save_figure(fig, output_dir, "legend", COMBINED_PAD_INCHES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the inference-time trade-off PDF figures."
    )
    parser.add_argument("--time-csv", type=Path, default=DEFAULT_TIME_CSV)
    parser.add_argument("--metric-csv", type=Path, default=DEFAULT_METRIC_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_csv = args.time_csv.expanduser().resolve()
    metric_csv = args.metric_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for input_path in (time_csv, metric_csv):
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    plot_frame = build_plot_frame(time_csv, metric_csv)
    x_limits = resolve_x_limits(plot_frame)
    outputs = create_single_figures(plot_frame, x_limits, output_dir)
    outputs.extend(create_row_figures(plot_frame, x_limits, output_dir))
    outputs.append(create_paper_grid_figure(plot_frame, x_limits, output_dir))
    outputs.append(create_one_row_grid_figure(plot_frame, x_limits, output_dir))
    outputs.append(create_legend_figure(plot_frame, output_dir))

    print(f"Generated {len(outputs)} PDF files in {output_dir}")
    for output_path in outputs:
        print(f"- {output_path.name}")


if __name__ == "__main__":
    main()
