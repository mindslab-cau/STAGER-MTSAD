"""Render manuscript Figures 3–5 from selected masks and saved measurements."""

import argparse
import csv
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
DATASETS = ("SWaT", "WADI", "HAI")
MASK_MODELS = (
    ("knn", "KNN"), ("ocsvm", "OCSVM"), ("deepsvdd", "DeepSVDD"),
    ("usad", "USAD"), ("tsmixer", "TSMixer"),
    ("anomaly_transformer", "AnomalyTrans"), ("fusagnet", "FuSAGNet"),
    ("mtscid", "MtsCID"), ("mshtrans", "MSHTrans"), ("stager", "STAGER"),
)
METRICS = (("f1", "F1"), ("pa_f1", "PA-F1"), ("eTaf1", "eTaF1"))
SENSITIVITY_STYLE = (("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^"))
TIMING_STYLE = {
    "usad": ("USAD", "o", "#90CAF9", 1., .7),
    "tsmixer": ("TSMixer", "s", "#F5BCDB", 1., .7),
    "anomaly_transformer": ("AT", "^", "#F27457", 1., .7),
    "fusagnet": ("FuSAGNet", "D", "#48B5A3", .8, .7),
    "mtscid": ("MtsCID", "P", "#6863F2", 1., .7),
    "mshtrans": ("MSHTrans", "X", "#585934", 1., .7),
    "stager": ("STAGER", "*", "#F2C572", 2., 1.),
}
ENLARGEMENT = 1.5
BASE_PAPER_WIDTH_PT = 594.398125 * .37936


def configure_font(font_dir):
    if font_dir is not None:
        if not font_dir.is_dir():
            raise ValueError(f"Font directory does not exist: {font_dir}")
        for path in sorted(font_dir.iterdir()):
            if path.suffix.lower() in {".otf", ".ttf"}:
                font_manager.fontManager.addfont(path)
    family = "Times New Roman"
    try:
        font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        family = "DejaVu Serif"
        warnings.warn("Times New Roman was unavailable. Using DejaVu Serif.", stacklevel=2)
    math_fonts = {name: family for name in ("rm", "sf", "tt", "cal")}
    math_fonts.update(it=f"{family}:italic", bf=f"{family}:bold", bfit=f"{family}:bold:italic")
    plt.rcParams.update({
        "font.family": family, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.fallback": None,
        **{f"mathtext.{name}": font for name, font in math_fonts.items()},
    })


def save_pdf(figure, output, name, **kwargs):
    destination = output / f"{name}.pdf"
    figure.savefig(destination, **kwargs)
    plt.close(figure)
    print(destination)


def read_binary(archive, key, expected_length=None):
    if key not in archive:
        raise ValueError(f"Missing Figure 3 mask: {key}")
    values = archive[key]
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError(f"Figure 3 mask must be one-dimensional and binary: {key}")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"Figure 3 mask length differs from ground truth: {key}")
    return values.astype(bool)


def dilate(mask, radius):
    return np.convolve(mask.astype(np.int32), np.ones(2 * radius + 1, dtype=np.int32), mode="same") > 0


def intervals(mask):
    delta = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(delta == 1)
    lengths = np.flatnonzero(delta == -1) - starts
    return list(zip(starts.tolist(), lengths.tolist()))


def render_predictions(output, path):
    if not path.is_file():
        raise ValueError(f"Figure 3 needs a local mask archive: {path}. Supply it with --masks.")
    gt_color, tp_color, fp_color = "#d62728", "#1f77b4", "#ff8a00"
    panel_width = 1116.79208 * 2 / 3
    width = panel_width + 2 * 1.44
    height = 974.653175 - 48 - 64 + 6 + 36
    row_height = (1 - (6 + 36) / height) / len(MASK_MODELS)
    with np.load(path, allow_pickle=False) as archive:
        for dataset, radius, fp_radius in zip(DATASETS, (20, 10, 16), (15, 6, 12)):
            key = dataset.lower()
            truth = read_binary(archive, f"{key}_gt")
            if len(truth) < 2 * radius + 1:
                raise ValueError(f"Figure 3 timeline is shorter than the display window: {dataset}")
            display_gt = dilate(truth, radius)
            figure = plt.figure(figsize=(width / 72, height / 72))
            for row, (model, label) in enumerate(MASK_MODELS):
                mask_prefix = f"{key}_{model}"
                if (
                    model == "stager"
                    and f"{mask_prefix}_observed" not in archive
                    and f"{mask_prefix}_prediction" not in archive
                ):
                    legacy_prefix = f"{key}_triad"
                    if (
                        f"{legacy_prefix}_observed" in archive
                        and f"{legacy_prefix}_prediction" in archive
                    ):
                        mask_prefix = legacy_prefix
                observed = read_binary(archive, f"{mask_prefix}_observed", len(truth))
                predicted = read_binary(archive, f"{mask_prefix}_prediction", len(truth)) & observed
                tp, fp = predicted & truth, predicted & ~truth
                display_tp, display_fp = dilate(tp, radius), dilate(fp, fp_radius)
                display_tp[fp] = False
                display_fp[tp] = False
                axis = figure.add_axes([
                    1.44 / width, 1 - 6 / height - (row + 1) * row_height,
                    panel_width / width, row_height,
                ])
                axis.broken_barh(intervals(display_gt), (0, .55), facecolors=gt_color,
                                 alpha=.40, edgecolors="none", linewidth=0, zorder=1)
                axis.broken_barh(intervals(display_tp), (0, .95), facecolors=tp_color,
                                 alpha=.85, edgecolors="none", linewidth=0, zorder=2)
                positions = np.flatnonzero(display_fp) + .5
                if positions.size:
                    segments = [((x, 0), (x, .95)) for x in positions]
                    lines = LineCollection(
                        segments, colors=fp_color, linewidths=.70,
                        linestyles=(0, (14., 7., 2., 7.)), alpha=.95, zorder=3,
                    )
                    lines.set_capstyle("butt")
                    axis.add_collection(lines)
                axis.text(.006, .94, label, va="top", ha="left", transform=axis.transAxes,
                          fontsize=25, zorder=5, bbox={
                              "boxstyle": "square,pad=.18", "facecolor": "white",
                              "edgecolor": "#999999", "linewidth": .5, "alpha": .96,
                          })
                axis.set(xlim=(0, len(truth)), ylim=(0, 1), xticks=[], yticks=[])
                for spine in axis.spines.values():
                    spine.set_linewidth(.8)
                if row == len(MASK_MODELS) - 1:
                    axis.set_xlabel("Time index", fontsize=24, labelpad=2)
            save_pdf(figure, output, f"figure3_{key}")

    figure = plt.figure(figsize=(14, 1))
    handles = [
        Patch(facecolor=gt_color, alpha=.40, label=": Ground Truth Anomaly"),
        Patch(facecolor=tp_color, alpha=.85, label=": True Positive"),
        Line2D([0], [0], color=fp_color, linestyle=(0, (4, 2, 1, 2)),
               linewidth=2., label=": False Positive"),
    ]
    legend = figure.legend(handles=handles, loc="center", ncol=3, frameon=False,
                           fontsize=24, handletextpad=.15, columnspacing=.85, handlelength=1.9)
    figure.canvas.draw()
    bbox = legend.get_window_extent().transformed(figure.dpi_scale_trans.inverted()).padded(2 / 72)
    save_pdf(figure, output, "figure3_legend", bbox_inches=bbox, pad_inches=0)


def render_sensitivity(output):
    with (HERE / "data/figure4_sensitivity.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # These are the page bounds and axis positions of the manuscript components.
    width_pt, full_height_pt = 891.5971875, 222.0850125
    scale = BASE_PAPER_WIDTH_PT / (width_pt / ENLARGEMENT)
    paper_width = width_pt * scale
    paper_height = full_height_pt * scale
    axis_width = (paper_width - 25 - 3 - 2 * 27) / 3
    axis_height = paper_height - 20 - 23
    limits = {"SWaT": (55, 100), "WADI": (50, 100), "HAI": (40, 95)}
    levels = [0., .1, .2, .3, .4]
    for fixed, symbol, vary, fixed_label, label, height_pt in (
        ("cls", "alpha", "alpha_reg", r"Cls. $\beta={weight:g}$", r"Reg. ($\alpha$)", full_height_pt),
        ("reg", "beta", "beta_cls", r"Reg. $\alpha={weight:g}$", r"Cls. ($\beta$)", 190.75167917),
    ):
        figure = plt.figure(figsize=(width_pt / 72, height_pt / 72))
        height = height_pt * scale
        for column, dataset in enumerate(DATASETS):
            points = [row for row in rows if row["dataset"] == dataset and row["fixed_parameter"] == fixed]
            points.sort(key=lambda row: float(row[vary]))
            if [float(row[vary]) for row in points] != levels:
                raise ValueError(f"Incomplete Figure 4 weights for {dataset}/{symbol}")
            reference = next(row for row in points if row["reference_point"] == "True")
            axis = figure.add_axes([(25 + column * (axis_width + 27)) / paper_width,
                                    20 / height, axis_width / paper_width, axis_height / height])
            for (field, metric), (color, marker) in zip(METRICS, SENSITIVITY_STYLE):
                axis.plot(levels, [float(row[field]) * 100 for row in points],
                          label=metric, color=color, marker=marker,
                          linewidth=1.9 * ENLARGEMENT, markersize=5 * ENLARGEMENT,
                          markeredgewidth=ENLARGEMENT)
            axis.axvline(float(reference[vary]), color="#777777", linestyle="--",
                         linewidth=.9 * ENLARGEMENT, alpha=.7, zorder=0)
            for (field, _), (color, marker) in zip(METRICS, SENSITIVITY_STYLE):
                axis.plot(float(reference[vary]), float(reference[field]) * 100, marker=marker,
                          markerfacecolor="white", markeredgecolor=color,
                          linewidth=1.5 * ENLARGEMENT, markeredgewidth=1.8 * ENLARGEMENT,
                          markersize=8 * ENLARGEMENT, zorder=5)
            setting = fixed_label.format(weight=float(reference["fixed_weight"]))
            axis.set_title(f"{dataset} | {setting}", fontsize=8.5 / scale, pad=2 / scale)
            axis.set_xlabel(label, fontsize=8 / scale, labelpad=1.3 / scale)
            axis.set_ylabel("Score (%)", fontsize=8 / scale, labelpad=1.3 / scale)
            axis.set_xticks(levels, [f"{level:g}" for level in levels])
            axis.set_xlim(-.02, .42)
            axis.set_yticks([value for value in range(40, 101, 20) if limits[dataset][0] <= value <= limits[dataset][1]])
            axis.set_ylim(*limits[dataset])
            axis.tick_params(labelsize=7 / scale, length=2 / scale, pad=1.25 / scale)
            axis.grid(alpha=.22, linestyle="--")
            axis.spines[["top", "right"]].set_visible(False)
        if fixed == "cls":
            handles, labels = axis.get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                          fontsize=7.5 / scale, borderaxespad=.23, borderpad=.16,
                          handlelength=1.7, handletextpad=.4, columnspacing=1.4)
        save_pdf(figure, output, f"figure4_{symbol}")


def render_tradeoff(output):
    with (HERE / "data/figure5_tradeoff.csv").open(newline="", encoding="utf-8") as stream:
        rows = {(row["dataset"], row["model"]): row for row in csv.DictReader(stream)}
    width_pt, height_pt = 847.8871875, 774.24805756
    scale = BASE_PAPER_WIDTH_PT / (width_pt / ENLARGEMENT)
    paper_width, paper_height = (point * scale for point in (width_pt, height_pt))
    panel_width = (paper_width - 23 - 3 - 2 * 14) / 3
    panel_height = (paper_height - 20 - 5 - 2 * 13) / 3
    figure = plt.figure(figsize=(width_pt / 72, height_pt / 72))
    for row, dataset in enumerate(DATASETS):
        for col, (metric, ylabel) in enumerate(METRICS):
            axis = figure.add_axes([(23 + col * (panel_width + 14)) / paper_width,
                                    (20 + (2 - row) * (panel_height + 13)) / paper_height,
                                    panel_width / paper_width, panel_height / paper_height])
            for model, (_, marker, color, multiplier, edge) in TIMING_STYLE.items():
                point = rows[dataset, model]
                axis.scatter(float(point["cuda_ms_per_window"]), float(point[metric]),
                             s=90 * multiplier * ENLARGEMENT ** 2, marker=marker, c=color,
                             edgecolors="black", linewidths=edge * ENLARGEMENT, zorder=3)
            max_latency = 5 if dataset == "HAI" else 6.5
            axis.set_xticks(range(math.floor(max_latency) + 1))
            axis.set_xlim(0, max_latency)
            axis.set_yticks([i / 5 for i in range(6)])
            axis.set_ylim(0, 1)
            axis.tick_params(labelsize=7 / scale, labelleft=col == 0, length=2 / scale, pad=1.25 / scale)
            axis.set_ylabel(ylabel, fontsize=8 / scale, labelpad=1.3 / scale)
            if row == 2:
                axis.set_xlabel("Inference Time (ms)", fontsize=8 / scale, labelpad=1.3 / scale)
            axis.grid(True, linestyle="--", linewidth=.7 * ENLARGEMENT, alpha=.22)
            axis.text(.5, .965, dataset, transform=axis.transAxes, ha="center", va="top",
                      fontsize=8.5 / scale, zorder=5, bbox={
                          "boxstyle": "square,pad=0.2", "fc": "white",
                          "ec": "black", "lw": .7 * ENLARGEMENT,
                      })
    save_pdf(figure, output, "figure5_tradeoff")

    legend_height = 30.08147256
    figure = plt.figure(figsize=(width_pt / 72, legend_height / 72))
    axis = figure.add_axes([0, 0, 1, 1], xlim=(0, width_pt), ylim=(0, legend_height))
    axis.set_axis_off()
    # Marker centers and text origins in points from the manuscript's detached legend.
    positions = (
        (44.147402, 60.598208), (147.037434, 163.488239), (266.630589, 283.081395),
        (341.270620, 357.721425), (477.582526, 494.033331),
        (590.910057, 607.360862), (725.550088, 742.000893),
    )
    for (marker_x, text_x), (label, marker, color, multiplier, edge) in zip(positions, TIMING_STYLE.values()):
        axis.plot(marker_x, 17.048549, linestyle="none", marker=marker,
                  markersize=math.sqrt(90 * multiplier) * ENLARGEMENT,
                  markerfacecolor=color, markeredgecolor="black", markeredgewidth=edge * ENLARGEMENT)
        axis.text(text_x, 10.468227, label, fontsize=7.5 / scale)
    save_pdf(figure, output, "figure5_legend", transparent=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=(3, 4, 5), action="append",
                        help="Repeat to select multiple figures. Default: 4 and 5.")
    parser.add_argument("--masks", type=Path, default=HERE / "inputs/selected_raw_masks.npz",
                        help="Local Figure 3 mask archive.")
    parser.add_argument("--font-dir", type=Path, help="Optional directory of local font files.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    figures = set(args.figure or (4, 5))
    configure_font(args.font_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if 3 in figures:
        render_predictions(args.output_dir, args.masks)
    if 4 in figures:
        render_sensitivity(args.output_dir)
    if 5 in figures:
        render_tradeoff(args.output_dir)


if __name__ == "__main__":
    main()
