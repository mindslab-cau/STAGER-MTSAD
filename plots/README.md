# Paper figures

Run the renderer from the repository root in the `stager` environment.
It draws from saved results using NumPy and Matplotlib.

```bash
python plots/paper/render_figures.py
```

The default command writes Figure 4 and Figure 5 PDFs to
`plots/paper/generated/`. Use `--output-dir` to write elsewhere.
For the manuscript typeface, use a system installation of Times New Roman
or pass the directory containing your font files to `--font-dir`.
The renderer uses DejaVu Serif when Times New Roman is unavailable.

| Figure | Output files | Input |
| --- | --- | --- |
| 3. Qualitative predictions | `figure3_swat.pdf`, `figure3_wadi.pdf`, `figure3_hai.pdf`, `figure3_legend.pdf` | Local selected masks. |
| 4. Auxiliary loss sensitivity | `figure4_alpha.pdf` above `figure4_beta.pdf` | [Saved sweep coordinates](paper/data/figure4_sensitivity.csv). |
| 5. Accuracy and inference time | `figure5_legend.pdf` above `figure5_tradeoff.pdf` | [Saved accuracy and per-window timing](paper/data/figure5_tradeoff.csv). |

## Figure 3 data

Place a NumPy `.npz` archive of test labels, scored rows, and predictions
aligned to the original test CSV at `plots/paper/inputs/selected_raw_masks.npz`.

```bash
python plots/paper/render_figures.py --figure 3
```

Pass a different archive path with `--masks`. The default input directory
is Git-ignored.
Use one-dimensional binary arrays on each dataset's original test row axis.
Use `{dataset}_gt` for labels and `{dataset}_{model}_observed` and
`{dataset}_{model}_prediction` for scored rows and pointwise predictions.
The dataset IDs are `swat`, `wadi`, and `hai`. The model IDs are `knn`,
`ocsvm`, `deepsvdd`, `usad`, `tsmixer`, `anomaly_transformer`, `fusagnet`,
`mtscid`, `mshtrans`, and `triad` for STAGER.

For the paper illustration each baseline shows the run with the most
separate false-positive intervals among runs at or below its median recall.
STAGER uses one reference run per dataset.
The renderer classifies TP and FP before applying display-only expansion.
