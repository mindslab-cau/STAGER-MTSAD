# STAGER-MTSAD

**Scale-Robust and Global-Aware Representation Learning for Multivariate Time Series Anomaly Detection**

Gyuwon Lee, Saejoon Park, Won-Seok Hwang, Bogoan Kim, and Yunyong Ko

STAGER detects anomalies by forecasting the next value of each sensor and
scoring its prediction error. It uses a temporal directed acyclic graph to
learn patterns at short and long time scales in each sensor's history. Learned
sensor profiles condition the local representations, and graph attention
models dependencies across sensors. The study compares STAGER with nine
baselines on SWaT, WADI, and HAI 22.04.

## 1. Install

Use Conda on a Linux machine with an NVIDIA GPU for the paper experiments.
Run these commands from the repository root. The environment uses Python 3.11
and PyTorch 2.4 with CUDA 11.8 and installs the STAGER commands below.

```bash
conda env create --file environment.yml
conda activate stager
```

If you already have a `stager` environment, run `python -m pip install --no-deps -e .`
from this checkout to refresh the editable installation and console commands.

## 2. Prepare the datasets

Request SWaT and WADI from SUTD iTrust and get the HAI 22.04 recordings
from their authors. Use SWaT December 2015 (Normal v1 and Attack v0), the
2019 WADI.A2 revision, and HAI 22.04. [SWaT and WADI request form](https://www.sutd.edu.sg/itrust/request-for-datasets/) · [HAI 22.04 download](https://github.com/icsdataset/hai/tree/master/hai-22.04)

### Run the preprocessor on the raw files

The examples read downloads from `data/raw/`. Adjust the input paths to match
your files. For HAI, `--raw-dir` is the folder that contains `train1.csv`
through `train6.csv` and `test1.csv` through `test4.csv`.

```bash
stager-prepare-data swat \
  --normal-raw data/raw/swat/Physical/SWaT_Dataset_Normal_v1.csv \
  --attack-raw data/raw/swat/Physical/SWaT_Dataset_Attack_v0.csv \
  --output-dir data/prepared/swat

stager-prepare-data wadi \
  --normal-raw data/raw/wadi/WADI_14days_new.csv \
  --attack-raw data/raw/wadi/WADI_attackdataLABLE.csv \
  --output-dir data/prepared/wadi

stager-prepare-data hai \
  --raw-dir data/raw/hai-22.04 \
  --output-dir data/prepared/hai
```

Each command writes `train.csv` and `test.csv`. [Dataset versions and preprocessing](docs/DATASETS.md)

### Set the paths to the prepared CSVs

Create a local copy of the dataset manifest.

```bash
if [ ! -e configs/datasets.local.yaml ]; then
  cp configs/datasets.yaml configs/datasets.local.yaml
fi
```

The template already points to the `train.csv` and `test.csv` files in
`data/prepared/<dataset>/` when you use the output directories above. If you
already have compatible 10-second CSVs or used other output directories, edit the
`train_path` and `test_path` fields under `datasets.swat`, `datasets.wadi`,
or `datasets.hai` in `configs/datasets.local.yaml`. Use absolute paths or
paths relative to the manifest's directory. Pass that file as
`--dataset-manifest` when running a model. [Dataset manifest template](configs/datasets.yaml)

## 3. Run STAGER or a baseline

The commands below run the selected Table 6 settings with all five seeds in
each preset. To change datasets, select `swat`, `wadi`, or `hai` in the preset
path and choose a fresh output directory.

```bash
CUDA_VISIBLE_DEVICES=0 python -m stager_mtsad.cli \
  --config configs/stager/swat.yaml \
  --dataset-manifest configs/datasets.local.yaml \
  --output-dir outputs/stager/swat

CUDA_VISIBLE_DEVICES=0 python -m stager_mtsad.baseline_cli --baseline tsmixer \
  --config configs/baselines/tsmixer/swat.yaml \
  --dataset-manifest configs/datasets.local.yaml \
  --output-dir outputs/tsmixer/swat
```

Each run writes `summary.json` with the seed aggregates and per-seed
`metrics.json` files. Use `--seed N` to run one seed or `--seed-iter N`
to draw and record a new set of seeds.

### GPU selection and CPU affinity

`CUDA_VISIBLE_DEVICES` selects which physical GPU is visible. With one GPU
visible, the neural launchers select it automatically when CUDA is available.
To pin CPU cores on Linux, insert `taskset -c 0-5` before `python` and
adjust the core list for your machine.

- [Check installation on SWaT and run the included models](docs/REPRODUCIBILITY.md)
- [Reproduce manuscript Figures 3–5](plots/README.md)
- [Baseline implementation sources](docs/BASELINES.md)

### Optional hyperparameter sweeps

Use the same commands with the `swat_sweep.yaml` in each runnable model's config
directory and a fresh output directory.

```bash
CUDA_VISIBLE_DEVICES=0 python -m stager_mtsad.cli \
  --config configs/stager/swat_sweep.yaml \
  --dataset-manifest configs/datasets.local.yaml \
  --output-dir outputs/stager/swat_sweep

CUDA_VISIBLE_DEVICES=0 python -m stager_mtsad.baseline_cli --baseline tsmixer \
  --config configs/baselines/tsmixer/swat_sweep.yaml \
  --dataset-manifest configs/datasets.local.yaml \
  --output-dir outputs/tsmixer/swat_sweep
```

The YAML points to a base preset with `base_config`. Use `values: [...]`
for alternatives and `value: ...` for a fixed setting such as a list of
`ms_windows`. Set `seeds: [...]` or `seed_iter` in the sweep YAML, or
inherit the base preset's seed settings. Every combination uses the same
seed list.

The STAGER example schedules 375 runs. Add `--plan` to preview the
combinations before training, or `--resume` to continue an interrupted
sweep. [STAGER YAML](configs/stager/swat_sweep.yaml) · [TSMixer YAML](configs/baselines/tsmixer/swat_sweep.yaml)
