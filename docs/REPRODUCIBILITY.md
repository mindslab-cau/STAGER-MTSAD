# Reproduce the experiments

Run these commands from the repository root on a Linux host with an NVIDIA
GPU. [Install STAGER](../README.md#1-install) and [prepare SWaT](../README.md#2-prepare-the-datasets)
before running the check.

## Check one run on SWaT

Check the GPU before starting the experiment.

```bash
CUDA_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.ones(1, device="cuda").item())'
```

The command should print `1.0`. Use the same `CUDA_VISIBLE_DEVICES` value
below. The neural launchers select the visible CUDA device by default.

```bash
CUDA_VISIBLE_DEVICES=0 python -m triad_tsad.cli \
  --config configs/stager/swat.yaml \
  --dataset-manifest configs/datasets.local.yaml \
  --output-dir outputs/checks/swat-stager \
  --seed 3974762829 --max-epochs 1

CUDA_VISIBLE_DEVICES=0 python -m triad_tsad.baseline_cli --baseline tsmixer \
  --config configs/baselines/tsmixer/swat.yaml \
  --dataset-manifest configs/datasets.local.yaml \
  --output-dir outputs/checks/swat-tsmixer \
  --seed 3161534991 --max-epochs 1
```

Each command checks data loading, one epoch of training, scoring, and result
writing. It should write `summary.json` and one seed directory containing
`metrics.json`. STAGER also writes `best_model.pt`.

## Run the included models

Prepare all three datasets. The loop runs STAGER and nine baselines with
five seeds per dataset and model. Use a fresh output root for each run.

```bash
for dataset in swat wadi hai; do
  CUDA_VISIBLE_DEVICES=0 python -m triad_tsad.cli \
    --config "configs/stager/$dataset.yaml" \
    --dataset-manifest configs/datasets.local.yaml \
    --output-dir "outputs/paper/table6-full/$dataset/stager"

  for method in knn ocsvm deep_svdd usad tsmixer anomaly_transformer fusagnet mtscid mshtrans; do
    CUDA_VISIBLE_DEVICES=0 python -m triad_tsad.baseline_cli --baseline "$method" \
      --config "configs/baselines/$method/$dataset.yaml" \
      --dataset-manifest configs/datasets.local.yaml \
      --output-dir "outputs/paper/table6-full/$dataset/$method"
  done
done
```

The kNN and OCSVM estimators run on CPU. Each model and dataset directory
contains `summary.json` and per-seed `metrics.json` files.

The [Table 6 provenance](../configs/table6_provenance.json) records the
original or replay seeds, scoring protocols, and available run settings
for each preset. For the paper's standard deviations, use the per-seed
metrics and the row's `table_std_ddof` value.
