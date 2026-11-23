# Datasets

The STAGER study uses three industrial control system benchmarks. Obtain SWaT
and WADI from SUTD iTrust and HAI from the dataset authors. The Table 6
presets include all three datasets.
[iTrust dataset request](https://www.sutd.edu.sg/itrust/request-for-datasets/),
[HAI download](https://github.com/icsdataset/hai#getting-the-dataset).

| Dataset | Release used in the study | Features used |
| --- | --- | ---: |
| Secure Water Treatment (SWaT) | SWaT.A2 Dec 2015, using Normal v1 and Attack v0 | 51 |
| Water Distribution (WADI) | WADI.A2, 2019 revision | 123 |
| HIL-based Augmented ICS (HAI) | HAI 22.04 | 86 |

## SWaT

SWaT is a scaled-down, operational six-stage water treatment testbed. The
December 2015 collection covers seven days of normal operation and four days
with attack scenarios. The experiments use the physical measurements from all
51 sensor and actuator channels. The publisher's Normal v1 file already
excludes the first 30 minutes of Normal v0.
[iTrust SWaT release notes](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/)

The preprocessor reads `Physical/SWaT_Dataset_Normal_v1.csv` for training
and `Physical/SWaT_Dataset_Attack_v0.csv` for testing. It drops the first
21,600 rows (six hours) from training. It groups both splits into 10-second
timestamp bins and takes each sensor's median and the mode of the labels.
An equal count of `Normal` and `Attack` resolves to `Attack`.

## WADI

WADI is a water distribution testbed with three main stages, the second of
which has two substages. The original collection covers 14 days of normal
operation and two days with attack scenarios. The 2019 WADI.A2
revision removes readings from unstable periods of normal operation and
adds attack labels. We use 123 feature channels. The input CSV layout has 127
candidate feature columns, of which four are entirely missing and removed.
Row number, date, time, and attack labels are metadata.
[iTrust WADI release notes](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/wadi/),
[WADI testbed paper](https://doi.org/10.1145/3055366.3055375).

The preprocessor reads `WADI_14days_new.csv` and
`WADI_attackdataLABLE.csv`. It strips the optional numbered export header
and trailer rows with empty row, date, and time fields, parses timestamps,
and fills remaining sensor gaps with the column mean within each split.
It drops the first 21,600 training rows before grouping both splits into
10-second timestamp bins. Each bin uses the sensor medians and label mode.
A tie resolves to `-1` (attack), and `1` denotes normal.

The paper used the cleaned test export `WADI_attackdataLABLE_fix.csv`.
Pass that file as `--attack-raw` if available. Equivalence of the prepared
test CSVs from the provider's export and the cleaned export is unverified.

## HAI

HAI uses a hardware-in-the-loop industrial control testbed with boiler,
turbine, water treatment, and simulation processes. The 22.04 release
contains 86 measured variables, 279 hours of normal training data across six
recordings, and 100.3 hours of test data across four recordings with 58
recorded attacks. [HAI dataset and testbed description](https://github.com/icsdataset/hai#hai-dataset),
[HAI 22.04 files](https://github.com/icsdataset/hai/tree/master/hai-22.04).

Download `train1.csv` through `train6.csv` and `test1.csv` through
`test4.csv`, including the data behind the Git LFS pointers if cloning the
source repository. The preprocessor concatenates recordings in numeric order
within each split. It groups consecutive blocks of ten rows starting at the
first row, keeping the first timestamp, the sensor medians, and the label
mode. A tie resolves to `0` (normal). It discards any incomplete final
block. The resulting data has a nominal 10-second period.
[HAI preprocessor](../src/triad_tsad/data_prep/hai.py)

## Preparing datasets

Run the preprocessor and register the CSV paths using the
[README commands](../README.md#2-prepare-the-datasets). It writes
`train.csv`, `test.csv`, `dataset_manifest.json`, and `preparation.json`
for each dataset. [Preprocessor source](../src/triad_tsad/data_prep/prepare.py)

When building forecasting windows, the STAGER CSV loader retains a window
of length `W` if its target timestamp is `W × 10` seconds after the first
input timestamp. [Window construction](../src/triad_tsad/utils/data.py)
