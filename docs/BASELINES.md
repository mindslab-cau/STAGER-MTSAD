# Baseline implementations

The baseline implementations come from the sources below. The experiment
pipeline prepares dataset inputs, aligns each model's anomaly scores to
timestamps, and applies the common threshold selection and evaluation.

| Method | Implementation in this repository | Source |
| --- | --- | --- |
| kNN | `pyod.models.knn.KNN` from PyOD 2.0.5. | [PyOD 2.0.5](https://pypi.org/project/pyod/2.0.5/) |
| OCSVM | `pyod.models.ocsvm.OCSVM` from PyOD 2.0.5. | [PyOD OCSVM documentation](https://pyod.readthedocs.io/en/latest/pyod.models.tabular.html#module-pyod.models.ocsvm) |
| Deep SVDD | Model code from the authors' implementation. | [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch/tree/1901612d595e23675fb75c4ebb563dd0ffebc21e) |
| USAD | Model code from the authors' implementation. | [USAD](https://github.com/manigalati/usad/tree/e25af45c8e1c32783aed94fc7e5ab85effef2b1a) |
| TSMixer | PyTorch port of the authors' TensorFlow TSMixer Basic for next-point forecasting. | [Google Research TSMixer Basic](https://github.com/google-research/google-research/tree/master/tsmixer/tsmixer_basic) |
| Anomaly Transformer | Model code from the authors' implementation. | [Anomaly Transformer](https://github.com/thuml/Anomaly-Transformer) |
| FuSAGNet | Model code from the authors' implementation. | [FuSAGNet](https://github.com/seansihohan/FuSAGNet/tree/9ab537e49950278036cb0a944b6448daaa81324f) |
| MtsCID | Model code from the authors' implementation. | [MtsCID](https://github.com/ilwoof/MtsCID/tree/5f94a7e4cdff83fddfeca28c2ba596e3e4d2cac1) |
| MSHTrans | Model code from the authors' implementation, which credits Ada-MSHyper. | [MSHTrans](https://github.com/chenzl23/MSHTrans/tree/e4638abc0145e2cefc817032b8517da2d09c8a28), [Ada-MSHyper](https://github.com/shangzongjiang/Ada-MSHyper) |

The common evaluation code calls the authors' vendored implementation of
enhanced time-series-aware precision and recall (eTaPR). The evaluator integrates it
with local threshold selection, point-wise metrics, and point-adjusted metrics.
[eTaPR source](https://github.com/wshw4ng/eTaPR), [vendored license](../third_party/etapr/LICENSE.txt).

Source licenses are retained with the corresponding components or under
[`third_party/notices/`](../third_party/notices/).
