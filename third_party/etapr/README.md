# eTaPR

This directory contains the eTaPR implementation imported as `eTaPR_pkg`.
STAGER's evaluator calls `eTaPR_pkg.etapr.evaluate_w_streams` for its
eTaPR scores. [Upstream implementation](https://github.com/wshw4ng/eTaPR) ·
[Bundled license](LICENSE.txt)

The [Conda environment](../../environment.yml) installs this local package
from the repository root. To install this package independently in an
existing environment, run the following from the same directory.

```bash
python -m pip install -e ./third_party/etapr
```

The pinned `faster-eTaPR==0.1.2` package is a separate PyPI dependency.
The evaluator also imports it.
