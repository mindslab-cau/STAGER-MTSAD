"""Project-wide random seed initialization."""

from __future__ import annotations

import random
import importlib
from numbers import Integral

import numpy as np


def set_random_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, PyTorch, and available CUDA devices.

    This preserves the seed behavior used by the original TSAD experiment
    harness. It configures deterministic cuDNN algorithm selection without
    claiming that every PyTorch or PyTorch Geometric operation is
    deterministic.
    """

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= int(seed) <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    seed = int(seed)

    torch = importlib.import_module("torch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
