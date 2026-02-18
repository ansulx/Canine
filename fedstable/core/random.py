"""Reproducibility: set all random seeds."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for numpy, torch, and Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: deterministic CuDNN (may slow training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
