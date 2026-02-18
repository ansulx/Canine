"""Differential privacy: gradient clipping and noise."""

from typing import List

import numpy as np


def apply_dp_gradient(
    parameters: List[np.ndarray],
    clip_norm: float = 1.0,
    noise_scale: float = 0.01,
    rng: np.random.Generator = None,
) -> List[np.ndarray]:
    """
    Clip combined gradient (as list of arrays) by global L2 norm, then add Gaussian noise.
    In practice, parameters here are the model update (current - previous) or gradient.
    """
    rng = rng or np.random.default_rng()
    total_norm = np.sqrt(sum(np.sum(p ** 2) for p in parameters))
    if total_norm > clip_norm and total_norm > 0:
        scale = clip_norm / total_norm
        parameters = [p * scale for p in parameters]
    return [p + noise_scale * rng.standard_normal(p.shape, dtype=p.dtype) for p in parameters]
