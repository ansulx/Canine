"""Secure aggregation interface. Stub: no-op; real implementation in future."""

from typing import List

import numpy as np


def mask_update(update: List[np.ndarray], client_id: int, round_id: int) -> List[np.ndarray]:
    """Mask client update for secure aggregation. Stub: returns as-is."""
    return update


def unmask_aggregate(masked_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Unmask and sum updates. Stub: returns sum of masked_updates (no crypto)."""
    if not masked_updates:
        return []
    out = [np.array(x) for x in masked_updates[0]]
    for upd in masked_updates[1:]:
        for i, arr in enumerate(upd):
            out[i] = out[i] + arr
    return out
