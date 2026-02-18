"""Partition data into silos by config (by_stablecoin, random, etc.)."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def partition_data_by_config(
    df: pd.DataFrame,
    silo_config: Dict[str, Any],
    partition_strategy: Optional[str] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Partition DataFrame into per-silo DataFrames.
    silo_config: dict with 'partition_strategy', 'silos' (list of {id, name, filter?}).
    Returns dict mapping silo id -> DataFrame.
    """
    strategy = partition_strategy or silo_config.get("partition_strategy", "random")
    silos = silo_config.get("silos", [])
    if not silos:
        raise ValueError("silo_config must contain 'silos' list")

    out: Dict[int, pd.DataFrame] = {}

    if strategy == "by_stablecoin":
        for s in silos:
            sid = s.get("id", len(out))
            filt = s.get("filter", {})
            if "stablecoin" in filt:
                subset = df[df["stablecoin"] == filt["stablecoin"]]
            else:
                subset = df
            out[sid] = subset
    elif strategy == "random":
        n = len(silos)
        if "client_id" in df.columns:
            # Use existing client_id if present
            for s in silos:
                sid = s.get("id", len(out))
                out[sid] = df[df["client_id"] == sid]
        else:
            # Assign random partition
            df = df.copy()
            df["_partition"] = np.random.randint(0, n, size=len(df))
            for s in silos:
                sid = s.get("id", len(out))
                out[sid] = df[df["_partition"] == sid].drop(columns=["_partition"])
    else:
        raise ValueError(f"Unknown partition_strategy: {strategy}")

    return out
