"""Generic tabular and sequence loaders."""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a single Parquet file."""
    return pd.read_parquet(path)


def load_silo_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV for one silo (fallback if Parquet not used)."""
    return pd.read_csv(path)
