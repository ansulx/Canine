"""Data schemas and validation for Pillar 1 depeg features."""

DEPEG_FEATURE_COLUMNS = [
    "timestamp",
    "client_id",
    "stablecoin",
    "price",
    "peg_deviation_pct",
    "volume_24h",
    "volatility_24h",
    "supply",
    "reserve_proxy",
]
DEPEG_LABEL_COLUMNS = ["label_depeg_1h", "label_depeg_6h", "label_depeg_24h"]


def validate_depeg_schema(df) -> bool:
    """Check that DataFrame has required columns for depeg task. Returns True if valid."""
    import pandas as pd
    required = DEPEG_FEATURE_COLUMNS + DEPEG_LABEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True
