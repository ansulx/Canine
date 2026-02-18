"""Simulator parameters for Pillar 3 derived from public data. Stub."""

from typing import Any, Dict


def load_env_params(config: Dict[str, Any]) -> Dict[str, float]:
    """Load env parameters (e.g. mean reversion, shock size) from config or data. Stub."""
    return config.get("env", {})
