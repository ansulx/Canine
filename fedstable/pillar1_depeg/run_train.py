"""Orchestrate Pillar 1 federated training. Returns metrics dict."""

from pathlib import Path
from typing import Any, Dict, Optional

from fedstable.core.config import load_config
from fedstable.core.random import set_seed


def run_train(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load config, set seed, run FL training for Pillar 1, save checkpoint and metrics.
    Returns dict with final metrics (e.g. loss, auc if evaluated).
    """
    config = load_config(config_path, overrides)
    set_seed(config.get("seed", 42))
    # TODO: implement full FL loop with Flower server + clients; save artifacts
    return {"status": "stub", "config_path": config_path}
