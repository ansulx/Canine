"""Load and merge YAML configs. Optional OmegaConf for overrides."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _resolve_paths(config: Dict[str, Any], root: str = ".") -> Dict[str, Any]:
    """Resolve ${data_root} in string values."""
    out = {}
    for k, v in config.items():
        if isinstance(v, dict):
            out[k] = _resolve_paths(v, root)
        elif isinstance(v, str) and "${data_root}" in v:
            out[k] = v.replace("${data_root}", root)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML config from path. If 'defaults' key exists, load those first and merge.
    overrides: optional dict of key paths to values (e.g. {"training.rounds": 10}).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Handle defaults (list of config names relative to config dir)
    defaults = config.pop("defaults", [])
    base_dir = path.parent
    if defaults:
        # Start from first default (e.g. base), then merge each next, then merge current file on top
        config = {}
        for name in defaults:
            default_path = base_dir / (name if name.endswith(".yaml") else f"{name}.yaml")
            if default_path.exists():
                default_config = load_config(str(default_path))
                _deep_merge(config, default_config)
        with open(path) as f2:
            current = yaml.safe_load(f2) or {}
        current.pop("defaults", None)
        _deep_merge(config, current)
    else:
        # No defaults: use current file only (config already loaded above)
        pass

    # Overrides (simple key.path.to.value = val)
    if overrides:
        for key_path, val in overrides.items():
            keys = key_path.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

    # Resolve ${data_root} in strings
    if "data_root" in config:
        config = _resolve_paths(config, str(config["data_root"]))

    return config


def _deep_merge(base: Dict, update: Dict) -> None:
    """In-place merge update into base."""
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
