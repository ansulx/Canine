# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

FedStable is a pure-Python federated learning platform (Flower + PyTorch) for stablecoin depeg and DeFi risk prediction. No Docker, databases, or external services are needed — everything runs locally in simulation mode. See `README.md` for full specification.

### Running services

| Service | Command | Notes |
|---------|---------|-------|
| Tests | `pytest tests/ -v` | 4 tests covering model shapes, silo partitioning, env step |
| Pillar 1 training | `python3 scripts/train_pillar1.py --config configs/pillar1.yaml` | Currently a stub; outputs `artifacts/pillar1/metrics.json` |
| Pillar 2 training | `python3 scripts/train_pillar2.py --config configs/pillar2.yaml` | Stub |
| Pillar 3 training | `python3 scripts/train_pillar3.py --config configs/pillar3.yaml` | Stub |

### Non-obvious caveats

- **No linter configured**: The project has no ruff, flake8, pylint, or mypy configuration. Python syntax checking is limited to `pytest` and `python3 -c "import fedstable"`.
- **Editable install required**: Scripts in `scripts/` import `fedstable` as a package. The update script runs `pip install -e .` to make this work. Without it, scripts fail with `ModuleNotFoundError`.
- **pyproject.toml fix**: The original `pyproject.toml` had an invalid `role` field in `project.authors` that prevented editable install. This was fixed by removing the field.
- **Device config defaults to CUDA**: `configs/base.yaml` sets `device: cuda`. On CPU-only machines, override with `device: cpu` or the training code should fall back gracefully (training is currently a stub).
- **Data directories**: `data/raw/`, `data/processed/`, and `artifacts/` must exist for scripts to write output. Create them with `mkdir -p data/raw data/processed artifacts`.
- **PATH**: pip installs scripts to `~/.local/bin`. Ensure this is on PATH: `export PATH="$HOME/.local/bin:$PATH"`.
