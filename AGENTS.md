# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

FedStable is a federated learning platform (Flower + PyTorch) for stablecoin depeg and DeFi risk prediction. It includes a FastAPI backend API and a React + Tailwind CSS dashboard. No Docker, databases, or external services needed — everything runs locally in simulation mode. See `README.md` for the full specification.

### Running services

| Service | Command | Port | Notes |
|---------|---------|------|-------|
| Backend API | `uvicorn fedstable.api.app:app --host 0.0.0.0 --port 8000` | 8000 | FastAPI; serves REST API and static dashboard build |
| Dashboard (dev) | `cd dashboard && npx vite --host 0.0.0.0 --port 5173` | 5173 | React dev server; proxies `/api` to backend |
| Tests | `pytest tests/ -v` | — | 4 tests: model shapes, silo partitioning, env step |
| Pillar 1 training | `python3 scripts/train_pillar1.py --config configs/pillar1.yaml` | — | Outputs `artifacts/pillar1/metrics.json` |
| Pillar 2 training | `python3 scripts/train_pillar2.py --config configs/pillar2.yaml` | — | Stub |
| Pillar 3 training | `python3 scripts/train_pillar3.py --config configs/pillar3.yaml` | — | Stub |

**To run the full dashboard:** Start both the backend API (port 8000) and the dashboard dev server (port 5173) simultaneously. The Vite dev server proxies API requests to the backend.

### Non-obvious caveats

- **Editable install required**: Scripts in `scripts/` and the API import `fedstable` as a package. The update script runs `pip install -e .` to make this work. Without it, imports fail with `ModuleNotFoundError`.
- **pyproject.toml fix**: The original `pyproject.toml` had an invalid `role` field in `project.authors` that prevented editable install. This was fixed by removing the field.
- **Device config defaults to CUDA**: `configs/base.yaml` sets `device: cuda`. On CPU-only machines, the training code falls back gracefully (training is currently a stub).
- **Data directories**: `data/raw/`, `data/processed/`, and `artifacts/` must exist for scripts to write output. Created by the update script.
- **PATH**: pip installs scripts to `~/.local/bin`. Ensure this is on PATH: `export PATH="$HOME/.local/bin:$PATH"`.
- **No linter configured**: The project has no ruff, flake8, pylint, or mypy configuration.
- **Dashboard build**: `cd dashboard && npm run build` creates a production build in `dashboard/dist/`. The FastAPI app auto-serves this if the directory exists.
- **WebSocket for training**: The Training page uses a WebSocket connection (`/ws/training`) for real-time round updates. Both backend and frontend dev server must be running.
