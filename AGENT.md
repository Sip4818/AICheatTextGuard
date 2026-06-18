# AGENT.md

Guidance for coding agents working in this repository.

## Project Overview

AITextGuard is a Python ML/MLOps project for detecting AI-generated text. It contains:

- A DVC training pipeline for ingestion, validation, transformation, training, and evaluation.
- A FastAPI inference backend in `app.py`.
- A Streamlit UI in `ui/streamlit_app.py`.
- Docker Compose services for backend, UI, Redis, and Prometheus.
- Model and data artifacts generated under `data/`, `artifact/`, `model/`, `metrics/`, `logs/`, `temp/`, and `dvc_plots/`.

## Important Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full training pipeline through DVC:

```bash
dvc repro
```

Run the imperative training pipeline:

```bash
python main.py
```

Run the FastAPI backend locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Run the Streamlit UI locally:

```bash
streamlit run ui/streamlit_app.py
```

Run the composed app:

```bash
docker compose -f compose.yml up
```

Run tests:

```bash
pytest
```

**Current status of project**: The current test coverage is minimal. Most test files (like `tests/test_basic.py`, `tests/test_security.py`, etc.) are empty placeholder files. We need to implement all the test cases to ensure reliability and correctness of the pipeline and application components.


## Key Files And Directories

- `app.py`: FastAPI prediction API. Loads `model/stacked_model.pkl` at import time and optionally uses Redis cache.
- `ui/streamlit_app.py`: Streamlit client. Reads backend URL from `BACKEND_URL`.
- `dvc.yaml`: Reproducible ML pipeline stages.
- `config/config.yaml`: Pipeline paths and stage settings.
- `params.yaml`: Tuned/final model hyperparameters.
- `src/components/`: Pipeline stage implementations.
- `src/pipeline/training/`: Thin stage entrypoints plus orchestration.
- `src/pipeline/prediction/prediction_pipeline.py`: Prediction wrapper.
- `src/feature_generation/`: Text feature and embedding generation.
- `src/model/stack_model.py`: Stacked model object used at inference.
- `Docker/`: Backend, UI, and monolithic Dockerfiles.
- `.github/workflows/cicd.yml`: Docker image build/push workflow with model download from GCS.

## Generated Or External Artifacts

Be careful with these paths:

- `data/`: raw and transformed datasets.
- `artifact/`: validation reports, preprocessors, intermediate model artifacts.
- `model/`: trained final model.
- `metrics/`: evaluation JSON and plots.
- `logs/`: runtime logs.
- `temp/`: downloaded or cached transformer model files.
- `dvc_plots/`: DVC-generated plots.
- `myvenv/`, `__pycache__/`, `*.pyc`: local environment/cache files.

Do not manually edit generated artifacts unless the user explicitly asks. Prefer regenerating them with DVC or the relevant pipeline command.

## Environment And Secrets

- `.env` exists locally and may contain secrets or machine-specific values. Do not print or commit secret values.
- GCS access is used by ingestion/model upload paths and CI. Local runs that touch GCS require valid Google Cloud credentials.
- The default final model path is `model/stacked_model.pkl`; API startup fails if the file is missing or incompatible.

## Known Sharp Edges

- `dvc.yaml` references `src.pipeline.training.model_evalute`; the filename is misspelled. Keep the current spelling unless renaming all references together.
- `config/config.yaml` has typos such as `metrices` and `auc_ruc`; changing them requires matching code changes.
- `src/constants/constants.py` appears to swap `test_file_name` and `train_file_name`; verify behavior before relying on those constants.
- `ui/streamlit_app.py` does not provide a fallback for missing `BACKEND_URL` and does not handle request exceptions.
- `app.py` hardcodes Redis host as `redis` instead of using the `REDIS_HOST` environment variable declared in Compose.
- `app.py` has an invalid `await r.get(key)` call on the standard synchronous `redis.Redis` client, which throws a `TypeError` at runtime.
- `app.py` performs a synchronous `r.set` call inside an async endpoint, which blocks the main event loop.
- Several files contain mojibake characters from encoding issues, especially `README.md`, `ui/streamlit_app.py`, and some log strings.
- [RESOLVED] `requirements.txt` included `dotenv==0.9.9` (updated to `python-dotenv==1.0.1`).
- Avoid broad refactors in pipeline code without adding tests, because artifacts, DVC dependencies, and serialized model compatibility are tightly coupled.

## Coding Guidelines

- **Always ask for explicit permission before pushing any changes to the remote repository.** Do not push without the user saying so.
- Before editing, creating, deleting, formatting, staging, committing, or pushing any file, ask the user for explicit permission. Do not change even a single line without approval.
- Keep edits scoped to the requested behavior.
- Preserve existing serialized model contracts unless retraining/regeneration is part of the task.
- Prefer explicit path/config changes over hardcoded values.
- Add focused tests for changed behavior, especially validation, feature generation, prediction request handling, and pipeline config parsing.
- Use structured YAML/JSON readers for config files; avoid ad hoc string edits.
- Do not remove user-generated artifacts or untracked files without explicit approval.
- Before pushing any changes, you MUST run `bash check.sh` (which runs ruff format, ruff lint, mypy, and pytest). This is the same set of checks that run in CI.

<!-- Test comment for permission validation -->

## Agent Changelog
- Removed unused commented-out assertion from `tests/test_config.py`.
- Implemented core test cases for the FastAPI backend in `tests/test_api.py` (including valid request and length constraints).

## Plan: Migrate to `uv`

### What is `uv`?
`uv` is a fast Python package manager (pip/pip-tools alternative) by Astral (the Ruff authors). It resolves and installs dependencies much faster than pip and supports `uv.lock` for reproducible builds.

### Migration Steps

1. **Install `uv`** globally (e.g., `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`).
2. **Replace `requirements.txt`** with a `pyproject.toml` (or keep `requirements.txt` and use `uv pip install`).
3. **Generate `uv.lock`** via `uv lock` for pinned, reproducible dependencies.
4. **Update CI/CD**:
   - In `.github/workflows/cicd.yml`, replace `pip install` with `uv pip install` or use `uv sync`.
   - Add caching for `uv.lock` and `~/.cache/uv`.
5. **Update Dockerfiles** (`Docker/`) to use `uv` for faster builds.
6. **Update local development commands** in `AGENT.md` and any contributing docs.
7. **Validate** that `dvc repro`, `pytest`, and `ruff`/`mypy` still work after the migration.

### Key Commands

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run a command in the uv-managed venv
uv run python main.py

# Add a package
uv add <package>

# Lock dependencies
uv lock
```

### Status
- [ ] Install `uv` and verify
- [ ] Create/replace dependency definition (e.g., `pyproject.toml`)
- [ ] Generate `uv.lock`
- [ ] Update CI/CD workflow
- [ ] Update Dockerfiles
- [ ] Update docs and commands
- [ ] Full validation (dvc repro, pytest, ruff, mypy)

