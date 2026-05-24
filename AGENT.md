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

Current test coverage is minimal; `tests/test_basic.py` is empty.

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
- Several files contain mojibake characters from encoding issues, especially `README.md`, `ui/streamlit_app.py`, and some log strings.
- `requirements.txt` includes `dotenv==0.9.9`; most Python projects intend `python-dotenv`.
- Avoid broad refactors in pipeline code without adding tests, because artifacts, DVC dependencies, and serialized model compatibility are tightly coupled.

## Coding Guidelines

- Before editing, creating, deleting, formatting, staging, committing, or pushing any file, ask the user for explicit permission. Do not change even a single line without approval.
- Keep edits scoped to the requested behavior.
- Preserve existing serialized model contracts unless retraining/regeneration is part of the task.
- Prefer explicit path/config changes over hardcoded values.
- Add focused tests for changed behavior, especially validation, feature generation, prediction request handling, and pipeline config parsing.
- Use structured YAML/JSON readers for config files; avoid ad hoc string edits.
- Do not remove user-generated artifacts or untracked files without explicit approval.
