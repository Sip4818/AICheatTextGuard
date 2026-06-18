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


## Plan: Implement Test Cases for the Entire Project

### Strategy

- Each source module gets a corresponding test file in `tests/` following the existing pattern (`test_<module>.py`).
- Tests use **pytest** with **mocks** for I/O-heavy operations (GCS, Redis, model files, external APIs).
- Data validation tests use **small in-memory DataFrames** instead of real CSVs.
- Pipeline component tests mock their dependencies and verify artifact outputs, error handling, and edge cases.
- No real model files, GCS buckets, or Redis instances are required to run the tests.

### Test Files and Coverage

#### 1. `tests/test_basic.py` — Placeholder → Utils & Exceptions
- **Source**: `src/utils/common.py`, `src/utils/exception.py`, `src/utils/logger.py`
- **Test cases**:
  - `test_assert_file_exists` — file exists / does not exist
  - `test_is_yaml_content_empty` — empty / non-empty YAML
  - `test_read_yaml` — valid YAML, missing file, empty file, invalid structure, YAML syntax error
  - `test_write_yaml` — writes correct content, creates parent dirs
  - `test_create_dir` — creates dir, handles existing dir
  - `test_upload_to_gcs` — mocks `storage.Client`, tests upload success / file not found / overwrite skip
  - `test_download_from_gcs` — mocks `storage.Client`, tests download success / bucket not found / blob not found
  - `test_read_csv_file` — valid CSV, non-CSV path, missing file
  - `test_save_csv` — saves correctly, creates parent dirs
  - `test_read_object` — valid pkl, missing file
  - `test_save_object` — saves correctly, creates parent dirs
  - `test_log_file_size` — valid file, missing file
  - `test_read_numpy` — valid `.npy`, non-`.npy` path
  - `test_save_numpy` — saves correctly, creates parent dirs
  - `test_extract_params` — basic extraction, non-primitive values converted to float
  - `test_to_dict` — nested ConfigBox, plain dict, list, primitive
  - `test_AITextException` — with traceback / without traceback, `__str__` format
  - `test_logger` — logger is configured, has file and console handlers (smoke test)

#### 2. `tests/test_config.py` — Config Entities & ConfigurationManager
- **Source**: `config/configuration.py`, `config/training_pipeline_config.py`, `src/entity/config_entity.py`, `src/entity/model_trainer_tuning_entity.py`, `src/entity/model_trainer_final_params_entity.py`
- *(Already partially implemented with tests for all config entity retrieval methods)*
- **Additional cases**:
  - `test_config_init_missing_yaml` — `ConfigurationManager` raises on missing config YAML
  - `test_config_init_empty_yaml` — `ConfigurationManager` handles empty params YAML (writes defaults)
  - `test_get_data_ingestion_config_creates_dirs` — verifies root dir is created
  - `test_get_model_trainer_tuning_config` — returns correct `ModelTrainerTuningConfig` with nested spaces
  - `test_get_model_trainer_final_params_config` — returns correct `ModelTrainerFinalParamsConfig`
  - `test_training_pipeline_config` — `TrainingPipelineConfig` has correct defaults
  - Config entity dataclass tests: each entity creates correctly with valid/invalid inputs

#### 3. `tests/test_feature_engineering.py` — Feature Generation
- **Source**: `src/feature_generation/basic_features.py`, `src/feature_generation/transformer_embedding.py`
- **Test cases for `BasicFeatureGenerator`**:
  - `test_get_cleaned_string` — lowercasing, special char removal, whitespace normalization, non-string input returns `""`
  - `test_get_punctuation_count` — various punctuation patterns
  - `test_get_avg_word_length` — normal text, empty string returns `0.0`, single word
  - `test_get_capital_words_count` — uppercase words, mixed case, no uppercase
  - `test_get_stopword_count` — multiple stopwords, no stopwords
  - `test_transform` — full transform on a sample DataFrame with `text` column, verifies all feature columns exist
  - `test_transform_missing_text_column` — raises `KeyError`
  - `test_fit_returns_self` — `fit()` returns `self`
- **Test cases for `EmbeddingFeaturesGenerator`**:
  - `test_init` — stores model_path and model_name defaults
  - `test_fit` — mocks `SentenceTransformer` to verify model is loaded with correct args
  - `test_transform` — mocks model.encode, uses DataFrame with `cleaned_text` and numeric columns, verifies output shape
  - `test_transform_missing_cleaned_text` — raises `KeyError`

#### 4. `tests/test_preprocessing.py` — Data Transformation
- **Source**: `src/components/data_transformation.py`
- **Test cases**:
  - `test_split_data` — splits X and y correctly from a sample DataFrame
  - `test_split_data_missing_target` — raises error when target column missing
  - `test_initiate_data_transformation` — mocks `read_csv_file`, `BasicFeatureGenerator`, `EmbeddingFeaturesGenerator`, verifies pipeline fit_transform is called, numpy array and pipeline object are saved
  - `test_initiate_data_transformation_file_not_found` — handles missing CSV gracefully
  - `test_identity_func` — returns input unchanged

#### 5. `tests/test_validation.py` — Data Validation
- **Source**: `src/components/data_validation.py`
- **Test cases**:
  - `test_validate_schema` — all required columns present / missing columns
  - `test_validate_missing_values` — no missing values, has NaN, empty strings in `text` column
  - `test_validate_allowed_values` — valid values, invalid values, truncation to 10 reported indices
  - `test_validate_dtype` — matching dtypes, mismatched dtypes
  - `test_validate_data_drift_and_leakage` — no overlap, some overlap, overlap with NaN handling
  - `test_initiate_data_validation` — mocks `read_csv_file`, verifies report JSON is written with all sections, returns `DataValidationArtifact`
  - `test_initiate_data_validation_raises` — handles exceptions during validation

#### 6. `tests/test_pipeline.py` — Pipeline Components
- **Source**: `src/components/data_ingestion.py`, `src/components/model_trainer.py`, `src/components/model_evaluation.py`, `src/pipeline/training/training_pipeline.py`, `src/pipeline/prediction/prediction_pipeline.py`
- **`DataIngestion`**:
  - `test_download_data` — mocks `download_from_gcs`, verifies call with correct args
  - `test_download_data_raises` — mocks failure in `download_from_gcs`
  - `test_initiate_data_ingestion` — mocks download, file assertions, CSV read, train/test split, CSV saves, returns `DataIngestionArtifact`
  - `test_initiate_data_ingestion_no_download` — when `to_download_data=False`
  - `test_initiate_data_ingestion_missing_data` — raises when data file missing after download
- **`ModelTrainer`**:
  - `test_init` — mocks `read_object` for preprocessor, verifies `object_storage`
  - `test_generate_oof_xgb_preds` — mocks StratifiedKFold, XGBClassifier fit/predict, numpy saves
  - `test_generate_oof_lr_preds` — mocks StratifiedKFold, Pipeline fit/predict, numpy saves
  - `test_log_metrics` — verifies AUC, precision, recall, F1 calculation
  - `test_train` — mocks and verifies all model saves and object_storage updates
  - `test_initiate_model_training` — full flow with tuning enabled / disabled, verifies final stacked model save
  - `test_initiate_model_training_tuning_enabled` — mocks Tuner, verifies params YAML write
- **`ModelEvaluation`**:
  - `test_split_data` — splits X/y correctly
  - `test_write_report` — writes JSON correctly
  - `test_push_model_to_gcs` — mocks `upload_to_gcs`
  - `test_initiate_model_evaluation` — mocks model read, test data read, computes metrics, saves JSON and plot
  - `test_initiate_model_evaluation_push_model` — verifies GCS push when config flag is True
- **`TrainingPipeline`**:
  - `test_full_pipeline_orchestration` — mocks all 4 stage methods, verifies they are called in order
  - `test_pipeline_initialization` — `TrainingPipeline.__init__` creates `ConfigurationManager`
- **`PredictionPipeline`**:
  - `test_init` — mocks `read_object`, loads model
  - `test_predict` — mocks model's `predict_proba`, returns expected array

#### 7. `tests/test_model.py` — Model & Tuning
- **Source**: `src/tuning/tuner.py`, `src/tuning/_objective.py`, `src/tuning/_search_spaces.py`, `src/tuning/_utils.py`
- **`SearchSpaces`**:
  - `test_build_space_float` — suggests float with/without log
  - `test_build_space_int` — suggests int
  - `test_build_space_categorical` — suggests categorical
  - `test_build_space_unsupported_type` — raises `ValueError`
- **`Objective`**:
  - `test_call` — mocks `_build_model` and `_evaluate`, returns AUC score
  - `test_build_model_lr` — creates `LogisticRegression` with params
  - `test_build_model_xgb` — creates `XGBClassifier` with params
  - `test_build_model_unknown` — raises `ValueError`
  - `test_evaluate` — mocks `cross_val_score`, returns mean AUC
- **`Tuner`**:
  - `test_tune_xgb_level1` — mocks `Objective`, `optuna.create_study`, mlflow, verifies best params returned
  - `test_tune_lr_level1` — same for LR level 1
  - `test_tune_lr_level2` — same for LR level 2, verifies ValueError when X_meta is None
  - `test_setup_mlflow` — mocks mlflow experiment creation
- **`_utils`**:
  - `test_set_seed` — seeds random and numpy
  - `test_save_study_best_params` — writes JSON correctly
  - `test_save_study` — saves trials DataFrame as CSV
  - `test_print_study_results` — smoke test (print capture)
  - `test_namespace_params` — prefixes keys correctly

#### 8. `tests/test_api.py` — FastAPI Backend
- **Source**: `app.py`
- *(Already has 3 tests: valid request, too short, too long)*
- **Additional cases**:
  - `test_predict_redis_cache_hit` — mocks Redis `get`, verifies cached response returned
  - `test_predict_redis_not_available` — mocks `r = None` scenario
  - `test_predict_redis_get_error` — mocks RedisError on get, falls through to prediction
  - `test_predict_redis_set_error` — mocks RedisError on set, still returns prediction
  - `test_cache_key` — `PredictRequest.cache_key()` returns deterministic SHA-256 hash
  - `test_predict_invalid_payload` — missing `text` key, non-string text
  - `test_predict_predictor_raises` — mocks `predictor.predict` to raise, verifies 500

#### 9. `tests/test_security.py` — Security & Input Validation
- **Source**: `app.py` (Pydantic model), `ui/streamlit_app.py`
- **Test cases**:
  - `test_predict_request_min_length` — Pydantic model validates `text` >= 250 chars
  - `test_predict_request_max_length` — Pydantic model validates `text` <= 5000 chars
  - `test_predict_request_exact_boundaries` — exactly 250 chars, exactly 5000 chars
  - `test_streamlit_backend_url_required` — verifies `ValueError` when `BACKEND_URL` not set

### Running the Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_validation.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov=app --cov-report=term-missing
```

### Implementation Order (Recommended)

1. **`tests/test_basic.py`** — Utils & exceptions (foundational, no complex dependencies)
2. **`tests/test_config.py`** — Config entities (already partially done, add remaining cases)
3. **`tests/test_validation.py`** — Data validation (pure logic, DataFrame-based)
4. **`tests/test_feature_engineering.py`** — Feature generation (standalone transformers)
5. **`tests/test_preprocessing.py`** — Data transformation (builds on feature gen)
6. **`tests/test_model.py`** — Model & tuning (model logic, search spaces)
7. **`tests/test_pipeline.py`** — Pipeline components (heavily mocked)
8. **`tests/test_api.py`** — API (builds on prediction pipeline mocks)
9. **`tests/test_security.py`** — Security & input validation
