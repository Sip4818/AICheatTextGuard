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
- If a commit is authored or assisted by AI, include `[AI assistance]` in the commit message (e.g., `feat: implement Redis caching tests [AI assistance]`).

<!-- Test comment for permission validation -->

## Agent Changelog
- Removed unused commented-out assertion from `tests/test_config.py`.
- Implemented core test cases for the FastAPI backend in `tests/test_api.py` (including valid request and length constraints).
- Added detailed phased plan for backend tests under `## Plan: Backend Tests (Phased)`.
- Added commit message convention: include `[AI assistance]` for AI-assisted commits.

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
## Plan: Backend Tests (Phased)

### Scope

This plan covers the **backend** of AITextGuard \— the FastAPI application, prediction pipeline, Pydantic input validation, utility functions used at inference time, and the Streamlit UI client.

| Component | File(s) | Purpose |
|---|---|---|
| FastAPI App | `app.py` | `/predict` endpoint, Redis caching, Prometheus metrics |
| Prediction Pipeline | `src/pipeline/prediction/prediction_pipeline.py` | Loads model, runs `predict_proba` |
| Pydantic Validation | `app.py` (`PredictRequest`) | Text length constraints, `cache_key()` |
| Backend Utilities | `src/utils/common.py` (subset) | `read_object`, `assert_file_exists`, `log_file_size` |
| Streamlit UI | `ui/streamlit_app.py` | Client that calls the backend API |

---

### Phase 1: Foundation \— Shared Test Infrastructure

**Goal**: Create reusable pytest fixtures, mocks, and helpers so all later phases are DRY and consistent.

**File**: `tests/conftest.py` (create)

**Fixtures**:

| Fixture | Purpose |
|---|---|
| `valid_text` | String ≥ 250 chars that passes Pydantic validation |
| `mock_predictor` | Patches `PredictionPipeline.predict` to return controlled probabilities |
| `mock_redis_available` | Patches `app.r` to a mock Redis client |
| `mock_redis_unavailable` | Patches `app.r = None` |
| `client` | `TestClient(app)' with model-loading suppressed at import |
| `sample_df` | Small in-memory DataFrame for prediction pipeline tests |
| `mock_model_object` | A mock with `predict_proba` returning known values |

---

### Phase 2: Prediction Pipeline Tests

**Goal**: Test the `PredictionPipeline` class that sits behind the API endpoint.

**File**: `tests/test_pipeline.py` (implement \— currently empty placeholder)

| Test | What it verifies |
|---|---|
| `test_prediction_pipeline_init` | `__init__` calls `read_object` with correct path |
| `test_prediction_pipeline_init_file_not_found` | Raises when model file is missing |
| `test_prediction_pipeline_predict` | `predict()` calls `model.predict_proba()` and returns expected array |
| `test_prediction_pipeline_predict_empty_df` | Handles empty DataFrame gracefully |
| `test_prediction_pipeline_predict_model_raises` | Propagates exception when model predict fails |

**Mocks needed**: `src.utils.common.read_object`

---

### Phase 3: Core API Tests

**Goal**: Extend the existing 3 tests in `tests/test_api.py` to full coverage of the `/predict` endpoint.

**Existing tests** (already present):
- `test_predict_valid_request`
- `test_predict_text_too_short`
- `test_predict_text_too_long`

**New tests to add**:

| Test | What it verifies |
|---|---|
| `test_predict_redis_cache_hit` | Mock Redis `get` returns cached JSON; endpoint returns it without calling predictor |
| `test_predict_redis_cache_miss` | Redis `get` returns None; predictor is called and result returned |
| `test_predict_redis_not_available` | `r = None`; skips Redis entirely, predictor is called |
| `test_predict_redis_get_error` | `redis.exceptions.RedisError` on get caught; falls through to prediction |
| `test_predict_redis_set_error` | `redis.exceptions.RedisError` on set caught; still returns prediction |
| `test_predict_invalid_payload_missing_text` | POST with `{}` returns 422 |
| `test_predict_invalid_payload_non_string` | POST with `{"text": 123}` returns 422 |
| `test_predict_predictor_raises` | When `predictor.predict` raises, endpoint returns 500 |
| `test_predict_cache_key_deterministic` | Same input produces same cache key |
| `test_predict_cache_key_different_inputs` | Different inputs produce different cache keys |
| `test_predict_response_structure` | Valid response has exactly `{"probability": float}` |
| `test_predict_probability_range` | Returned probability is between 0.0 and 1.0 |

**Mocks needed**: `app.predictor.predict`, `app.r.get`, `app.r.set`, `app.r.ping`

---

### Phase 4: Security & Input Validation Tests

**Goal**: Test Pydantic request validation and backend-related security concerns.

**File**: `tests/test_security.py` (implement \— currently empty placeholder)

| Test | What it verifies |
|---|---|
| `test_predict_request_min_length` | Text < 250 chars raises validation error |
| `test_predict_request_max_length` | Text > 5000 chars raises validation error |
| `test_predict_request_exact_min_boundary` | Exactly 250 chars is valid |
| `test_predict_request_exact_max_boundary` | Exactly 5000 chars is valid |
| `test_predict_request_empty_string` | Empty string raises validation error |
| `test_predict_request_whitespace_only` | Whitespace string of valid length behavior |
| `test_predict_request_cache_key_format` | `cache_key()` returns string starting with "Predict:" and is valid SHA-256 hex |
| `test_predict_request_cache_key_length` | Full cache key length is predictable |
| `test_streamlit_backend_url_required` | When `BACKEND_URL` is unset, raises `ValueError` |
| `test_streamlit_backend_url_set` | When `BACKEND_URL` is set, `API_URL` matches it |

---

### Phase 5: Backend Utility Tests

**Goal**: Test utility functions that the backend depends on.

**File**: `tests/test_basic.py` (implement \— currently empty placeholder)

| Test | What it verifies |
|---|---|
| `test_read_object_valid` | Reads valid `.pkl` and returns loaded object |
| `test_read_object_missing_file` | Raises `AITextException` when file does not exist |
| `test_read_object_corrupted_file` | Handles corrupted pickle gracefully |
| `test_read_object_invalid_path` | Non-existent path raises |
| `test_assert_file_exists_valid` | No exception for existing file |
| `test_assert_file_exists_missing` | Raises `AITextException` for missing file |
| `test_log_file_size_valid` | Returns expected size for known file |
| `test_log_file_size_missing` | Raises for missing file |
| `test_to_dict_with_configbox` | Converts nested `ConfigBox` to plain dict |
| `test_to_dict_with_plain_dict` | Passes through plain dict unchanged |

---

### Phase 6: Edge Cases & Integration Smoke Tests

**Goal**: Test non-happy-path scenarios and validate that components wire together correctly.

**File**: `tests/test_api.py` (extend further)

| Test | What it verifies |
|---|---|
| `test_predict_large_payload_near_limit` | Text at exactly 5000 chars accepted and processed |
| `test_predict_special_characters` | Unicode, emoji, newlines handled |
| `test_predict_redis_cache_expiry` | Verify `ex=600` is passed to `r.set` |
| `test_predict_redis_cache_key_consistency` | Same text from different requests produces same key |
| `test_app_startup_redis_fail` | When Redis unreachable, `app.r` is `None`, app still works |

---

### Phase 7: Streamlit UI Tests

**Goal**: Test the UI component's integration with the backend.

**File**: `tests/test_ui.py` (create)

| Test | What it verifies |
|---|---|
| `test_streamlit_missing_backend_url_raises` | Raises `ValueError` when env var not set |
| `test_streamlit_backend_url_from_env` | Reads `BACKEND_URL` correctly |
| `test_streamlit_page_config` | Page configured with correct title |
| `test_streamlit_predict_success` | Mocks `requests.post` returning 200 with probability |
| `test_streamlit_predict_failure` | Mocks `requests.post` returning non-200 |

---

### Mock Strategy

| External Dependency | Mock Strategy |
|---|---|
| `model/stacked_model.pkl` | Patch `src.utils.common.read_object` → mock with `predict_proba` |
| Redis server | Patch `app.r` with `unittest.mock.MagicMock` (get, set, ping) |
| Google Cloud Storage | Not needed for backend tests |
| Sentence Transformers | Not needed for backend tests |
| Prometheus | `Instrumentator` called at module level — patch/suppress during import |

### Files to Create / Modify

| File | Action | Phase |
|---|---|---|
| `tests/conftest.py` | **Create** | 1 |
| `tests/test_pipeline.py` | **Implement** (was empty) | 2 |
| `tests/test_api.py` | **Extend** (3 → 15+ tests) | 3, 6 |
| `tests/test_security.py` | **Implement** (was empty) | 4 |
| `tests/test_basic.py` | **Implement** (was empty) | 5 |
| `tests/test_ui.py` | **Create** | 7 |

### Recommended Execution Order

```
Phase 1: conftest.py          ─┐
                               │
Phase 2: Prediction Pipeline  ─┤  (no Redis or FastAPI needed)
                               │
Phase 5: Backend Utilities    ─┘  (pure logic, easy to mock)
                               │
Phase 3: Core API Tests       ─┤  (extends existing test_api.py)
                               │
Phase 4: Security Tests       ─┤  (Pydantic + Streamlit)
                               │
Phase 6: Edge Cases           ─┤  (depends on Phase 1 + 3)
                               │
Phase 7: Streamlit UI Tests   ─┘  (can be done in parallel with 6)
```

## Inconsistencies, Spelling Mistakes & Issues Found

### Spelling / Typos

| # | File | Issue | Should Be |
|---|------|-------|-----------|
| 1 | `dvc.yaml` (line 54) | Stage cmd `model_evalute` | `model_evaluate` |
| 2 | `src/pipeline/training/model_evalute.py` | Filename misspelled `model_evalute` | `model_evaluate` |
| 3 | `config/config.yaml` (line 69) | `metrices` (field name) | `metrics` |
| 4 | `config/config.yaml` (line 69) | `auc_ruc` (value in metrices list) | `auc_roc` |
| 5 | `src/entity/config_entity.py` (line 60) | `metrices` (dataclass field) | `metrics` |
| 6 | `config/configuration.py` (line 212) | `metrices=` (parameter passing) | `metrics=` |
| 7 | `src/constants/constants.py` (line 67) | `validation_report_tempelate` | `validation_report_template` |
| 8 | `src/utils/common.py` (line 207) | Docstring `\"\"\"Loggin file size\"\"\"` | `\"\"\"Logging file size\"\"\"` |
| 9 | `config/configuration.py` (line 203) | log message `\"Model evalutaion root artifact\"` | `\"Model evaluation root artifact\"` |
| 10 | `src/tuning/tuner.py` (line 14) | Experiment name `My_Staking_Model_Project` | `My_Stacking_Model_Project` |
| 11 | `src/tuning/tuner.py` (line 55) | MLflow run name `XGB_Level1_Tuning` (inconsistent casing with `Lr_Level1_Tuning`) | `XGB_Level1_Tuning` (acceptable but inconsistent with `Lr_` prefix) |
| 12 | `src/tuning/tuner.py` (line 82, 110) | MLflow run names `Lr_Level1_Tuning` and `lr_Level2_Tuning` (inconsistent case) | Use consistent casing e.g. `LR_Level1_Tuning`, `LR_Level2_Tuning` |

### Naming / Consistency Issues

| # | File | Issue |
|---|------|-------|
| 13 | `src/constants/constants.py` (lines 16-17) | `test_file_name = \"train.csv\"` and `train_file_name = \"test.csv\"` — values appear swapped |
| 14 | `config/config.yaml` (line 16) | Validation report path uses `.txt` extension (`data_validation_report.txt`) but code writes JSON content |
| 15 | `config/config.yaml` (line 16) | Config key `validated_data_report_file_path` vs entity field name `data_validation_report_path` — inconsistent naming |
| 16 | `config/config.yaml` (line 71) | Config key `model_evaluation_file_path` vs entity field `model_evaluation_artifact_file_path` — inconsistent naming |
| 17 | `config/config.yaml` (line 72) | Plot file named `confusion_metrics.png` but it is a confusion matrix plot — should be `confusion_matrix.png` |
| 18 | `ui/streamlit_app.py` (line 5) | Variable named `API_URL` but reads env var `BACKEND_URL` — inconsistent naming |
| 19 | `ui/streamlit_app.py` (line 9) | Error message says `API_URL` but env var is `BACKEND_URL` |

### Critical Bugs

| # | File | Issue |
|---|------|-------|
| 20 | `src/model/stack_model.py` | **File does not exist!** However `src/components/model_trainer.py` (line 30) imports `from src.model.stack_model import StackedModel`. The `src/model/` directory itself is missing. This will cause an `ImportError` at runtime when training runs. |
| 21 | `app.py` (line 21) | Redis host is hardcoded as `\"redis\"` instead of reading the `REDIS_HOST` env var declared in `compose.yml` |
| 22 | `app.py` (line 49) | `await r.get(key)` used on synchronous `redis.Redis` client — `TypeError` at runtime |
| 23 | `app.py` (line 66) | Synchronous `r.set()` call inside `async def predict` endpoint — blocks event loop |

### Dead / Unused Code

| # | File | Issue |
|---|------|-------|
| 24 | `src/tuning/_utils.py` | Contains 5 functions (`set_seed`, `save_study_best_params`, `save_study`, `print_study_results`, `namespace_params`) that are never imported or used anywhere in the project |
| 25 | `tests/test_basic.py` | Empty placeholder file |
| 26 | `tests/test_feature_engineering.py` | Empty placeholder file |
| 27 | `tests/test_model.py` | Empty placeholder file |
| 28 | `tests/test_pipeline.py` | Empty placeholder file |
| 29 | `tests/test_preprocessing.py` | Empty placeholder file |
| 30 | `tests/test_security.py` | Empty placeholder file |
| 31 | `tests/test_validation.py` | Empty placeholder file |

### Config / Structure Issues

| # | File | Issue |
|---|------|-------|
| 32 | `config/config.yaml` | Inconsistent indentation — some sections use different spacing patterns |
| 33 | `data_schema/schema.yaml` | Has `\r\n` (CRLF) line endings while other YAML files use `\n` (LF) — mixed line endings |
| 34 | `structure.txt` | Outdated project structure file — lists files that no longer exist |
| 35 | `AGENT.md` (line 72) | References `src/model/stack_model.py` as a key file but it does not exist |

### Test Code Issues

| # | File | Issue |
|---|------|-------|
| 36 | `tests/test_config.py` (line 234) | Uses `metrices` attribute (matches the misspelled entity field, so functionally consistent, but propagates the typo) |

### Minor Issues

| # | File | Issue |
|---|------|-------|
| 37 | `src/utils/exception.py` (line 30) | Commented-out `if __name__==\"__main__\":` block — dead debug code |
| 38 | `src/components/data_transformation.py` (line 42) | Commented-out line `# df_train = df_train.sample(50)` — leftover debug code |
| 39 | `app.py` (line 23) | `load_dotenv()` is missing from `app.py` — should be called so `.env` variables like `REDIS_HOST` would be loaded |
| 40 | `src/tuning/tuner.py` | `mlflow` is a dependency but not listed in `requirements.txt` (if used at runtime outside tuning) — verify it is installed |

### Notes

- Some misspellings (like `metrices` and `model_evalute`) are referenced across multiple files. Fixing them requires changing **all references** simultaneously, including `dvc.yaml`, config, entities, and test assertions.
- The `src/model/stack_model.py` missing file (issue #20) is the most critical — the training pipeline cannot run without it.
- Many of the `app.py` Redis issues (hardcoded host, async/sync mismatch) are already documented in the Known Sharp Edges section above.
