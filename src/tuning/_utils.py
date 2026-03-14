import json
import os
import numpy as np
import random

import optuna


# ---------------------------------------------------------
# Set seed for reproducibility
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------
# Save Optuna study best parameters as JSON
# ---------------------------------------------------------
def save_study_best_params(study: optuna.Study, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(study.best_params, f, indent=4)


# ---------------------------------------------------------
# Save entire Optuna study for later analysis
# ---------------------------------------------------------
def save_study(study: optuna.Study, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    study.trials_dataframe().to_csv(filepath, index=False)


# ---------------------------------------------------------
# Pretty print study results
# ---------------------------------------------------------
def print_study_results(study: optuna.Study):
    print("=" * 60)
    print("Best Score :", study.best_value)
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)


# ---------------------------------------------------------
# Add prefix to parameter names (optional)
# e.g. ("lr", {"C": 0.1}) → {"lr_C": 0.1}
# ---------------------------------------------------------
def namespace_params(prefix: str, params: dict):
    return {f"{prefix}_{k}": v for k, v in params.items()}
