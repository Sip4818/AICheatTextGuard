# Pipeline config
data_root = "data"
artifacts_root = "artifact"
config_yaml_file_path = "config/config.yaml"

SEED = 42

schema_yaml_file_path = "data_schema/schema.yaml"

params_yaml_file_path = "params.yaml"

# Data Ingestion Config
bucket_name = "ai_text_guard_bucket"
test_file_name = "train.csv"
train_file_name = "test.csv"

# Validation


LR_KEYS = ["C", "penalty", "solver", "max_iter"]

XGB_KEYS = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
]

params_dict_format = {
    "model_trainer": {
        "level1": {
            "lr": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 100,
            },
            "xgb": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
        },
        "level2": {
            "lr": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 100,
            }
        },
    }
}
validation_report_tempelate = {
            "schema_check": {},
            "missing_values_check": {},
            "allowed_values_check": {},
            "dtype_check": {},
        }

final_model_path = "model/stacked_model.pkl"
