from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DataIngestionConfig:
    cloud_data_path: str
    bucket_name: str
    local_data_path: str
    local_train_path: str
    local_test_path: str
    test_split_size: int


@dataclass
class DataValidationConfig:
    raw_train_data_path: str
    raw_test_data_path: str
    data_validation_report_path: str
    required_columns: List
    columns_dtype: Dict
    allowed_values: Dict


@dataclass
class DataTransformationConfig:
    validated_data_train_path: str
    transformed_train_data_path: str
    data_transformation_object_path: str
    temp_model_dir: str
    target_column_name: str
    test_split_size: int


@dataclass
class ModelTrainerConfig:
    transformed_train_data_path: str
    preprocessing_object_path: str
    lr_level1_model_path: str
    xgb_level1_model_path: str
    meta_lr_path: str
    enable_tuning: bool
    final_model_path: str
    lr_level1_oof_predictions_path: str
    xgb_level1_oof_predictions_path: str
    folds: int


@dataclass
class ModelEvaluationConfig:
    raw_test_data_path: str
    final_model_path: str
    target_column_name: str
    model_evaluation_artifact_file_path: str
    metrices: list[str]
    push_model_to_gcs: bool
    gcs_bucket_name: str

