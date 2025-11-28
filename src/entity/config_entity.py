from dataclasses import dataclass
from typing import List, Dict

from src.components import data_transformation
@dataclass
class DataIngestionConfig:
    cloud_train_path: str
    cloud_test_path: str
    bucket_name: str
    local_data_dir: str
    local_train_path: str
    local_test_path: str

@dataclass
class DataValidationConfig:
    data_validation_report_artifact_path: str
    data_validation_report_path: str
    required_columns: List
    columns_dtype: Dict
    allowed_values: Dict

@dataclass
class DataTransformationConfig:
    validated_data_train_path: str
    validated_data_test_path: str
    transformed_train_data_path: str
    transformed_test_data_path: str
    transformed_val_data_path: str
    data_transformation_object_path: str
    model_dir: str
    target_column_name: str
    test_split_ratio: int

