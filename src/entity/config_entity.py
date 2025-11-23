from dataclasses import dataclass
from typing import List, Dict
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
    raw_train_data_path: str
    raw_test_data_path: str
    data_validation_report_path: str
    required_columns: List
    columns_dtype: Dict
    allowed_values: Dict