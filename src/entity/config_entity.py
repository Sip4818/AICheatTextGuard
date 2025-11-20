from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    cloud_train_path: str
    cloud_test_path: str
    bucket_name: str
    local_data_dir: str
    local_train_path: str
    local_test_path: str
   