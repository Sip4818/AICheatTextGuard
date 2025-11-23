from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    local_train_file_path: str
    local_test_file_path: str

@dataclass
class DataValidationArtifact:
    data_validation_report_path: str