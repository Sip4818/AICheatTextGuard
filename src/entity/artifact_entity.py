from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    local_train_file_path: str
    local_test_file_path: str

@dataclass
class DataValidationArtifact:
    data_validation_report_artifact_path: str
    data_validation_report_path: str

@dataclass
class DataTransformationArtifact:
    data_transformation_object_path: str
    transformed_data_dir: str