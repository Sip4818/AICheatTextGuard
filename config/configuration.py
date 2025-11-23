from config.training_pipeline_config import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.constants.constants import config_yaml_file_path,schema_yaml_file_path
from src.utils.common import read_yaml
import os

class ConfigurationManager:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig,
                  yaml_file_path: str=config_yaml_file_path,
                  schema_file_path: str =schema_yaml_file_path):
        self.data_root=training_pipeline_config.data_root
        self.artifact_dir=training_pipeline_config.artifact_dir
        self.timestamp=training_pipeline_config.timestamp
        self.config=read_yaml(yaml_file_path)
        self.schema=read_yaml(schema_file_path)

        os.makedirs(self.data_root,exist_ok=True)
        os.makedirs(self.artifact_dir,exist_ok=True)

    def get_data_ingestion_config(self)->DataIngestionConfig:
        self.data_ingestion_config=self.config.data_ingestion
        os.makedirs(self.data_ingestion_config.local_data_dir,exist_ok=True)
        return DataIngestionConfig(
            cloud_train_path=self.data_ingestion_config.cloud_train_data_path,
            cloud_test_path=self.data_ingestion_config.cloud_test_data_path,
            bucket_name=self.data_ingestion_config.bucket_name,
            local_data_dir=self.data_ingestion_config.local_data_dir,
            local_train_path=self.data_ingestion_config.local_train_path,
            local_test_path=self.data_ingestion_config.local_test_path
        )
        

    def get_data_validation_config(self)->DataValidationConfig:
        self.data_validation_config=self.config.data_validation
        data_validation_dir=os.path.join(self.artifact_dir,self.data_validation_config.data_validation_dir_name)
        os.makedirs(data_validation_dir,exist_ok=True)
        data_validation_report_path=os.path.join(data_validation_dir,self.data_validation_config.validated_data_report_file_name)

        return DataValidationConfig(
                raw_train_data_path=self.data_validation_config.validate_train_data_path,
                raw_test_data_path=self.data_validation_config.validate_test_data_path,
                data_validation_report_path=data_validation_report_path,
                required_columns=self.schema.required_columns,
                columns_dtype=self.schema.columns_dtype,
                allowed_values=self.schema.allowed_values
        )

