from config.training_pipeline_config import TrainingPipelineConfig
from src.components import data_transformation
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig, DataTransformationConfig
from src.constants.constants import config_yaml_file_path,schema_yaml_file_path
from src.utils.common import read_yaml
from src.utils.logger import logger
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
        logger.info("Data root directory created")
        os.makedirs(self.artifact_dir,exist_ok=True)
        logger.info("Artifact root directory created")

        self.data_ingestion_config=self.config.data_ingestion

        self.data_validation_config=self.config.data_validation

        self.data_transformation_config=self.config.data_transformation


    def get_data_ingestion_config(self)->DataIngestionConfig:
        os.makedirs(self.data_ingestion_config.local_data_dir,exist_ok=True)
        logger.info("Local raw data directory created")
        return DataIngestionConfig(
            cloud_train_path=self.data_ingestion_config.cloud_train_data_path,
            cloud_test_path=self.data_ingestion_config.cloud_test_data_path,
            bucket_name=self.data_ingestion_config.bucket_name,
            local_data_dir=self.data_ingestion_config.local_data_dir,
            local_train_path=self.data_ingestion_config.local_train_path,
            local_test_path=self.data_ingestion_config.local_test_path
        )
        

    def get_data_validation_config(self)->DataValidationConfig:
        data_validation_report_artifact_dir=os.path.join(self.artifact_dir,self.data_validation_config.data_validation_dir_name)
        os.makedirs(data_validation_report_artifact_dir,exist_ok=True)
        data_validation_report_artifact_path=os.path.join(data_validation_report_artifact_dir,self.data_validation_config.validated_data_report_file_name)
        os.makedirs(self.data_validation_config.data_validation_report_directory,exist_ok=True)
        data_validation_report_path=os.path.join(self.data_validation_config.data_validation_report_directory,self.data_validation_config.validated_data_report_file_name)
        return DataValidationConfig(
                raw_train_data_path=self.data_ingestion_config.local_train_path,
                raw_test_data_path=self.data_ingestion_config.local_test_path,
                data_validation_report_artifact_path=data_validation_report_artifact_path,
                data_validation_report_path=data_validation_report_path,
                required_columns=self.schema.required_columns,
                columns_dtype=self.schema.columns_dtype,
                allowed_values=self.schema.allowed_values
        )

    def get_data_transformation_config(self)-> DataTransformationConfig:

        os.makedirs(self.data_transformation_config.transformed_data_dir,exist_ok=True)
        logger.info(f"Transformed data directory created")


        os.makedirs(os.path.join(self.artifact_dir,
                                 self.data_transformation_config.artifact_objects_dir),exist_ok=True)
        logger.info(f"Artifact object directory created")

        os.makedirs(self.data_transformation_config.model_dir, exist_ok=True)
        logger.info("Models directory created")
        transfomed_train_data_path=os.path.join(self.data_transformation_config.transformed_data_dir,
                                                self.data_transformation_config.transformed_train_file_name)
        transfomed_test_data_path=os.path.join(self.data_transformation_config.transformed_data_dir,
                                                self.data_transformation_config.transformed_test_file_name)
        transfomed_val_data_path=os.path.join(self.data_transformation_config.transformed_data_dir,
                                                self.data_transformation_config.transformed_val_file_name)

        data_transformation_object_path=os.path.join(self.artifact_dir,
                                 self.data_transformation_config.artifact_objects_dir,
                                 self.data_transformation_config.data_transformation_object_name)
        return DataTransformationConfig(
                validated_data_train_path=self.data_ingestion_config.local_train_path,
                validated_data_test_path=self.data_ingestion_config.local_test_path,
                transformed_train_data_path=transfomed_train_data_path,
                transformed_test_data_path=transfomed_test_data_path,
                transformed_val_data_path=transfomed_val_data_path,
                data_transformation_object_path=data_transformation_object_path,
                model_dir=self.data_transformation_config.model_dir,
                target_column_name=self.schema.target_column_name,
                test_split_ratio=self.data_transformation_config.test_split_ratio

        )

