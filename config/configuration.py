import logging
from numpy import True_
from config.training_pipeline_config import TrainingPipelineConfig
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig)

from src.entity.model_trainer_tuning_entity import (
    LRSpace, XGBSpace, 
    Level1TuningConfig, Level2TuningConfig,
    ModelTrainerTuningConfig
)
from src.entity.model_trainer_final_params_entity import (
    LRFinalParams, XGBFinalParams,
    Level1FinalParams, Level2FinalParams,
    ModelTrainerFinalParamsConfig
)

from src.constants.constants import (
     config_yaml_file_path,
     schema_yaml_file_path,
     params_yaml_file_path
     )

from src.utils.common import read_yaml
from src.utils.logger import logger
import os

class ConfigurationManager:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig,
                  yaml_file_path: str = config_yaml_file_path,
                  schema_file_path: str = schema_yaml_file_path,
                  params_file_path: str = params_yaml_file_path ):
        
        self.data_root=training_pipeline_config.data_root
        self.artifact_dir=training_pipeline_config.artifact_dir
        self.config=read_yaml(yaml_file_path)
        self.schema=read_yaml(schema_file_path)
        self.params= read_yaml(params_file_path)

        os.makedirs(self.data_root,exist_ok=True)
        logger.info("Data root directory created")
        os.makedirs(self.artifact_dir,exist_ok=True)
        logger.info("Artifact root directory created")

        self.data_ingestion_config=self.config.data_ingestion

        self.data_validation_config=self.config.data_validation

        self.data_transformation_config=self.config.data_transformation

        self.model_trainer_config = self.config.model_trainer

        self.tuning_cfg = self.config.model_trainer.tuning

        self.final_params_cfg = self.params.model_trainer

    def get_data_ingestion_config(self)->DataIngestionConfig:

        os.makedirs(self.data_ingestion_config.root_dir,exist_ok=True)
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

        os.makedirs(self.data_validation_config.root_dir,exist_ok=True)
        logging.info("Data validation root dir created")


        return DataValidationConfig(
                raw_train_data_path=self.data_ingestion_config.local_train_path,
                raw_test_data_path=self.data_ingestion_config.local_test_path,
                data_validation_report_path=self.data_validation_config.validated_data_report_file_path,
                required_columns=self.schema.required_columns,
                columns_dtype=self.schema.columns_dtype,
                allowed_values=self.schema.allowed_values
        )

    def get_data_transformation_config(self)-> DataTransformationConfig:

        os.makedirs(self.data_transformation_config.data_root_dir,exist_ok=True)
        logger.info(f"Transformed data root directory created")


        os.makedirs(self.data_transformation_config.artifact_root_dir, exist_ok=True)
        logger.info(f"Data transformation artifact directory created")

        os.makedirs(self.data_transformation_config.temp_model_dir, exist_ok=True)
        logger.info("Temporery models directory created")

        return DataTransformationConfig(
                validated_data_train_path=self.data_ingestion_config.local_train_path,
                validated_data_test_path=self.data_ingestion_config.local_test_path,
                transformed_train_data_path=self.data_transformation_config.transformed_train_data_path,
                transformed_test_data_path=self.data_transformation_config.transformed_test_data_path,
                transformed_val_data_path=self.data_transformation_config.transformed_val_data_path,
                data_transformation_object_path=self.data_transformation_config.data_transformation_object_path,
                temp_model_dir=self.data_transformation_config.temp_model_dir,
                target_column_name=self.schema.target_column_name

        )

    def get_model_trainer_config(self)-> ModelTrainerConfig:

        os.makedirs(self.model_trainer_config.root_dir,exist_ok=True)
        logger.info("Model trainer root directory created")

        os.makedirs(os.path.dirname(self.model_trainer_config.lr_level1_oof_predictions_path),exist_ok=True)
        logger.info("Model trainer OOF level 1 predictions directory created")


        return ModelTrainerConfig(
            transformed_train_data_path= self.transfomed_train_data_path,
            transformed_test_data_path=self.transfomed_test_data_path,
            lr_level_1_model_path= self.model_trainer_config.lr_level_1_path,
            xgb_level_1_model_path=self.model_trainer_config.xgb_level_1_path,
            meta_lr_path=self.model_trainer_config.meta_lr_path,
            final_mode_path=self.model_trainer_config.final_mode_path,
            lr_level1_oof_predictions_path=self.model_trainer_config.lr_level1_oof_predictions_path,
            xgb_level1_oof_predictions_path=self.model_trainer_config.xgb_level1_oof_predictions_path

        )

    def get_model_trainer_tuning_config(self)-> ModelTrainerTuningConfig:
        lr_space = LRSpace(
                **self.model_trainer_config.tuning.level1.lr
        )
        xgb_space = XGBSpace(
                **self.model_trainer_config.tuning.level1.xgb
        )

        Level1TuningConfig(
            lr=lr_space,
            xgb=xgb_space
        )
        Level2TuningConfig(
            lr=lr_space
        )

        return ModelTrainerTuningConfig(
            level1= Level1TuningConfig,
            level2=Level2TuningConfig
        )
    
    def get_model_trainer_final_params_config(self)->ModelTrainerConfig:

        lr_final_params=LRFinalParams(
            **self.params.model_trainer.level1.lr
        )

        xgb_final_params=XGBFinalParams(
            **self.params.model_trainer.level1.xgb
        )
        
        level1_final_params=Level1FinalParams(
            lr=lr_final_params,
            xgb=xgb_final_params
        )

        level2_final_params=Level2FinalParams(
            **self.params.model_trainer.level2.lr
        )

        return ModelTrainerFinalParamsConfig(
            folds= self.params.model_trainer.train.folds,
            level1=level1_final_params,
            level2=level2_final_params
        )
            