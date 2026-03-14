import os

from config.training_pipeline_config import TrainingPipelineConfig
from src.utils.common import (
    read_yaml,
    create_dir,
    write_yaml,
    is_yaml_content_empty
)
from src.utils.exception import AITextException
from src.utils.logger import logger

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from src.entity.model_trainer_tuning_entity import (
    LRSpace,
    XGBSpace,
    Level1TuningConfig,
    Level2TuningConfig,
    ModelTrainerTuningConfig,
)
from src.entity.model_trainer_final_params_entity import (
    LRFinalParams,
    XGBFinalParams,
    Level1FinalParams,
    Level2FinalParams,
    ModelTrainerFinalParamsConfig,
)

from src.constants.constants import (
    config_yaml_file_path,
    schema_yaml_file_path,
    params_yaml_file_path,
    params_dict_format,
)


class ConfigurationManager:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig,
        yaml_file_path: str = config_yaml_file_path,
        schema_file_path: str = schema_yaml_file_path,
        params_file_path: str = params_yaml_file_path,
    ):
        # Read YAMLs
        self.config = read_yaml(yaml_file_path)
        self.schema = read_yaml(schema_file_path)
        if is_yaml_content_empty(params_file_path):
            write_yaml(params_dict_format, params_file_path)
        self.params = read_yaml(params_file_path)

        logger.info("Config, schema, and params YAML files loaded")

        # Store root paths
        self.data_root = training_pipeline_config.data_root
        self.artifact_dir = training_pipeline_config.artifact_dir

        create_dir(self.data_root, "Data root")
        create_dir(self.artifact_dir, "Artifact root")

        # Shortcut references
        self.data_ingestion_cfg = self.config.data_ingestion
        self.data_validation_cfg = self.config.data_validation
        self.data_transformation_cfg = self.config.data_transformation
        self.model_trainer_cfg = self.config.model_trainer
        self.tuning_cfg = self.config.model_trainer.tuning
        self.final_params_cfg = self.params.model_trainer
        self.model_evaluation_cfg = self.config.model_evaluation

    # 1. Data Ingestion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            create_dir(self.data_ingestion_cfg.root_dir, "Data ingestion root")

            return DataIngestionConfig(
                cloud_data_path=self.data_ingestion_cfg.cloud_data_path,
                bucket_name=self.data_ingestion_cfg.bucket_name,
                local_data_path=self.data_ingestion_cfg.local_data_path,
                local_train_path=self.data_ingestion_cfg.local_train_path,
                local_test_path=self.data_ingestion_cfg.local_test_path,
                test_split_size=self.data_ingestion_cfg.test_split_size,
                target_column_name=self.schema.target_column_name,
                required_columns= self.schema.required_columns,
                to_download_data= self.data_ingestion_cfg.to_download_data
            )
        except Exception as e:
            logger.error("Failed to build DataIngestionConfig")
            raise AITextException(e)

    # 2. Data Validation
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            create_dir(self.data_validation_cfg.root_dir, "Data validation")

            return DataValidationConfig(
                raw_train_data_path=self.data_ingestion_cfg.local_train_path,
                raw_test_data_path=self.data_ingestion_cfg.local_test_path,
                data_validation_report_path=self.data_validation_cfg.validated_data_report_file_path,
                required_columns=self.schema.required_columns,
                columns_dtype=self.schema.columns_dtype,
                allowed_values=self.schema.allowed_values,

            )
        except Exception as e:
            logger.error("Failed to build DataValidationConfig")
            raise AITextException(e)

    # 3. Data Transformation
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            create_dir(
                self.data_transformation_cfg.data_root_dir, "Data transformation root"
            )
            create_dir(
                self.data_transformation_cfg.artifact_root_dir,
                "Transformation artifacts",
            )
            create_dir(self.data_transformation_cfg.temp_model_dir, "Temporary models")

            return DataTransformationConfig(
                validated_data_train_path=self.data_ingestion_cfg.local_train_path,
                transformed_train_data_path=self.data_transformation_cfg.transformed_train_data_path,
                data_transformation_object_path=self.data_transformation_cfg.data_transformation_object_path,
                temp_model_dir=self.data_transformation_cfg.temp_model_dir,
                target_column_name=self.schema.target_column_name,
                test_split_size=self.data_transformation_cfg.test_split_size,
                requred_columns=self.schema.required_columns
            )
        except Exception as e:
            logger.error("Failed to build DataTransformationConfig")
            raise AITextException(e)

    # 4. Model Trainer
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            create_dir(self.model_trainer_cfg.root_dir, "Model trainer root")

            create_dir(
                os.path.dirname(self.model_trainer_cfg.lr_level1_oof_predictions_path),
                "LR OOF predictions",
            )

            create_dir(
                os.path.dirname(self.model_trainer_cfg.final_model_path),
                "Final model path",
            )

            return ModelTrainerConfig(
                transformed_train_data_path=self.data_transformation_cfg.transformed_train_data_path,
                preprocessing_object_path=self.data_transformation_cfg.data_transformation_object_path,
                lr_level1_model_path=self.model_trainer_cfg.lr_level_1_path,
                xgb_level1_model_path=self.model_trainer_cfg.xgb_level_1_path,
                meta_lr_path=self.model_trainer_cfg.meta_lr_path,
                enable_tuning=self.model_trainer_cfg.enable_tuning,
                final_model_path=self.model_trainer_cfg.final_model_path,
                lr_level1_oof_predictions_path=self.model_trainer_cfg.lr_level1_oof_predictions_path,
                xgb_level1_oof_predictions_path=self.model_trainer_cfg.xgb_level1_oof_predictions_path,
                folds=self.model_trainer_cfg.folds,
            )
        except Exception as e:
            logger.error("Failed to build ModelTrainerConfig")
            raise AITextException(e)

    # 5. Model Trainer Tuning Config (Optuna)
    def get_model_trainer_tuning_config(self) -> ModelTrainerTuningConfig:
        try:
            lr_space = LRSpace(**self.tuning_cfg.level1.lr)
            xgb_space = XGBSpace(**self.tuning_cfg.level1.xgb)

            level1 = Level1TuningConfig(lr=lr_space, xgb=xgb_space)
            level2 = Level2TuningConfig(lr=LRSpace(**self.tuning_cfg.level2.lr))

            return ModelTrainerTuningConfig(
                level1=level1,
                level2=level2,
                n_trials=self.tuning_cfg.number_of_trials,
            )
        except Exception as e:
            logger.error("Failed to build ModelTrainerTuningConfig")
            raise AITextException(e)

    # 6. Final parameters after Optuna tuning
    def get_model_trainer_final_params_config(self) -> ModelTrainerFinalParamsConfig:
        try:
            lr_final = LRFinalParams(**self.final_params_cfg.level1.lr)
            xgb_final = XGBFinalParams(**self.final_params_cfg.level1.xgb)

            level1_final = Level1FinalParams(lr=lr_final, xgb=xgb_final)
            level2_final = Level2FinalParams(
                lr=LRFinalParams(**self.final_params_cfg.level2.lr)
            )

            return ModelTrainerFinalParamsConfig(
                level1=level1_final, level2=level2_final
            )
        except Exception as e:
            logger.error("Failed to build ModelTrainerFinalParamsConfig")
            raise AITextException(e)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            create_dir(self.model_evaluation_cfg.root_dir, 'Model evalutaion root artifact')

            return ModelEvaluationConfig(
                raw_test_data_path= self.data_ingestion_cfg.local_test_path,
                final_model_path=self.model_trainer_cfg.final_model_path,
                target_column_name=self.schema.target_column_name,
                model_evaluation_artifact_file_path= self.model_evaluation_cfg.model_evaluation_file_path,
                metrices=self.model_evaluation_cfg.metrices,
                push_model_to_gcs=self.model_evaluation_cfg.push_model_to_gcs,
                gcs_bucket_name= self.data_ingestion_cfg.bucket_name
            )
        except Exception as e:
            logger.error("Failed to build model evaluation config")
            raise AITextException(e)

    