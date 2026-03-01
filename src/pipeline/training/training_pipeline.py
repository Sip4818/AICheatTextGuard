import argparse
import sys
from src.components import model_evaluation
from src.utils.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)
from config.configuration import ConfigurationManager
from config.training_pipeline_config import TrainingPipelineConfig
from src.constants.constants import config_yaml_file_path, schema_yaml_file_path


class TrainingPipeline:
    def __init__(self):
        pipeline_config = TrainingPipelineConfig()

        self.config_manager = ConfigurationManager(
            training_pipeline_config=pipeline_config,
            yaml_file_path=config_yaml_file_path,
            schema_file_path=schema_yaml_file_path,
        )

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Data Ingestion stage started.")

        # Step 3 - Fetch DataIngestionConfig object
        data_ingestion_config = self.config_manager.get_data_ingestion_config()

        # Step 4 - Create DataIngestion component
        data_ingestion = DataIngestion(cfg=data_ingestion_config)

        # Step 5 - Run ingestion
        ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info(f"Data Ingestion stage completed: {ingestion_artifact}.")
        return ingestion_artifact

    def start_data_validation(self) -> DataValidationArtifact:
        logger.info("Data validation stage started")

        data_validation_config = self.config_manager.get_data_validation_config()

        data_validation = DataValidation(cfg=data_validation_config)

        validation_artifact = data_validation.initiate_data_validation()
        logger.info(f"Data validation stage completed: {validation_artifact}")

        return validation_artifact

    def start_data_transformation(self) -> DataTransformationArtifact:
        logger.info("Data Transformation stage started")

        data_transformation_config = (
            self.config_manager.get_data_transformation_config()
        )

        data_transformation = DataTransformation(data_transformation_config)

        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )

        logger.info(f"Data Transformation stage completed")

        return data_transformation_artifact

    def start_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Model trainer stage started")

        model_trainer_config = self.config_manager.get_model_trainer_config()
        model_tuning_config = self.config_manager.get_model_trainer_tuning_config()
        model_trainer_final_params_config = (
            self.config_manager.get_model_trainer_final_params_config()
        )

        model_trainer = ModelTrainer(
            model_trainer_config, model_tuning_config, model_trainer_final_params_config
        )

        model_trainer_artifact = model_trainer.initiate_model_training()
        logger.info("Model trainer stage completed")

        return model_trainer_artifact
    
    def start_model_evaluation(self) -> ModelEvaluationArtifact:
        logger.info(' model evaluation stage started')
        model_evaluation_config =self.config_manager.get_model_evaluation_config()

        model_evaluation= ModelEvaluation(model_evaluation_config)
        model_evaluation_artifact=model_evaluation.initiate_model_evaluation()
        logger.info("Model Evaluation stage completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true', help="Run a fast smoke test")
    args = parser.parse_args()

    try:
        pipeline = TrainingPipeline()
        
        # 1. Start Ingestion
        ingestion_artifact = pipeline.start_data_ingestion()
        
        # DEBUG LOGIC: If debugging, we limit the data early 
        # to save time/compute in GitHub Actions
        if args.debug_mode:
            logger.info("DEBUG MODE ENABLED: Skipping heavy training.")
            # In debug mode, we might just stop after ingestion or validation
            # to prove the code 'links' together without training for 20 mins.
            sys.exit(0) 

        # 2. Continue with full pipeline if not debugging
        pipeline.start_data_validation()
        pipeline.start_data_transformation()
        pipeline.start_model_trainer()
        pipeline.start_model_evaluation()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)