from src.utils.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import (
     DataIngestionArtifact,
       DataTransformationArtifact,
       DataValidationArtifact,
         ModelTrainerArtifact
)
from config.configuration import ConfigurationManager
from config.training_pipeline_config import TrainingPipelineConfig
from src.constants.constants import config_yaml_file_path,schema_yaml_file_path

class TrainingPipeline:
    def __init__(self):
        pipeline_config = TrainingPipelineConfig()

        self.config_manager = ConfigurationManager(
            training_pipeline_config=pipeline_config,
            yaml_file_path=config_yaml_file_path,
            schema_file_path=schema_yaml_file_path
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


    def start_data_validation(self)->DataValidationArtifact:

        logger.info("Data validation stage started")

        data_validation_config=self.config_manager.get_data_validation_config()

        data_validation=DataValidation(cfg=data_validation_config)

        validation_artifact=data_validation.initiate_data_validation()
        logger.info(f"Data validation stage completed: {validation_artifact}")

        return validation_artifact

    def start_data_transformation(self)->DataTransformationArtifact:
        
        logger.info("Data Transformation stage started")

        data_transformation_config=self.config_manager.get_data_transformation_config()

        data_transformation=DataTransformation(data_transformation_config)

        data_transformation_artifact=data_transformation.initiate_data_transformation()

        logger.info(f"Data Transformation stage completed")

        return data_transformation_artifact
    

    def start_model_trainer(self)->ModelTrainerArtifact:

        logger.info("Model trainer stage started")

        model_trainer_config= self.config_manager.get_model_trainer_config()
        model_tuning_config=self.config_manager.get_model_trainer_tuning_config()
        model_trainer_final_params_config=self.config_manager.get_model_trainer_final_params_config()

        model_trainer= ModelTrainer(model_trainer_config,model_tuning_config,model_trainer_final_params_config)

        model_trainer_artifact=model_trainer.initiate_model_training()
        logger.info("Model trainer stage completed")

        return model_trainer_artifact

