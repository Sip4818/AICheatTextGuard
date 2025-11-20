from src.components.data_ingestion import DataIngestion
from src.entity.artifact_entity import DataIngestionArtifact
from config.configuration import ConfigurationManager
from config.training_pipeline_config import TrainingPipelineConfig
from src.constants.constants import config_yaml_file_path

class TrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self) -> DataIngestionArtifact:

        # Step 1 - Create pipeline root config
        pipeline_config = TrainingPipelineConfig()

        # Step 2 - Create config manager with YAML + pipeline config
        config_manager = ConfigurationManager(
            training_pipeline_config=pipeline_config,
            yaml_file_path=config_yaml_file_path
        )

        # Step 3 - Fetch DataIngestionConfig object
        data_ingestion_config = config_manager.get_data_ingestion_config()

        # Step 4 - Create DataIngestion component
        data_ingestion = DataIngestion(cfg=data_ingestion_config)

        # Step 5 - Run ingestion
        ingestion_artifact = data_ingestion.initiate_data_ingestion()

        return ingestion_artifact


