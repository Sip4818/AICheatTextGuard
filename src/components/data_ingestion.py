import os
from src.utils.logger import logger
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.common import download_from_gcs, log_file_size, assert_file_exists
from src.utils.exception import AITextException


class DataIngestion:
    
    def __init__(self, cfg: DataIngestionConfig) -> None:
        self.cfg = cfg

    def download_data(self) -> None:
        try:
            
            logger.info("Starting data download from GCS")

            logger.info(f"Downloading TRAIN file: {self.cfg.cloud_train_path}")
            download_from_gcs(
                bucket_name=self.cfg.bucket_name,
                source_path=self.cfg.cloud_train_path,
                local_path=self.cfg.local_train_path,
            )

            logger.info(f"Downloading TEST file: {self.cfg.cloud_test_path}")
            download_from_gcs(
                bucket_name=self.cfg.bucket_name,
                source_path=self.cfg.cloud_test_path,
                local_path=self.cfg.local_test_path,
            )

            logger.info("Data download completed successfully")

        except Exception as e:
            logger.error("Data download failed during ingestion step")
            raise AITextException(e)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Initiating data ingestion pipeline")
            self.download_data()

            assert_file_exists(self.cfg.local_train_path, "Train file")
            assert_file_exists(self.cfg.local_test_path, "Test file")
            
            log_file_size(self.cfg.local_train_path, "Train file")
            log_file_size(self.cfg.local_test_path, "Test file")

            artifact = DataIngestionArtifact(
                local_train_file_path=self.cfg.local_train_path,
                local_test_file_path=self.cfg.local_test_path
            )

            logger.info(f"Data ingestion completed: {artifact}")
            return artifact

        except Exception as e:
            logger.error("Failed to initiate data ingestion")
            raise AITextException(e)
