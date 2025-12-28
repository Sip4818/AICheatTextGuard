from src.utils.logger import logger
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.common import (
    download_from_gcs,
    log_file_size,
    assert_file_exists,
    read_csv_file,
    save_csv,
)
from src.utils.exception import AITextException
from sklearn.model_selection import train_test_split
from src.constants.constants import SEED



class DataIngestion:
    def __init__(self, cfg: DataIngestionConfig) -> None:
        self.cfg = cfg

    def download_data(self) -> None:
        try:
            logger.info("Starting data download from GCS")

            logger.info(f"Downloading data : {self.cfg.cloud_data_path}")
            download_from_gcs(
                bucket_name=self.cfg.bucket_name,
                source_path=self.cfg.cloud_data_path,
                local_path=self.cfg.local_data_path,
            )

            logger.info("Data download completed successfully")

        except Exception as e:
            logger.error("Data download failed during ingestion step")
            raise AITextException(e)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Initiating data ingestion pipeline")
            if self.cfg.to_download_data:
                self.download_data()
                logger.info("Data download config is set False: Data is not downloaded")

            assert_file_exists(self.cfg.local_data_path, "data file")

            log_file_size(self.cfg.local_data_path, "data file")

            data = read_csv_file(self.cfg.local_data_path)
            train, test = train_test_split(
                data, test_size=self.cfg.test_split_size, random_state=SEED, stratify= data[self.cfg.target_column_name]
            )

            save_csv(train, self.cfg.local_train_path)
            assert_file_exists(self.cfg.local_train_path, "train data file")
            log_file_size(self.cfg.local_train_path, "train data")

            save_csv(test, self.cfg.local_test_path)
            assert_file_exists(self.cfg.local_test_path, "test data file")
            log_file_size(self.cfg.local_test_path, "test data")

            artifact = DataIngestionArtifact(
                local_train_file_path=self.cfg.local_train_path,
                local_test_file_path=self.cfg.local_test_path,
            )

            logger.info(f"Data ingestion completed: {artifact}")
            return artifact

        except Exception as e:
            logger.error("Failed to initiate data ingestion")
            raise AITextException(e)
