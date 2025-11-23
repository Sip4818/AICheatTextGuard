from src.utils.logger import logger
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.common import download_from_gcs
from src.utils.exception import AITextException
class DataIngestion:
    def __init__(self, cfg: DataIngestionConfig):
        self.cfg = cfg

    def download_data(self):

        download_from_gcs(
            bucket_name=self.cfg.bucket_name,
            source_path=self.cfg.cloud_train_path,
            local_path=self.cfg.local_train_path,
        )
        download_from_gcs(
            bucket_name=self.cfg.bucket_name,
            source_path=self.cfg.cloud_test_path,
            local_path=self.cfg.local_test_path,
        )


    def initiate_data_ingestion(self)->DataIngestionArtifact:
        self.download_data()
        return DataIngestionArtifact(
            local_train_file_path=self.cfg.local_train_path,
            local_test_file_path=self.cfg.local_test_path
        )
