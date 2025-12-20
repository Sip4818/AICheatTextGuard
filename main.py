from src.pipeline.training.training_pipeline import TrainingPipeline
from dotenv import load_dotenv

load_dotenv()
from src.utils.logger import logging
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="google_crc32c")


if __name__ == "__main__":
    logging.info(" Starting Training Pipeline...")
    pipeline = TrainingPipeline()
    pipeline.start_data_ingestion()
    pipeline.start_data_validation()
    pipeline.start_data_transformation()
    pipeline.start_model_trainer()
    logging.info("Training Pipeline Completed.")
