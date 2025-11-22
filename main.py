from src.pipeline.training_pipeline import TrainingPipeline
from dotenv import load_dotenv
load_dotenv()
from src.utils.logger import logging


if __name__ == "__main__":
    logging.info("🚀 Starting Training Pipeline...")
    pipeline = TrainingPipeline()
    pipeline.start_data_ingestion()
    logging.info("✅ Training Pipeline Completed.")