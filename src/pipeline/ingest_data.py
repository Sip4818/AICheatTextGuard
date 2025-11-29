from src.pipeline.training_pipeline import TrainingPipeline
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_data_ingestion()
