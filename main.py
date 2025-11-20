from src.pipeline.training_pipeline import TrainingPipeline
from dotenv import load_dotenv
load_dotenv()
training_pipeline=TrainingPipeline()
print(training_pipeline.start_data_ingestion())