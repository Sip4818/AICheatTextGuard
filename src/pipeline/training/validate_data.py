from src.pipeline.training.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_data_validation()
