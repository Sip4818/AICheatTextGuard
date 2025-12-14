from src.entity.config_entity import PredictionConfig
from src.utils.common import read_object

class PredictionPipeline:
    def __init__(self, cfg: PredictionConfig):
        self.model = read_object(cfg.final_trained_model_path)

    def predict(self, X):
        return self.model.predict_proba(X)
