from src.utils.common import read_object


class PredictionPipeline:
    def __init__(self, final_model_path: str):
        self.model = read_object(final_model_path)

    def predict(self, X):
        return self.model.predict_proba(X)
