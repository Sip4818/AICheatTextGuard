from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi import FastAPI
from src.pipeline.prediction.prediction_pipeline import PredictionPipeline
from src.utils.common import read_yaml
from src.constants.constants import config_yaml_file_path
from src.entity.config_entity import PredictionConfig
app = FastAPI()
# ---- CREATE ON STARTUP ----
def load_prediction_config() -> PredictionConfig:
    cfg = read_yaml(config_yaml_file_path)
    pred_cfg = cfg.prediction # or cfg["prediction"] if you add it

    return PredictionConfig(
        final_trained_model_path=pred_cfg.final_model_path
    )

prediction_cfg = load_prediction_config()
predictor = PredictionPipeline(prediction_cfg)
class PredictRequest(BaseModel):
    id: str
    topic: str
    answer: str

@app.post("/predict")
async def predict(req: PredictRequest):
    df = pd.DataFrame([{
    "id":req.id,
    "topic": req.topic,        # OR req.topic if that's correct
    "answer": req.answer
}])
    proba = predictor.predict(df)
    return {"probability": float(proba[0][1])}

