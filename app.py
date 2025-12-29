from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI
from src.pipeline.prediction.prediction_pipeline import PredictionPipeline
from src.constants.constants import final_model_path


app = FastAPI()

predictor = PredictionPipeline(final_model_path)


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(req: PredictRequest):
    df = pd.DataFrame(
        [
            {
                "text": req.text,
            }
        ]
    )
    proba = predictor.predict(df)
    return {"probability": float(proba[0][1])}
