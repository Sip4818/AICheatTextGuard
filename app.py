from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI
from src.pipeline.prediction.prediction_pipeline import PredictionPipeline
from src.constants.constants import final_model_path

app = FastAPI()


predictor = PredictionPipeline(final_model_path)


class PredictRequest(BaseModel):
    id: str
    topic: str
    answer: str


@app.post("/predict")
async def predict(req: PredictRequest):
    df = pd.DataFrame(
        [
            {
                "id": req.id,
                "topic": req.topic,
                "answer": req.answer,
            }
        ]
    )
    proba = predictor.predict(df)
    return {"probability": float(proba[0][1])}
