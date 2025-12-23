import os
from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI
from src.pipeline.prediction.prediction_pipeline import PredictionPipeline
from src.utils.common import download_from_gcs
from src.constants.constants import final_model_path, bucket_name
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

if not os.path.exists(final_model_path):
    download_from_gcs(
        bucket_name=bucket_name,
        source_path=final_model_path,
        local_path=final_model_path
    )



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
