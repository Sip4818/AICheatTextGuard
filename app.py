import redis.exceptions
from pydantic import BaseModel, Field
import pandas as pd
from typing import Annotated
from fastapi import FastAPI
from src.pipeline.prediction.prediction_pipeline import PredictionPipeline
import redis
from prometheus_fastapi_instrumentator import Instrumentator
from src.constants.constants import final_model_path
import hashlib
import json
import asyncio


app = FastAPI()

try:
    r = redis.Redis(host='redis', port=6379, decode_responses=True)
    r.ping()  # Force connection test
except redis.exceptions.RedisError:
    print("Redis not available. Running without cache.")
    r = None


Instrumentator().instrument(app).expose(app)

predictor = PredictionPipeline(final_model_path)


class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length= 250, max_length = 5000)] 

    def cache_key(self):
        raw = json.dumps(self.model_dump(), sort_keys=True)
        return f"Predict:{hashlib.sha256(raw.encode()).hexdigest()}"

@app.post("/predict")
async def predict(req: PredictRequest):

    key = req.cache_key()
    cached_result = None
    if r:
        try:

            cached_result = r.get(key)
        except redis.exceptions.RedisError:
            cached_result = None
    if cached_result:
        print("Serving prediction from cache!")
        return json.loads(cached_result)
    df = pd.DataFrame(
        [
            {
                "text": req.text,
            }
        ]
    )
    proba = await asyncio.to_thread(predictor.predict,df)
    response = {"probability": float(proba[0][1])}
    if r:
        try:
            r.set(key, json.dumps(response), ex =600)
        except redis.exceptions.RedisError:
            pass
    return response
