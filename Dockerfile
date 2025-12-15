FROM python:3.14-slim

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY ./artifact/model_trainer/stacked_model.pkl ./artifact/model_trainer/

COPY ./src/pipeline/prediction/prediction_pipeline.py ./src/pipeline/prediction/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--port", "8080"]
