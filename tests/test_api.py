from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

from app import app

client = TestClient(app)

valid_text = (
    "This is a sufficiently long text that needs to be at least two hundred and fifty characters long to pass the pydantic validation in the predict request. "
    * 3
)


def test_predict_valid_request():
    with patch("app.predictor.predict") as mock_predict:
        # Mock the return value of predict. `proba[0][1]` is accessed.
        mock_predict.return_value = np.array([[0.1, 0.9]])

        response = client.post("/predict", json={"text": valid_text})
        assert response.status_code == 200
        assert "probability" in response.json()
        assert response.json()["probability"] == 0.9


def test_predict_text_too_short():
    short_text = "Too short"
    response = client.post("/predict", json={"text": short_text})
    assert response.status_code == 422
    assert "detail" in response.json()
    assert response.json()["detail"][0]["type"] == "string_too_short"


def test_predict_text_too_long():
    long_text = "A" * 5001
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422
    assert "detail" in response.json()
    assert response.json()["detail"][0]["type"] == "string_too_long"
