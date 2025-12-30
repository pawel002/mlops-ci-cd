from fastapi.testclient import TestClient
from unittest.mock import patch
from sentiment_app.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("sentiment_app.app.inference")
def test_predict_endpoint_success(mock_inference):
    mock_inference.predict.return_value = "positive"
    
    payload = {"text": "Hello world"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}
    
    mock_inference.predict.assert_called_once_with("Hello world")

@patch("sentiment_app.app.inference")
def test_predict_endpoint_empty_string(mock_inference):
    mock_inference.predict.return_value = "neutral"
    
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"prediction": "neutral"}

def test_predict_endpoint_validation_error():
    response = client.post("/predict", json={})
    
    assert response.status_code == 422