from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_valid_input():
    response = client.post("/predict", json={"text": "Wall Street stocks rise on Fed news"})
    assert response.status_code == 200
    data = response.json()

    # Check all required fields exist
    assert "label" in data
    assert "confidence" in data
    assert "model_used" in data
    assert "latency_ms" in data

    # Check types and value ranges
    assert isinstance(data["label"], str)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_empty_text_rejected():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_predict_whitespace_only_rejected():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
