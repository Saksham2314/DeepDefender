import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_200_regardless_of_model_state():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_empty_text_returns_422():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_field_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_malformed_json_returns_422():
    response = client.post("/predict", content=b"not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_predict_whitespace_only_text_returns_422():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_when_model_unavailable_returns_503_not_a_traceback():
    # In an environment where the model hasn't been trained/loaded yet,
    # /predict must fail cleanly with 503, never a raw stack trace.
    response = client.post("/predict", json={"text": "tum bahut bakwas ho"})
    assert response.status_code in (200, 503)
    body = response.json()
    assert "detail" not in body or "Traceback" not in str(body.get("detail", ""))


def test_predict_very_long_text_is_rejected_or_handled():
    long_text = "bakwas " * 2000  # well beyond MAX_TEXT_LENGTH
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code in (200, 422, 503)


def test_explain_missing_field_returns_422():
    response = client.post("/explain", json={})
    assert response.status_code == 422


def test_explain_invalid_method_returns_422():
    response = client.post("/explain", json={"text": "test", "method": "not_a_real_method"})
    assert response.status_code == 422
