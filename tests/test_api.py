import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.config import DRIFT_MIN_SAMPLES


@pytest.fixture
def client(champion):
    with TestClient(app) as client:  # startet den Lifespan, lädt also den Champion
        yield client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["model_type"] in {"LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"}
    assert set(info["test_metrics"]) == {"roc_auc", "precision", "recall", "f1"}


def test_predict_returns_probability_and_decision(client, application, prediction_log):
    response = client.post("/predict", json=application)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["decision"] in {"approve", "decline"}
    assert prediction_log.exists()


def test_high_risk_application_has_higher_probability(client, application):
    risky = {**application, "fico": 640, "int_rate": 0.2, "inq_last_6mths": 6, "credit_policy": 0}
    safe = client.post("/predict", json=application).json()["default_probability"]
    assert client.post("/predict", json=risky).json()["default_probability"] > safe


def test_predict_rejects_invalid_input(client, application):
    response = client.post("/predict", json={**application, "fico": 1000})
    assert response.status_code == 422


def test_drift_needs_enough_requests(client, application):
    client.post("/predict", json=application)
    assert client.get("/drift").json()["status"] == "insufficient_data"


def test_drift_detects_shifted_inputs(client, application):
    for i in range(DRIFT_MIN_SAMPLES):
        client.post("/predict", json={**application, "fico": 800, "revol_bal": 100.0 + i})
    report = client.get("/drift").json()
    assert report["status"] == "drift"
    assert report["features"]["fico"]["drift"] is True
