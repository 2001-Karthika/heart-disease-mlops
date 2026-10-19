import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import create_app  # noqa: E402


@pytest.fixture()
def client(trained_artifacts):
    model_path, _ = trained_artifacts
    app = create_app(model_path)
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_root_lists_endpoints(client):
    assert "predict" in client.get("/").json()


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    resp = client.post("/predict", json=sample_patient)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert "model_name" in body


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    payload = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_missing_required_field(client, sample_patient):
    payload = dict(sample_patient)
    payload.pop("age")
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_out_of_range_value(client, sample_patient):
    assert client.post("/predict", json=dict(sample_patient, sex=5)).status_code == 422
    assert client.post("/predict", json=dict(sample_patient, thal=4)).status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "predictions_total" in text


def test_predict_without_model_returns_503(tmp_path, sample_patient):
    app = create_app(tmp_path / "missing.joblib")
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["status"] == "degraded"
        assert test_client.post("/predict", json=sample_patient).status_code == 503
