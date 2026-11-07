"""Strict request validation, error privacy and real bundle execution."""

import copy

import pytest
from fastapi.testclient import TestClient

from vigil.cloud.app import create_app
from vigil.cloud.bundle import BundlePredictor
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS
from vigil.ml.dataset.windowing import build_sample_feature_vector
from vigil.ml.inference import TemporalInferenceEngine


@pytest.fixture
def client(bundle_path):
    with TestClient(create_app(bundle_path)) as client:
        yield client


@pytest.fixture
def payload(client):
    return {
        "feature_channels": list(LOCKED_FEATURE_CHANNELS),
        "features": [[0.30, 0.0, 0.0, 0.1, 1.0, 1.0, 1.0, 1.0] for _ in range(60)],
        "bundle_id": client.get("/health").json()["bundle_id"],
    }


def test_health_and_real_inference(client, payload):
    assert client.get("/health").status_code == 200
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ("ACTIVE", "DROWSY", "SLEEPING")
    assert sum(data["probabilities"].values()) == pytest.approx(1)
    assert data["latency_ms"] >= 0
    assert len(response.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize(
    "mutation",
    ["short", "wide", "reorder", "nan", "bool", "mask", "missing", "image", "huge"],
)
def test_malformed_payloads_do_not_echo_inputs(client, payload, mutation):
    data = copy.deepcopy(payload)
    if mutation == "short":
        data["features"].pop()
    elif mutation == "wide":
        data["features"][0].append(1.0)
    elif mutation == "reorder":
        data["feature_channels"].reverse()
    elif mutation == "nan":
        # JSON strings must never coerce into a float or reveal input in errors.
        data["features"][0][0] = "NaN"
    elif mutation == "bool":
        data["features"][0][0] = True
    elif mutation == "mask":
        data["features"][0][4] = 0.5
    elif mutation == "missing":
        data["features"][0][4] = 0.0
    elif mutation == "image":
        data["image"] = "/private/secret.png"
    elif mutation == "huge":
        data["features"][0][0] = float("1e308")
    response = client.post("/predict", json=data)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid temporal feature payload"}


def test_invalid_json_nonfinite_and_size(client, payload):
    import json

    text = json.dumps(payload).replace("0.3", "NaN", 1)
    response = client.post(
        "/predict", content=text, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert client.post("/predict", content=b"x" * 131073).status_code == 413


def test_missing_bundle_is_not_ready(tmp_path):
    with TestClient(create_app(tmp_path / "secret.pt")) as client:
        assert client.get("/health").status_code == 503
        assert "secret.pt" not in client.get("/health").text


def test_failure_and_changed_bundle(client, payload):
    payload["bundle_id"] = "0" * 64
    assert client.post("/predict", json=payload).status_code == 409
    payload["bundle_id"] = client.get("/health").json()["bundle_id"]

    def broken(_):
        raise RuntimeError("/private/model.pt secret")

    client.app.state.predictor.predict = broken
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json() == {"detail": "inference unavailable"}


def test_api_matches_local_normalization_and_prediction(
    client, live_frames, bundle_path, payload
):
    engine = TemporalInferenceEngine.from_bundle(bundle_path)
    for sample in live_frames[:60]:
        result = engine.process_sample(sample)
    payload["features"] = [build_sample_feature_vector(s) for s in live_frames[:60]]
    remote = client.post("/predict", json=payload).json()
    assert remote["predicted_class"] == result.predicted_class
    assert remote["probabilities"] == pytest.approx(result.probabilities)


def test_predictor_rejects_mismatched_bundle(bundle_path, payload):
    from vigil.cloud.contracts import PredictRequest

    payload["bundle_id"] = "0" * 64
    with pytest.raises(ValueError):
        BundlePredictor(bundle_path).predict(PredictRequest(**payload))
