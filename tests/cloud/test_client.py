"""Mock HTTPS boundary and verify parity of temporal control flow."""

import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from vigil.cloud.app import create_app
from vigil.ml.inference import TemporalInferenceEngine
from vigil.runtime.cloud_inference import CloudInferenceEngine, CloudTransport
from vigil.runtime.inference import ModelInferenceError


def make_transport(handler, **kwargs):
    return CloudTransport(
        "https://vigil.example",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://vigil.example",
        "https://user:pass@host",
        "https://host/?secret=x",
        "file:///model",
    ],
)
def test_transport_requires_clean_https(url):
    with pytest.raises(ValueError):
        CloudTransport(url)


def test_transport_retries_are_bounded():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private endpoint detail")

    transport = make_transport(handler, retries=1)
    with pytest.raises(ModelInferenceError, match="^cloud inference unavailable$"):
        transport.connect()
    assert len(calls) == 2
    transport.close()


def test_client_and_server_match_local_warmup_stride_gaps(bundle_path, live_frames):
    requests = []
    with TestClient(create_app(bundle_path)) as api:

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=api.get("/model-info").json())
            payload = json.loads(request.content)
            requests.append(payload)
            assert set(payload) == {"feature_channels", "features", "bundle_id"}
            response = api.post("/predict", json=payload)
            return httpx.Response(response.status_code, json=response.json())

        transport = make_transport(handler).connect()
        cloud = CloudInferenceEngine(transport)
        local = TemporalInferenceEngine.from_bundle(bundle_path)
        for frame in live_frames:
            a, b = local.process_sample(frame), cloud.process_sample(frame)
            assert a.status == b.status
            assert a.reason == b.reason
            assert a.predicted_class == b.predicted_class
            if a.probabilities:
                assert a.probabilities == pytest.approx(b.probabilities)
        assert len(requests) == 3  # frames 59, 74, 89
        gap = replace(live_frames[-1], frame_index=93, timestamp_ms=2790.0)
        assert cloud.process_sample(gap).reason == local.process_sample(gap).reason
        assert cloud.process_sample(gap).status == "STATUS_INVALID_INPUT"
        transport.close()


@pytest.mark.parametrize("mutation", ["class", "confidence", "bundle", "nan"])
def test_bad_cloud_prediction_is_rejected(mutation):
    data = {
        "predicted_class": "ACTIVE",
        "class_id": 0,
        "confidence": 0.8,
        "probabilities": {"ACTIVE": 0.8, "DROWSY": 0.1, "SLEEPING": 0.1},
        "latency_ms": 1.0,
        "bundle_id": "a" * 64,
    }
    if mutation == "class":
        data["class_id"] = 2
    elif mutation == "confidence":
        data["confidence"] = 0.5
    elif mutation == "bundle":
        data["bundle_id"] = "b" * 64
    else:
        data["latency_ms"] = float("nan")
    transport = make_transport(
        lambda _: httpx.Response(
            200, content=json.dumps(data), headers={"Content-Type": "application/json"}
        )
    )
    transport.bundle_id = "a" * 64
    with pytest.raises(ModelInferenceError):
        transport.predict([[0.3, 0.0, 0.0, 0.1, 1.0, 1.0, 1.0, 1.0]] * 60)
    transport.close()


def test_all_missing_window_is_not_sent(live_frames):
    class NoNetwork:
        def predict(self, _):
            raise AssertionError("must not send an unobservable window")

    engine = CloudInferenceEngine(NoNetwork())
    for frame in live_frames[:60]:
        result = engine.process_sample(
            replace(
                frame,
                face_detected=False,
                landmarks_valid=False,
                ear_left=None,
                ear_right=None,
                ear_avg=None,
                is_eye_closed=None,
                perclos=None,
            )
        )
    assert result.status == "STATUS_INVALID_INPUT"
