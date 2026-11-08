"""Create an untrained contract fixture or smoke-test a running local API.

Never use this fixture as an approved live demo bundle or quality evidence.
"""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


urlopen = build_opener(_NoRedirect()).open


def create_fixture(path):
    from vigil.ml.models.export import export_model_bundle
    from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier
    from vigil.ml.dataset.normalizer import FeatureNormalizer

    normalizer = FeatureNormalizer()
    normalizer.means = {"ear_avg": 0.28, "blink_duration_ms": 150.0, "perclos": 0.12}
    normalizer.stds = {"ear_avg": 0.05, "blink_duration_ms": 50.0, "perclos": 0.08}
    normalizer.observed_counts = {name: 100 for name in normalizer.means}
    normalizer.is_fitted = True
    export_model_bundle(
        GRUTemporalClassifier(GRUConfig(hidden_dim=8, num_layers=1, dropout=0.0)),
        normalizer,
        path,
        metadata={"purpose": "untrained infrastructure fixture"},
    )
    print("Created untrained infrastructure fixture, not a demo model.")


def check(url, expected_bundle_sha256=None):
    from vigil.cloud.contracts import PredictResponse
    from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS

    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("smoke URL must be HTTPS or loopback HTTP without credentials")
    with urlopen(url.rstrip("/") + "/health", timeout=10) as response:
        health = json.load(response)
    bundle_id = health.get("bundle_id")
    if health.get("status") != "ready" or not isinstance(bundle_id, str) or not re.fullmatch(r"[a-f0-9]{64}", bundle_id):
        raise ValueError("cloud service is not ready with a valid bundle digest")
    if expected_bundle_sha256 is not None and bundle_id != expected_bundle_sha256:
        raise ValueError("cloud bundle digest does not match approved digest")
    unauthenticated = Request(url.rstrip("/") + "/model-info")
    try:
        urlopen(unauthenticated, timeout=10)
    except HTTPError as error:
        if error.code != 401:
            raise
    else:
        raise AssertionError("model metadata was accessible without authentication")
    payload = {
        "bundle_id": bundle_id,
        "feature_channels": list(LOCKED_FEATURE_CHANNELS),
        "features": [[0.3, 0.0, 0.0, 0.1, 1.0, 1.0, 1.0, 1.0]] * 60,
    }
    token = os.environ.get("VIGIL_API_TOKEN", "")
    if len(token) < 32:
        raise ValueError("VIGIL_API_TOKEN must be configured for smoke testing")
    info = Request(
        url.rstrip("/") + "/model-info",
        headers={"Authorization": "Bearer " + token},
    )
    with urlopen(info, timeout=10) as response:
        metadata = json.load(response)
    if (
        metadata.get("bundle_id") != bundle_id
        or metadata.get("feature_channels") != list(LOCKED_FEATURE_CHANNELS)
        or metadata.get("window_length") != 60
        or metadata.get("stride") != 15
        or metadata.get("input_encoding") != "raw_with_masks_v1"
    ):
        raise ValueError("cloud model metadata does not match the locked contract")
    request = Request(
        url.rstrip("/") + "/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    with urlopen(request, timeout=10) as response:
        prediction = PredictResponse.model_validate(json.load(response))
    if prediction.bundle_id != bundle_id:
        raise ValueError("cloud prediction bundle digest changed")
    print("Health and prediction schema passed; no model quality claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-fixture", type=Path)
    mode.add_argument("--url")
    parser.add_argument("--expected-bundle-sha256")
    args = parser.parse_args()
    if args.create_fixture:
        create_fixture(args.create_fixture)
    else:
        check(args.url, args.expected_bundle_sha256)
