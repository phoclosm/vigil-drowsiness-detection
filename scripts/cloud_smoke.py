"""Create an untrained contract fixture or smoke-test a running local API.

Never use this fixture as an approved live demo bundle or quality evidence.
"""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


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


def check(url):
    from vigil.cloud.contracts import PredictResponse
    from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS

    with urlopen(url.rstrip("/") + "/health", timeout=10) as response:
        health = json.load(response)
    payload = {
        "bundle_id": health["bundle_id"],
        "feature_channels": list(LOCKED_FEATURE_CHANNELS),
        "features": [[0.3, 0.0, 0.0, 0.1, 1.0, 1.0, 1.0, 1.0]] * 60,
    }
    request = Request(
        url.rstrip("/") + "/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        PredictResponse.model_validate(json.load(response))
    print("Health and prediction schema passed; no model quality claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-fixture", type=Path)
    mode.add_argument("--url")
    args = parser.parse_args()
    if args.create_fixture:
        create_fixture(args.create_fixture)
    else:
        check(args.url)
