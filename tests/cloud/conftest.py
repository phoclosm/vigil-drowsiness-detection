"""Temporary untrained bundles test mechanics, never real model quality."""
# ruff: noqa: E402 -- check optional dependencies before importing cloud fixtures.

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from vigil.ml.models.export import export_model_bundle
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.data import LiveInputFrame


@pytest.fixture
def bundle_path(tmp_path):
    normalizer = FeatureNormalizer()
    normalizer.means = {"ear_avg": 0.28, "blink_duration_ms": 150.0, "perclos": 0.12}
    normalizer.stds = {"ear_avg": 0.05, "blink_duration_ms": 50.0, "perclos": 0.08}
    normalizer.observed_counts = {name: 100 for name in normalizer.means}
    normalizer.is_fitted = True
    return export_model_bundle(
        GRUTemporalClassifier(GRUConfig(hidden_dim=8, num_layers=1, dropout=0.0)),
        normalizer,
        tmp_path / "untrained-contract-only.pt",
        metadata={"purpose": "synthetic contract fixture; not trained"},
    )


@pytest.fixture
def live_frames():
    return [
        LiveInputFrame(
            session_id="contract-test",
            frame_index=i,
            timestamp_ms=float(i * 30),
            frame_delta_ms=None if i == 0 else 30.0,
            face_detected=True,
            landmarks_valid=True,
            ear_left=0.30,
            ear_right=0.30,
            ear_avg=0.30,
            is_eye_closed=False,
            blink_count=0,
            blink_duration_ms=0.0,
            perclos=0.1,
            fps=None,
        )
        for i in range(90)
    ]
