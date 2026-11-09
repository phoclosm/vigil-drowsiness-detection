"""M2-owned GRU bundle handoff contract tests.

These tests use an untrained infrastructure fixture and make no model-quality
or accuracy claim.
"""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from vigil.cloud.bundle import BundlePredictor  # noqa: E402
from vigil.cloud.contracts import PredictRequest  # noqa: E402
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS  # noqa: E402
from vigil.ml.dataset.normalizer import FeatureNormalizer  # noqa: E402
from vigil.ml.models.export import export_model_bundle, load_model_bundle  # noqa: E402
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier  # noqa: E402
from vigil.ml.provenance_types import NormalizerProvenance  # noqa: E402


def _fixture_normalizer() -> FeatureNormalizer:
    normalizer = FeatureNormalizer(
        means={"ear_avg": 0.28, "blink_duration_ms": 150.0, "perclos": 0.12},
        stds={"ear_avg": 0.05, "blink_duration_ms": 50.0, "perclos": 0.08},
        observed_counts={
            "ear_avg": 100,
            "blink_duration_ms": 100,
            "perclos": 100,
        },
        is_fitted=True,
        provenance=NormalizerProvenance(
            fit_partition="train",
            fit_record_digest="synthetic-training-records",
            fit_record_count=60,
            fit_subjects=("synthetic-subject",),
            fit_sessions=("synthetic-session",),
            feature_channels=LOCKED_FEATURE_CHANNELS,
            source_manifest_digest="synthetic-manifest",
            parameter_digest="synthetic-parameters",
            training_partition_digest="synthetic-partition",
            status="legacy_unverified",
        ),
    )
    return normalizer


def test_export_load_and_m1_predictor_preserve_provenance_and_contract(tmp_path):
    bundle_path = tmp_path / "untrained-m1-handoff-fixture.pt"
    model = GRUTemporalClassifier(GRUConfig(hidden_dim=4))
    export_model_bundle(model, _fixture_normalizer(), bundle_path, metadata={
        "purpose": "untrained infrastructure fixture; not a release model",
        "input_encoding": "raw_with_masks_v1",
    })

    expected_digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    loaded = load_model_bundle(bundle_path)
    assert loaded.normalizer.provenance is not None
    assert loaded.normalizer.provenance.status == "legacy_unverified"
    assert loaded.normalizer.provenance.source_manifest_digest == "synthetic-manifest"

    predictor = BundlePredictor(bundle_path)
    request = PredictRequest(
        feature_channels=list(LOCKED_FEATURE_CHANNELS),
        features=[[0.30, 0.0, 0.0, 0.1, 1.0, 1.0, 1.0, 1.0] for _ in range(60)],
        bundle_id=expected_digest,
    )
    response = predictor.predict(request)
    assert predictor.bundle_id == expected_digest
    assert response.bundle_id == expected_digest
    assert response.class_id in {0, 1, 2}
    assert response.predicted_class in {"ACTIVE", "DROWSY", "SLEEPING"}
    assert sum(response.probabilities.values()) == pytest.approx(1.0)
    assert response.confidence == pytest.approx(
        response.probabilities[response.predicted_class]
    )
    assert response.latency_ms >= 0
