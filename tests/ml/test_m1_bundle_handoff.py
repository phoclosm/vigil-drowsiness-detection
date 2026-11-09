"""M2-owned GRU bundle handoff contract tests.

These tests use an untrained infrastructure fixture and make no model-quality
or accuracy claim.
"""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from vigil.cloud.bundle import BundlePredictor  # noqa: E402
from vigil.cloud.contracts import PredictRequest  # noqa: E402
from vigil.ml.data.manifest import ApprovedDataManifest, ManifestRecording  # noqa: E402
from vigil.ml.data.schema import SampleRecord  # noqa: E402
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS  # noqa: E402
from vigil.ml.dataset.normalizer import FeatureNormalizer  # noqa: E402
from vigil.ml.models.export import export_model_bundle, load_model_bundle  # noqa: E402
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier  # noqa: E402
from vigil.ml.provenance_types import (  # noqa: E402
    NormalizerProvenance,
    TrustedPartition,
    manifest_digest,
)


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


def _records(session_id: str, subject_id: str) -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id=f"{session_id}_f{index}",
            session_id=session_id,
            subject_id=subject_id,
            frame_index=index,
            timestamp_ms=float(index * 33.3),
            frame_delta_ms=None if index == 0 else 33.3,
            face_detected=True,
            landmarks_valid=True,
            ear_left=0.3,
            ear_right=0.3,
            ear_avg=0.3,
            is_eye_closed=False,
            blink_count=0,
            blink_duration_ms=0.0,
            perclos=0.1,
            fps=30.0,
            label="ACTIVE",
            label_id=0,
        )
        for index in range(60)
    ]


def _verified_normalizer() -> FeatureNormalizer:
    approved_manifest = ApprovedDataManifest(
        dataset_name="fixture",
        dataset_version="1",
        source_uri="fixture://m1",
        authorization_evidence="fixture-only",
        consent_evidence="fixture-only",
        label_mapping={
            "ACTIVE": "ACTIVE",
            "DROWSY": "DROWSY",
            "SLEEPING": "SLEEPING",
            "AMBIGUOUS": "AMBIGUOUS",
        },
        split_strategy="subject",
        recordings=(
            ManifestRecording("train", "train.jsonl", "train-subject", "train", "train"),
            ManifestRecording("validation", "validation.jsonl", "val-subject", "validation", "validation"),
            ManifestRecording("held-out", "held-out.jsonl", "test-subject", "held-out", "held_out"),
        ),
    )
    trusted = TrustedPartition.from_manifest(
        approved_manifest,
        train_records=_records("train", "train-subject"),
        validation_records=_records("validation", "val-subject"),
        held_out_records=_records("held-out", "test-subject"),
    )
    return FeatureNormalizer().fit(
        list(trusted.train_records),
        manifest_digest=manifest_digest(approved_manifest),
        trusted_partition=trusted,
    )


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


def test_verified_factory_provenance_round_trips(tmp_path):
    bundle_path = tmp_path / "verified-lineage-fixture.pt"
    normalizer = _verified_normalizer()
    export_model_bundle(
        GRUTemporalClassifier(GRUConfig(hidden_dim=4)),
        normalizer,
        bundle_path,
        metadata={"purpose": "synthetic infrastructure fixture"},
    )
    loaded = load_model_bundle(bundle_path)
    assert loaded.normalizer.provenance == normalizer.provenance
    assert loaded.normalizer.provenance_status == "verified"


def test_legacy_bundle_without_provenance_remains_unverified(tmp_path):
    bundle_path = tmp_path / "legacy-lineage-fixture.pt"
    export_model_bundle(
        GRUTemporalClassifier(GRUConfig(hidden_dim=4)),
        _fixture_normalizer(),
        bundle_path,
    )
    import torch

    raw = torch.load(bundle_path, weights_only=False)
    raw["normalizer"].pop("provenance", None)
    torch.save(raw, bundle_path)
    loaded = load_model_bundle(bundle_path)
    assert loaded.normalizer.provenance is None
    assert loaded.normalizer.provenance_status == "legacy_unverified"


def test_approved_release_requires_verified_lineage(tmp_path):
    metadata = {
        "release_metadata": {
            "experiment_identity": {"schema_version": "1", "identity_digest": "0" * 64},
            "dataset_identity": {
                "dataset_name": "fixture",
                "dataset_version": "1",
                "manifest_digest": "1" * 64,
                "label_mapping_digest": "2" * 64,
                "partition_digest": "3" * 64,
            },
            "model_identity": {"model_type": "gru", "config_digest": "4" * 64},
            "checkpoint": {
                "checkpoint_sha256": "5" * 64,
                "artifact_sha256": "6" * 64,
                "best_epoch": 1,
            },
            "source_revision": "a" * 40,
            "is_synthetic": False,
            "approval_status": "approved",
            "approval_evidence_ref": "human-review-record",
            "normalizer_provenance": {
                "status": "verified",
                "source_manifest_digest": "1" * 64,
                "training_partition_digest": "3" * 64,
            },
        }
    }
    model = GRUTemporalClassifier(GRUConfig(hidden_dim=4))
    with pytest.raises(ValueError, match="model-development provenance"):
        export_model_bundle(model, _fixture_normalizer(), tmp_path / "unverified.pt", metadata)

    model.training_provenance = {
        "strategy": "subject",
        "train_subjects": ["train-subject"],
        "val_subjects": ["val-subject"],
        "train_sessions": ["train"],
        "val_sessions": ["validation"],
    }
    with pytest.raises(ValueError, match="verified normalizer provenance"):
        export_model_bundle(model, _fixture_normalizer(), tmp_path / "unverified-normalizer.pt", metadata)

    bundle_path = tmp_path / "approved-lineage-only.pt"
    export_model_bundle(model, _verified_normalizer(), bundle_path, metadata)
    loaded = load_model_bundle(bundle_path)
    assert loaded.metadata["release_metadata"]["approval_status"] == "approved"


@pytest.mark.parametrize(
    "release_update",
    [
        {"dataset_identity": {"dataset_name": "only-name"}},
        {"source_revision": "not-a-git-sha"},
        {"checkpoint": {"checkpoint_sha256": "0" * 64, "artifact_sha256": "bad", "best_epoch": 1}},
        {"is_synthetic": True, "approval_status": "approved"},
    ],
)
def test_release_metadata_rejects_incomplete_or_unapproved_claims(tmp_path, release_update):
    metadata = {
        "release_metadata": {
            "experiment_identity": {"schema_version": "1", "identity_digest": "0" * 64},
            "dataset_identity": {
                "dataset_name": "fixture",
                "dataset_version": "1",
                "manifest_digest": "1" * 64,
                "label_mapping_digest": "2" * 64,
                "partition_digest": "3" * 64,
            },
            "model_identity": {"model_type": "gru", "config_digest": "4" * 64},
            "checkpoint": {
                "checkpoint_sha256": "5" * 64,
                "artifact_sha256": "6" * 64,
                "best_epoch": 1,
            },
            "source_revision": "a" * 40,
            "is_synthetic": False,
            "approval_status": "pending",
        }
    }
    metadata["release_metadata"].update(release_update)
    with pytest.raises((TypeError, ValueError)):
        export_model_bundle(
            GRUTemporalClassifier(GRUConfig(hidden_dim=4)),
            _fixture_normalizer(),
            tmp_path / "invalid-release.pt",
            metadata=metadata,
        )
