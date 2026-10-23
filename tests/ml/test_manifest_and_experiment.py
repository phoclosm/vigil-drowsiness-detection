"""Regression tests for approved-data manifests and the Day 1 experiment harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigil.ml.data.manifest import ApprovedDataManifest, ManifestRecording
from vigil.ml.data.schema import SampleRecord
from vigil.ml.experiments.runner import run_development_experiment
from vigil.ml.training.config import TrainingConfig


def make_record(frame_index: int, session_id: str, subject_id: str) -> SampleRecord:
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=float(frame_index * 33),
        frame_delta_ms=None if frame_index == 0 else 33.0,
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


def make_manifest(*recordings: ManifestRecording) -> ApprovedDataManifest:
    return ApprovedDataManifest(
        dataset_name="approved-stationary-dev",
        dataset_version="2026-10-23",
        source_uri="internal://approved/stationary",
        authorization_evidence="approval-ticket-001",
        consent_evidence="consent-register-001",
        label_mapping={
            "ACTIVE": "alert",
            "DROWSY": "drowsy",
            "SLEEPING": "sleeping",
            "AMBIGUOUS": "uncertain",
        },
        split_strategy="subject",
        recordings=recordings,
    )


def test_manifest_round_trip_and_group_validation() -> None:
    manifest = make_manifest(
        ManifestRecording("r1", "train/s1.jsonl", "subject_1", "session_1", "train"),
        ManifestRecording("r2", "val/s2.jsonl", "subject_2", "session_2", "validation"),
        ManifestRecording("r3", "test/s3.jsonl", "subject_3", "session_3", "held_out"),
    )

    restored = ApprovedDataManifest.from_json(manifest.to_json())

    assert restored.to_dict() == manifest.to_dict()
    assert len(restored.recordings_for_split("held_out")) == 1


def test_manifest_rejects_cross_split_subjects_and_unsafe_paths() -> None:
    with pytest.raises(ValueError, match="multiple manifest splits"):
        make_manifest(
            ManifestRecording("r1", "s1.jsonl", "subject_1", "session_1", "train"),
            ManifestRecording("r2", "s2.jsonl", "subject_1", "session_2", "validation"),
        )

    with pytest.raises(ValueError, match="relative path"):
        ManifestRecording("r1", "../s1.jsonl", "subject_1", "session_1", "train")


def test_manifest_loader_validates_identity_and_blank_lines(tmp_path: Path) -> None:
    recording = ManifestRecording("r1", "samples.jsonl", "subject_1", "session_1", "train")
    manifest = make_manifest(
        recording,
        ManifestRecording("r2", "held.jsonl", "subject_2", "session_2", "held_out"),
    )
    (tmp_path / "samples.jsonl").write_text(
        "\n" + make_record(0, "session_1", "subject_1").to_json() + "\n",
        encoding="utf-8",
    )

    records = manifest.load_records(tmp_path, "train")

    assert len(records) == 1
    assert records[0].session_id == "session_1"


def test_development_runner_records_manifest_and_training_metadata() -> None:
    records = [
        make_record(frame, "session_1", "subject_1")
        for frame in range(80)
    ] + [
        make_record(frame, "session_2", "subject_2")
        for frame in range(80)
    ]
    manifest = make_manifest(
        ManifestRecording("r1", "s1.jsonl", "subject_1", "session_1", "train"),
        ManifestRecording("r2", "s2.jsonl", "subject_2", "session_2", "validation"),
        ManifestRecording("r3", "s3.jsonl", "subject_3", "session_3", "held_out"),
    )

    _, experiment = run_development_experiment(
        records,
        manifest,
        git_sha="abc123",
        config=TrainingConfig(max_epochs=1, hidden_dim=4),
        is_synthetic=True,
    )

    serialized = json.loads(experiment.to_json())
    assert serialized["git_sha"] == "abc123"
    assert serialized["manifest"]["dataset_version"] == "2026-10-23"
    assert serialized["epoch_count"] == 1
    assert serialized["best_checkpoint_epoch"] == 1
    assert serialized["is_synthetic"] is True


def test_development_runner_rejects_held_out_records() -> None:
    manifest = make_manifest(
        ManifestRecording("r1", "s1.jsonl", "subject_1", "session_1", "held_out"),
    )
    with pytest.raises(ValueError, match="Held-out session"):
        run_development_experiment(
            [make_record(0, "session_1", "subject_1")],
            manifest,
            git_sha="abc123",
        )
