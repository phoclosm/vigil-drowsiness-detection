"""Unit tests for Vigil M2 SessionRecorder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigil.ml.data.recorder import SessionExistsError, SessionRecorder
from vigil.ml.data.schema import (
    CameraConfig,
    FeatureConfig,
    PerclosConfig,
    ProtocolConfig,
    SampleRecord,
    SessionMetadata,
)


def make_metadata(
    session_id: str = "session_001",
    subject_id: str = "subject_001",
) -> SessionMetadata:
    """Helper to create standard test SessionMetadata."""
    return SessionMetadata(
        session_id=session_id,
        subject_id=subject_id,
        recording_start_utc="2026-10-06T10:00:00Z",
        camera=CameraConfig(
            model="Test-Cam",
            width=1920,
            height=1080,
            target_fps=30.0,
        ),
        feature_config=FeatureConfig(
            version="v1.0",
            eye_closure_threshold_ear=0.20,
            perclos=PerclosConfig(
                window_mode="seconds",
                window_value=60.0,
                closure_threshold_ear=0.20,
            ),
        ),
        protocol=ProtocolConfig(
            labeling_method="controlled_annotation",
            notes="Unit test recording session.",
        ),
    )


def make_sample(
    frame_index: int,
    timestamp_ms: float,
    frame_delta_ms: float | None = None,
    session_id: str = "session_001",
    subject_id: str = "subject_001",
    landmarks_valid: bool = True,
    ear_avg: float | None = 0.30,
    label: str = "ACTIVE",
    label_id: int = 0,
) -> SampleRecord:
    """Helper to create a valid SampleRecord."""
    ear_val = ear_avg if landmarks_valid else None
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        frame_delta_ms=frame_delta_ms,
        face_detected=landmarks_valid,
        landmarks_valid=landmarks_valid,
        ear_left=ear_val,
        ear_right=ear_val,
        ear_avg=ear_val,
        is_eye_closed=False if landmarks_valid else None,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=0.04 if landmarks_valid else None,
        fps=30.0,
        label=label,
        label_id=label_id,
    )


class TestSessionRecorder:
    """Tests covering SessionRecorder directory creation, invariants, and serialization."""

    def test_session_directory_and_metadata_created(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        recorder = SessionRecorder(session_dir=tmp_path / metadata.session_id, metadata=metadata)
        recorder.close()

        assert recorder.session_dir.exists()
        assert recorder.session_dir.is_dir()
        assert recorder.meta_path.exists()
        assert recorder.samples_path.exists()

        # Verify metadata is valid JSON matching metadata
        meta_content = json.loads(recorder.meta_path.read_text(encoding="utf-8"))
        assert meta_content["session_id"] == "session_001"
        assert meta_content["subject_id"] == "subject_001"
        assert meta_content["camera"]["width"] == 1920
        assert meta_content["feature_config"]["perclos"]["window_mode"] == "seconds"

    def test_reopening_non_empty_session_rejected(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        # 1. Record a valid sample and close the session
        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            recorder.append(make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None))

        # 2. Reopening non-empty session must be rejected with SessionExistsError
        with pytest.raises(
            (SessionExistsError, FileExistsError, ValueError),
            match="already contains recorded samples.*Reopening non-empty recording sessions is not permitted",
        ):
            SessionRecorder(session_dir=session_dir, metadata=metadata)

    def test_reopening_empty_session_with_matching_metadata_permitted(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        # 1. Initialize session without writing any samples
        recorder = SessionRecorder(session_dir=session_dir, metadata=metadata)
        recorder.close()

        # 2. Reopening an empty session with matching metadata is allowed
        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder2:
            recorder2.append(make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None))
            assert recorder2.sample_count == 1

    def test_incompatible_existing_metadata_rejected(self, tmp_path: Path) -> None:
        metadata_a = make_metadata(session_id="session_001", subject_id="subject_001")
        session_dir = tmp_path / metadata_a.session_id

        # Create session directory and write metadata A
        session_dir.mkdir(parents=True, exist_ok=True)
        session_dir.joinpath("session_meta.json").write_text(
            metadata_a.to_json(), encoding="utf-8"
        )

        # Attempt to open recorder with conflicting metadata B (different subject_id)
        metadata_b = make_metadata(session_id="session_001", subject_id="subject_002")
        with pytest.raises(
            ValueError,
            match="Existing metadata.*does not match supplied session metadata",
        ):
            SessionRecorder(session_dir=session_dir, metadata=metadata_b)

    def test_samples_file_format_and_ordering(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            s0 = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
            s1 = make_sample(frame_index=1, timestamp_ms=33.3, frame_delta_ms=33.3)
            s2 = make_sample(frame_index=2, timestamp_ms=66.6, frame_delta_ms=33.3)
            recorder.append(s0)
            recorder.append(s1)
            recorder.append(s2)

        lines = session_dir.joinpath("samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

        # Verify each line is valid JSON and preserves ordering
        parsed = [json.loads(line) for line in lines]
        assert [p["frame_index"] for p in parsed] == [0, 1, 2]
        assert [p["timestamp_ms"] for p in parsed] == [0.0, 33.3, 66.6]
        assert parsed[0]["frame_delta_ms"] is None
        assert parsed[1]["frame_delta_ms"] == 33.3

    def test_json_null_used_for_missing_feature_values(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            s0 = make_sample(
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                landmarks_valid=False,
                ear_avg=None,
            )
            recorder.append(s0)

        raw_line = session_dir.joinpath("samples.jsonl").read_text(encoding="utf-8").strip()
        assert '"ear_left": null' in raw_line
        assert '"ear_right": null' in raw_line
        assert '"ear_avg": null' in raw_line
        assert '"is_eye_closed": null' in raw_line
        assert '"perclos": null' in raw_line
        assert '"frame_delta_ms": null' in raw_line
        assert "NaN" not in raw_line

    def test_first_sample_frame_delta_invariants(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            # First sample must have frame_index=0
            bad_s0 = make_sample(frame_index=1, timestamp_ms=33.3, frame_delta_ms=33.3)
            with pytest.raises(ValueError, match="First sample must have frame_index == 0"):
                recorder.append(bad_s0)

    def test_later_sample_delta_validated_against_timestamp_difference(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            s0 = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
            recorder.append(s0)

            # Sample has timestamp 40.0, so expected delta is 40.0. If sample says 30.0, reject!
            inconsistent_s1 = make_sample(frame_index=1, timestamp_ms=40.0, frame_delta_ms=30.0)
            with pytest.raises(ValueError, match="does not match timestamp difference"):
                recorder.append(inconsistent_s1)

    def test_duplicate_frame_index_rejected(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            s0 = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
            s1 = make_sample(frame_index=1, timestamp_ms=33.3, frame_delta_ms=33.3)
            recorder.append(s0)
            recorder.append(s1)

            # Attempt to append frame_index=1 again
            dup_s1 = make_sample(frame_index=1, timestamp_ms=66.6, frame_delta_ms=33.3)
            with pytest.raises(ValueError, match="Duplicate frame_index 1 encountered"):
                recorder.append(dup_s1)

    def test_frame_index_going_backwards_rejected(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            s0 = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
            s5 = make_sample(frame_index=5, timestamp_ms=100.0, frame_delta_ms=100.0)
            recorder.append(s0)
            recorder.append(s5)

            # Attempt to append frame_index=3 (< 5)
            back_s = make_sample(frame_index=3, timestamp_ms=133.3, frame_delta_ms=33.3)
            with pytest.raises(ValueError, match="frame_index went backwards"):
                recorder.append(back_s)

    def test_non_monotonic_timestamp_rejected(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            s0 = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
            s1 = make_sample(frame_index=1, timestamp_ms=50.0, frame_delta_ms=50.0)
            recorder.append(s0)
            recorder.append(s1)

            # Timestamp equal to previous
            equal_s = make_sample(frame_index=2, timestamp_ms=50.0, frame_delta_ms=10.0)
            with pytest.raises(ValueError, match="timestamp_ms must strictly increase"):
                recorder.append(equal_s)

            # Timestamp less than previous
            back_ts_s = make_sample(frame_index=2, timestamp_ms=40.0, frame_delta_ms=10.0)
            with pytest.raises(ValueError, match="timestamp_ms must strictly increase"):
                recorder.append(back_ts_s)

    def test_mismatched_session_id_rejected(self, tmp_path: Path) -> None:
        metadata = make_metadata(session_id="session_A")
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            wrong_sample = make_sample(
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                session_id="session_B",
            )
            with pytest.raises(ValueError, match="Sample session_id 'session_B' does not match"):
                recorder.append(wrong_sample)

    def test_mismatched_subject_id_rejected(self, tmp_path: Path) -> None:
        metadata = make_metadata(subject_id="subject_001")
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            wrong_sample = make_sample(
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                subject_id="subject_002",
            )
            with pytest.raises(ValueError, match="Sample subject_id 'subject_002' does not match"):
                recorder.append(wrong_sample)

    def test_large_positive_delta_preserved_truthfully(self, tmp_path: Path) -> None:
        # A real camera stall/delay causes a 1500 ms gap
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            s0 = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
            s1 = make_sample(frame_index=1, timestamp_ms=1500.0, frame_delta_ms=1500.0)
            recorder.append(s0)
            recorder.append(s1)

        lines = session_dir.joinpath("samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
        p1 = json.loads(lines[1])
        assert p1["frame_delta_ms"] == 1500.0
        assert p1["timestamp_ms"] == 1500.0

    def test_closed_recorder_rejects_further_writes(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        recorder = SessionRecorder(session_dir=session_dir, metadata=metadata)
        s0 = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
        recorder.append(s0)
        recorder.close()

        assert recorder.is_closed is True
        s1 = make_sample(frame_index=1, timestamp_ms=33.3, frame_delta_ms=33.3)
        with pytest.raises(RuntimeError, match="Cannot append to a closed SessionRecorder"):
            recorder.append(s1)

    def test_context_manager_closes_cleanly(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            recorder.append(make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None))
            assert recorder.is_closed is False

        assert recorder.is_closed is True
        assert recorder.sample_count == 1

    def test_deterministic_serialization_key_structure(self, tmp_path: Path) -> None:
        metadata = make_metadata()
        session_dir = tmp_path / metadata.session_id

        sample = make_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=None)
        with SessionRecorder(session_dir=session_dir, metadata=metadata) as recorder:
            recorder.append(sample)

        line = session_dir.joinpath("samples.jsonl").read_text(encoding="utf-8").strip()
        data = json.loads(line)

        expected_keys = [
            "sample_id",
            "session_id",
            "subject_id",
            "frame_index",
            "timestamp_ms",
            "frame_delta_ms",
            "face_detected",
            "landmarks_valid",
            "ear_left",
            "ear_right",
            "ear_avg",
            "is_eye_closed",
            "blink_count",
            "blink_duration_ms",
            "perclos",
            "fps",
            "label",
            "label_id",
        ]
        assert list(data.keys()) == expected_keys
