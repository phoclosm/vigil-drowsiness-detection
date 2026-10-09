"""Unit tests for Vigil M2 data schema, labels, and metadata validation."""

from __future__ import annotations

import json
import math

import pytest

from vigil.ml.data.schema import (
    CANONICAL_CLASSES,
    CameraConfig,
    FatigueLabel,
    FeatureConfig,
    PerclosConfig,
    ProtocolConfig,
    SampleRecord,
    SessionMetadata,
    validate_iso8601_utc,
    validate_label_pair,
)


def make_valid_metadata(
    session_id: str = "session_001",
    subject_id: str = "subject_001",
    recording_start_utc: str = "2026-10-06T12:00:00Z",
) -> SessionMetadata:
    """Helper to construct a valid SessionMetadata instance."""
    return SessionMetadata(
        session_id=session_id,
        subject_id=subject_id,
        recording_start_utc=recording_start_utc,
        camera=CameraConfig(
            model="C920-HD-Pro",
            width=1280,
            height=720,
            target_fps=30.0,
        ),
        feature_config=FeatureConfig(
            version="v1.0",
            eye_closure_threshold_ear=0.21,
            perclos=PerclosConfig(
                window_mode="seconds",
                window_value=60.0,
                closure_threshold_ear=0.21,
            ),
        ),
        protocol=ProtocolConfig(
            labeling_method="controlled_annotation",
            notes="Subject monitored during afternoon protocol session.",
        ),
    )


def make_valid_sample(
    frame_index: int = 0,
    timestamp_ms: float = 0.0,
    frame_delta_ms: float | None = None,
    label: str = "ACTIVE",
    label_id: int = 0,
    landmarks_valid: bool = True,
    ear_left: float | None = 0.32,
    ear_right: float | None = 0.32,
    ear_avg: float | None = 0.32,
    fps: float | None = 30.0,
    session_id: str = "session_001",
    subject_id: str = "subject_001",
) -> SampleRecord:
    """Helper to construct a valid SampleRecord instance."""
    left = ear_left if landmarks_valid else None
    right = ear_right if landmarks_valid else None
    avg = ear_avg if landmarks_valid else None
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        frame_delta_ms=frame_delta_ms,
        face_detected=landmarks_valid,
        landmarks_valid=landmarks_valid,
        ear_left=left,
        ear_right=right,
        ear_avg=avg,
        is_eye_closed=False if landmarks_valid else None,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=0.05 if landmarks_valid else None,
        fps=fps,
        label=label,
        label_id=label_id,
    )


class TestFatigueLabels:
    """Tests for FatigueLabel enum and canonical ordering."""

    def test_canonical_classes_ordering(self) -> None:
        assert [c.name for c in CANONICAL_CLASSES] == ["ACTIVE", "DROWSY", "SLEEPING"]
        assert [c.value for c in CANONICAL_CLASSES] == [0, 1, 2]
        assert FatigueLabel.ACTIVE.value == 0
        assert FatigueLabel.DROWSY.value == 1
        assert FatigueLabel.SLEEPING.value == 2
        assert FatigueLabel.AMBIGUOUS.value == -1

    def test_from_name_and_from_id(self) -> None:
        assert FatigueLabel.from_name("ACTIVE") == FatigueLabel.ACTIVE
        assert FatigueLabel.from_name("DROWSY") == FatigueLabel.DROWSY
        assert FatigueLabel.from_name("SLEEPING") == FatigueLabel.SLEEPING
        assert FatigueLabel.from_name("AMBIGUOUS") == FatigueLabel.AMBIGUOUS

        assert FatigueLabel.from_id(0) == FatigueLabel.ACTIVE
        assert FatigueLabel.from_id(1) == FatigueLabel.DROWSY
        assert FatigueLabel.from_id(2) == FatigueLabel.SLEEPING
        assert FatigueLabel.from_id(-1) == FatigueLabel.AMBIGUOUS

    def test_invalid_label_name_or_id(self) -> None:
        with pytest.raises(ValueError, match="Invalid fatigue label"):
            FatigueLabel.from_name("UNKNOWN")

        with pytest.raises(ValueError, match="Invalid fatigue label_id"):
            FatigueLabel.from_id(99)

    def test_validate_label_pair(self) -> None:
        validate_label_pair("ACTIVE", 0)
        validate_label_pair("DROWSY", 1)
        validate_label_pair("SLEEPING", 2)
        validate_label_pair("AMBIGUOUS", -1)

        with pytest.raises(ValueError, match="does not match"):
            validate_label_pair("ACTIVE", 1)
        with pytest.raises(ValueError, match="does not match"):
            validate_label_pair("DROWSY", 0)
        with pytest.raises(ValueError, match="does not match"):
            validate_label_pair("AMBIGUOUS", 0)


class TestSampleRecordValidation:
    """Tests for SampleRecord invariant enforcement."""

    def test_valid_active_record_accepted(self) -> None:
        sample = make_valid_sample(label="ACTIVE", label_id=0)
        assert sample.label == "ACTIVE"
        assert sample.label_id == 0

    def test_valid_drowsy_record_accepted(self) -> None:
        sample = make_valid_sample(
            frame_index=1,
            timestamp_ms=33.3,
            frame_delta_ms=33.3,
            label="DROWSY",
            label_id=1,
        )
        assert sample.label == "DROWSY"
        assert sample.label_id == 1

    def test_valid_sleeping_record_accepted(self) -> None:
        sample = make_valid_sample(
            frame_index=1,
            timestamp_ms=33.3,
            frame_delta_ms=33.3,
            label="SLEEPING",
            label_id=2,
        )
        assert sample.label == "SLEEPING"
        assert sample.label_id == 2

    def test_valid_ambiguous_record_accepted(self) -> None:
        sample = make_valid_sample(
            frame_index=1,
            timestamp_ms=33.3,
            frame_delta_ms=33.3,
            label="AMBIGUOUS",
            label_id=-1,
        )
        assert sample.label == "AMBIGUOUS"
        assert sample.label_id == -1

    def test_canonical_sample_id_accepted(self) -> None:
        # Canonical format is {session_id}_f{frame_index}
        s0 = make_valid_sample(session_id="session_001", frame_index=0)
        assert s0.sample_id == "session_001_f0"

        s17 = make_valid_sample(
            session_id="session_001",
            frame_index=17,
            timestamp_ms=566.6,
            frame_delta_ms=33.3,
        )
        assert s17.sample_id == "session_001_f17"

    def test_incorrect_sample_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match canonical format"):
            SampleRecord(
                sample_id="wrong_id",
                session_id="session_001",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=30.0,
                label="ACTIVE",
                label_id=0,
            )

        with pytest.raises(ValueError, match="does not match canonical format"):
            # Mismatched frame index in sample_id
            SampleRecord(
                sample_id="session_001_f1",
                session_id="session_001",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=30.0,
                label="ACTIVE",
                label_id=0,
            )

    def test_face_detected_false_with_landmarks_valid_true_rejected(self) -> None:
        # landmarks_valid=True requires face_detected=True
        with pytest.raises(
            ValueError,
            match="landmarks_valid cannot be True when face_detected is False",
        ):
            SampleRecord(
                sample_id="session_001_f0",
                session_id="session_001",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=False,  # Inconsistent: face not detected but landmarks true
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=30.0,
                label="ACTIVE",
                label_id=0,
            )

    def test_face_detected_true_with_landmarks_valid_false_accepted(self) -> None:
        # Face detected but landmarks failed validation is valid
        sample = SampleRecord(
            sample_id="session_001_f0",
            session_id="session_001",
            subject_id="subject_001",
            frame_index=0,
            timestamp_ms=0.0,
            frame_delta_ms=None,
            face_detected=True,
            landmarks_valid=False,
            ear_left=None,
            ear_right=None,
            ear_avg=None,
            is_eye_closed=None,
            blink_count=0,
            blink_duration_ms=0.0,
            perclos=None,
            fps=30.0,
            label="ACTIVE",
            label_id=0,
        )
        assert sample.face_detected is True
        assert sample.landmarks_valid is False

    def test_label_id_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match supplied label_id"):
            make_valid_sample(label="ACTIVE", label_id=1)

    def test_missing_ear_values_accepted_when_landmarks_invalid(self) -> None:
        sample = make_valid_sample(
            landmarks_valid=False,
            ear_avg=None,
        )
        assert sample.landmarks_valid is False
        assert sample.ear_left is None
        assert sample.ear_right is None
        assert sample.ear_avg is None
        assert sample.is_eye_closed is None

    def test_missing_ear_incorrectly_zero_rejected_when_landmarks_invalid(self) -> None:
        # Landmarks invalid but caller supplied ear_avg = 0.0 (must be rejected!)
        with pytest.raises(ValueError, match="ear_avg must be None when landmarks_valid is False"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=False,
                landmarks_valid=False,
                ear_left=None,
                ear_right=None,
                ear_avg=0.0,  # Invalid! Coerced zero must be rejected
                is_eye_closed=None,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
                label="ACTIVE",
                label_id=0,
            )

        with pytest.raises(ValueError, match="is_eye_closed must be None when landmarks_valid is False"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=False,
                landmarks_valid=False,
                ear_left=None,
                ear_right=None,
                ear_avg=None,
                is_eye_closed=False,  # Invalid! Must be None when landmarks invalid
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
                label="ACTIVE",
                label_id=0,
            )

    def test_invalid_perclos_rejected(self) -> None:
        with pytest.raises(ValueError, match="perclos must be within"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=-0.05,  # Out of range (< 0.0)
                fps=30.0,
                label="ACTIVE",
                label_id=0,
            )

        with pytest.raises(ValueError, match="perclos must be within"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=1.05,  # Out of range (> 1.0)
                fps=30.0,
                label="ACTIVE",
                label_id=0,
            )

    def test_invalid_negative_blink_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="blink_count must be an integer >= 0"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=-1,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
                label="ACTIVE",
                label_id=0,
            )

    def test_invalid_negative_blink_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="blink_duration_ms must be finite and >= 0.0"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=-15.0,
                perclos=None,
                fps=None,
                label="ACTIVE",
                label_id=0,
            )

    def test_empty_session_subject_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            make_valid_sample(session_id="")

        with pytest.raises(ValueError, match="subject_id must be a non-empty string"):
            make_valid_sample(subject_id="")

        with pytest.raises(ValueError, match="subject_id must not be 'unknown'"):
            make_valid_sample(subject_id="unknown")

    def test_non_finite_floats_rejected(self) -> None:
        with pytest.raises(ValueError, match="timestamp_ms must be finite"):
            make_valid_sample(timestamp_ms=float("nan"))

        with pytest.raises(ValueError, match="ear_avg must be finite"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=float("inf"),
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
                label="ACTIVE",
                label_id=0,
            )

        with pytest.raises(ValueError, match="blink_duration_ms must be finite"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=float("nan"),
                perclos=None,
                fps=None,
                label="ACTIVE",
                label_id=0,
            )

    def test_first_frame_delta_invariants(self) -> None:
        # Frame 0 must have frame_delta_ms is None
        with pytest.raises(ValueError, match="First sample.*frame_delta_ms is None"):
            make_valid_sample(frame_index=0, timestamp_ms=0.0, frame_delta_ms=33.3)

        # Frame 0 must have timestamp_ms == 0.0
        with pytest.raises(ValueError, match="First sample.*timestamp_ms == 0.0"):
            make_valid_sample(frame_index=0, timestamp_ms=10.0, frame_delta_ms=None)

        # Later frame must have non-null frame_delta_ms
        with pytest.raises(ValueError, match="must have a positive frame_delta_ms"):
            make_valid_sample(frame_index=1, timestamp_ms=33.3, frame_delta_ms=None)

    def test_non_boolean_flags_rejected(self) -> None:
        with pytest.raises(TypeError, match="face_detected must be an actual boolean"):
            SampleRecord(
                sample_id="s1_f0",
                session_id="s1",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=1,  # type: ignore[arg-type]
                landmarks_valid=True,
                ear_left=0.3,
                ear_right=0.3,
                ear_avg=0.3,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
                label="ACTIVE",
                label_id=0,
            )

    def test_sample_serialization_roundtrip(self) -> None:
        sample = make_valid_sample(
            frame_index=1,
            timestamp_ms=33.33,
            frame_delta_ms=33.33,
            label="ACTIVE",
            label_id=0,
        )
        json_str = sample.to_json()
        assert "NaN" not in json_str
        assert "Infinity" not in json_str

        loaded = json.loads(json_str)
        assert loaded["frame_index"] == 1
        assert math.isclose(loaded["timestamp_ms"], 33.33, rel_tol=1e-5)

        restored = SampleRecord.from_json(json_str)
        assert restored == sample

    def test_ear_greater_than_max_rejected(self) -> None:
        # EAR values > 0.60 must be rejected
        with pytest.raises(ValueError, match="ear_left must be within \\[0.0, 0.60\\]"):
            make_valid_sample(ear_left=0.61, ear_right=0.30, ear_avg=0.455)

        with pytest.raises(ValueError, match="ear_right must be within \\[0.0, 0.60\\]"):
            make_valid_sample(ear_left=0.30, ear_right=0.65, ear_avg=0.475)

        with pytest.raises(ValueError, match="ear_avg must be within \\[0.0, 0.60\\]"):
            make_valid_sample(ear_left=0.30, ear_right=0.30, ear_avg=0.61)

    def test_ear_less_than_min_rejected(self) -> None:
        # EAR values < 0.0 must be rejected
        with pytest.raises(ValueError, match="ear_left must be within \\[0.0, 0.60\\]"):
            make_valid_sample(ear_left=-0.01, ear_right=0.30, ear_avg=0.145)

        with pytest.raises(ValueError, match="ear_right must be within \\[0.0, 0.60\\]"):
            make_valid_sample(ear_left=0.30, ear_right=-0.05, ear_avg=0.125)

        with pytest.raises(ValueError, match="ear_avg must be within \\[0.0, 0.60\\]"):
            make_valid_sample(ear_left=0.30, ear_right=0.30, ear_avg=-0.01)

    def test_inconsistent_ear_avg_rejected(self) -> None:
        # ear_avg != (ear_left + ear_right) / 2 must be rejected
        with pytest.raises(ValueError, match="Inconsistent ear_avg"):
            make_valid_sample(ear_left=0.30, ear_right=0.30, ear_avg=0.35)

        with pytest.raises(ValueError, match="Inconsistent ear_avg"):
            make_valid_sample(ear_left=0.20, ear_right=0.22, ear_avg=0.25)

    def test_consistent_ear_avg_accepted(self) -> None:
        # Exact average
        sample1 = make_valid_sample(ear_left=0.25, ear_right=0.27, ear_avg=0.26)
        assert sample1.ear_avg == 0.26

        # Average with decimal within tolerance 1e-3
        sample2 = make_valid_sample(ear_left=0.250, ear_right=0.251, ear_avg=0.2505)
        assert math.isclose(sample2.ear_avg, 0.2505)

        # Serialized 3-decimal rounded representation (diff = 0.0005 <= 1e-3)
        sample3 = make_valid_sample(ear_left=0.250, ear_right=0.251, ear_avg=0.251)
        assert sample3.ear_avg == 0.251

    def test_fps_range_invariants(self) -> None:
        # fps == 0 rejected
        with pytest.raises(ValueError, match="fps must be positive"):
            make_valid_sample(fps=0.0)

        # fps < 0 rejected
        with pytest.raises(ValueError, match="fps must be positive"):
            make_valid_sample(fps=-10.0)

        # fps == 120 accepted
        sample_120 = make_valid_sample(fps=120.0)
        assert sample_120.fps == 120.0

        # fps > 120 rejected
        with pytest.raises(ValueError, match="fps must be <= 120.0"):
            make_valid_sample(fps=120.1)

        with pytest.raises(ValueError, match="fps must be <= 120.0"):
            make_valid_sample(fps=240.0)

        # fps is None accepted (optional field)
        sample_none = make_valid_sample(fps=None)
        assert sample_none.fps is None


class TestSessionMetadataValidation:
    """Tests for SessionMetadata validation and serialization."""

    def test_valid_metadata_accepted(self) -> None:
        meta = make_valid_metadata()
        assert meta.session_id == "session_001"
        assert meta.subject_id == "subject_001"

    def test_metadata_serialization_roundtrip(self) -> None:
        meta = make_valid_metadata()
        json_str = meta.to_json()
        restored = SessionMetadata.from_json(json_str)
        assert restored == meta

    def test_invalid_subject_unknown_rejected(self) -> None:
        with pytest.raises(ValueError, match="subject_id must not be 'unknown'"):
            make_valid_metadata(subject_id="unknown")

    def test_invalid_iso_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a valid ISO-8601 string"):
            validate_iso8601_utc("not-a-timestamp")

        with pytest.raises(ValueError, match="must specify UTC timezone"):
            validate_iso8601_utc("2026-10-06T12:00:00+05:30")

    def test_camera_validation(self) -> None:
        with pytest.raises(ValueError, match="Camera width must be a positive integer"):
            CameraConfig(model="Cam", width=-10, height=720, target_fps=30.0).validate()
        with pytest.raises(ValueError, match="Camera target_fps must be positive and finite"):
            CameraConfig(model="Cam", width=1280, height=720, target_fps=0.0).validate()

    def test_perclos_validation(self) -> None:
        with pytest.raises(ValueError, match="Perclos window_mode must be 'seconds' or 'frames'"):
            PerclosConfig(window_mode="minutes", window_value=1.0, closure_threshold_ear=0.2).validate()
        with pytest.raises(ValueError, match="Perclos window_value must be positive and finite"):
            PerclosConfig(window_mode="seconds", window_value=-5.0, closure_threshold_ear=0.2).validate()
