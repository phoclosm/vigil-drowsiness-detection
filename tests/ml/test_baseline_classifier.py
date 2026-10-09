"""Unit tests for Vigil M2 rule-based fatigue baseline classifier."""

from __future__ import annotations

import pytest

from vigil.ml.baseline.classifier import BaselinePrediction, RuleBasedBaseline
from vigil.ml.baseline.config import BaselineConfig
from vigil.ml.data.schema import FatigueLabel, SampleRecord


def make_test_sample(
    frame_index: int = 0,
    timestamp_ms: float = 0.0,
    face_detected: bool = True,
    landmarks_valid: bool = True,
    ear_avg: float | None = 0.30,
    is_eye_closed: bool | None = False,
    blink_duration_ms: float = 0.0,
    perclos: float | None = 0.05,
    session_id: str = "session_001",
    label: str = "ACTIVE",
    label_id: int = 0,
) -> SampleRecord:
    """Helper to generate a valid SampleRecord for classifier testing."""
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id="subject_001",
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=face_detected,
        landmarks_valid=landmarks_valid,
        ear_left=ear_avg,
        ear_right=ear_avg,
        ear_avg=ear_avg,
        is_eye_closed=is_eye_closed,
        blink_count=0,
        blink_duration_ms=blink_duration_ms,
        perclos=perclos,
        fps=30.0,
        label=label,
        label_id=label_id,
    )


class TestBaselineConfig:
    """Tests for BaselineConfig validation and defaults."""

    def test_default_config_is_valid_and_provisional(self) -> None:
        config = BaselineConfig()
        assert config.is_provisional is True
        assert "Empirical calibration is pending" in config.calibration_notes
        assert config.ear_close_threshold == 0.20
        assert config.drowsy_closure_duration_ms == 400.0
        assert config.sleeping_closure_duration_ms == 1500.0
        assert config.drowsy_perclos_threshold == 0.15
        assert config.sleeping_perclos_threshold == 0.35
        assert config.drowsy_blink_duration_ms == 350.0
        assert config.alert_sustain_window_ms == 1000.0
        assert config.missing_feature_fallback == FatigueLabel.ACTIVE

    def test_custom_valid_config(self) -> None:
        config = BaselineConfig(
            ear_close_threshold=0.22,
            drowsy_closure_duration_ms=500.0,
            sleeping_closure_duration_ms=2000.0,
            drowsy_perclos_threshold=0.20,
            sleeping_perclos_threshold=0.40,
            drowsy_blink_duration_ms=400.0,
            alert_sustain_window_ms=1500.0,
            immediate_sleep_alert=False,
            is_provisional=False,
        )
        assert config.is_provisional is False
        assert config.ear_close_threshold == 0.22

    def test_invalid_parameters_rejected(self) -> None:
        with pytest.raises(ValueError, match="ear_close_threshold"):
            BaselineConfig(ear_close_threshold=0.0)

        with pytest.raises(ValueError, match="ear_close_threshold"):
            BaselineConfig(ear_close_threshold=0.65)

        with pytest.raises(ValueError, match="drowsy_closure_duration_ms must be positive"):
            BaselineConfig(drowsy_closure_duration_ms=-10.0)

        with pytest.raises(ValueError, match="sleeping_closure_duration_ms.*strictly greater"):
            BaselineConfig(
                drowsy_closure_duration_ms=1000.0,
                sleeping_closure_duration_ms=500.0,
            )

        with pytest.raises(ValueError, match="drowsy_perclos_threshold"):
            BaselineConfig(drowsy_perclos_threshold=0.0)

        with pytest.raises(ValueError, match="sleeping_perclos_threshold.*strictly greater"):
            BaselineConfig(
                drowsy_perclos_threshold=0.30,
                sleeping_perclos_threshold=0.20,
            )

        with pytest.raises(TypeError, match="must be numeric"):
            BaselineConfig(ear_close_threshold=None)  # type: ignore[arg-type]


class TestRuleBasedBaselineClassification:
    """Tests for RuleBasedBaseline state classification."""

    def test_active_state_on_normal_open_eyes(self) -> None:
        clf = RuleBasedBaseline()
        sample = make_test_sample(
            ear_avg=0.32,
            is_eye_closed=False,
            perclos=0.05,
            blink_duration_ms=150.0,
        )
        pred = clf.classify(sample)

        assert isinstance(pred, BaselinePrediction)
        assert pred.label == FatigueLabel.ACTIVE
        assert pred.label_id == 0
        assert pred.reason == "normal_alert"
        assert pred.closure_duration_ms == 0.0
        assert pred.sustained_fatigue_ms == 0.0
        assert pred.alert_triggered is False
        assert pred.is_provisional is True

    def test_drowsy_state_via_sustained_eye_closure(self) -> None:
        clf = RuleBasedBaseline(
            BaselineConfig(drowsy_closure_duration_ms=400.0, sleeping_closure_duration_ms=1500.0)
        )
        # Frame 0 at t=0ms: eye closes
        s0 = make_test_sample(frame_index=0, timestamp_ms=0.0, ear_avg=0.15, is_eye_closed=True)
        pred0 = clf.classify(s0)
        assert pred0.label == FatigueLabel.ACTIVE  # only 0 ms closed

        # Frame 10 at t=300ms: still closed (< 400ms)
        s1 = make_test_sample(frame_index=10, timestamp_ms=300.0, ear_avg=0.15, is_eye_closed=True)
        pred1 = clf.classify(s1)
        assert pred1.label == FatigueLabel.ACTIVE

        # Frame 15 at t=450ms: closed for 450ms (>= 400ms) -> DROWSY!
        s2 = make_test_sample(frame_index=15, timestamp_ms=450.0, ear_avg=0.15, is_eye_closed=True)
        pred2 = clf.classify(s2)
        assert pred2.label == FatigueLabel.DROWSY
        assert pred2.label_id == 1
        assert pred2.reason == "prolonged_closure_drowsy"
        assert pred2.closure_duration_ms == 450.0

    def test_sleeping_state_via_prolonged_eye_closure(self) -> None:
        clf = RuleBasedBaseline(
            BaselineConfig(drowsy_closure_duration_ms=400.0, sleeping_closure_duration_ms=1500.0)
        )
        # Eye closes at t=0ms
        s0 = make_test_sample(frame_index=0, timestamp_ms=0.0, ear_avg=0.12, is_eye_closed=True)
        clf.classify(s0)

        # Eye stays closed until t=1600ms (>= 1500ms) -> SLEEPING!
        s_sleep = make_test_sample(frame_index=48, timestamp_ms=1600.0, ear_avg=0.12, is_eye_closed=True)
        pred = clf.classify(s_sleep)

        assert pred.label == FatigueLabel.SLEEPING
        assert pred.label_id == 2
        assert pred.reason == "prolonged_closure_sleeping"
        assert pred.closure_duration_ms == 1600.0

    def test_drowsy_state_via_perclos_threshold(self) -> None:
        clf = RuleBasedBaseline(BaselineConfig(drowsy_perclos_threshold=0.15, sleeping_perclos_threshold=0.35))
        # Eyes currently open, but trailing PERCLOS is 0.18 (>= 0.15)
        sample = make_test_sample(
            ear_avg=0.30,
            is_eye_closed=False,
            perclos=0.18,
        )
        pred = clf.classify(sample)
        assert pred.label == FatigueLabel.DROWSY
        assert pred.reason == "elevated_perclos_drowsy"

    def test_sleeping_state_via_high_perclos_threshold(self) -> None:
        clf = RuleBasedBaseline(BaselineConfig(drowsy_perclos_threshold=0.15, sleeping_perclos_threshold=0.35))
        # PERCLOS is 0.40 (>= 0.35) -> SLEEPING
        sample = make_test_sample(
            ear_avg=0.30,
            is_eye_closed=False,
            perclos=0.40,
        )
        pred = clf.classify(sample)
        assert pred.label == FatigueLabel.SLEEPING
        assert pred.reason == "high_perclos_sleeping"

    def test_drowsy_state_via_sluggish_blink_duration(self) -> None:
        clf = RuleBasedBaseline(BaselineConfig(drowsy_blink_duration_ms=350.0))
        # Eyes open, PERCLOS low, but sluggish blink duration = 400ms
        sample = make_test_sample(
            ear_avg=0.30,
            is_eye_closed=False,
            perclos=0.05,
            blink_duration_ms=400.0,
        )
        pred = clf.classify(sample)
        assert pred.label == FatigueLabel.DROWSY
        assert pred.reason == "sluggish_blink_drowsy"

    def test_alert_controller_immediate_sleep_and_sustained_drowsy(self) -> None:
        clf = RuleBasedBaseline(
            BaselineConfig(
                alert_sustain_window_ms=1000.0,
                immediate_sleep_alert=True,
                drowsy_perclos_threshold=0.15,
                sleeping_perclos_threshold=0.35,
            )
        )

        # 1. Immediate alert on SLEEPING
        s_sleep = make_test_sample(perclos=0.40)
        p_sleep = clf.classify(s_sleep)
        assert p_sleep.label == FatigueLabel.SLEEPING
        assert p_sleep.alert_triggered is True

        # Reset for clean drowsy trial
        clf.reset()

        # 2. Drowsy sustained alert: t=0ms drowsy -> alert False
        s_d0 = make_test_sample(frame_index=0, timestamp_ms=0.0, perclos=0.20)
        p_d0 = clf.classify(s_d0)
        assert p_d0.label == FatigueLabel.DROWSY
        assert p_d0.alert_triggered is False

        # t=500ms drowsy (< 1000ms sustain) -> alert False
        s_d1 = make_test_sample(frame_index=15, timestamp_ms=500.0, perclos=0.20)
        p_d1 = clf.classify(s_d1)
        assert p_d1.alert_triggered is False

        # t=1100ms drowsy (>= 1000ms sustain) -> alert True!
        s_d2 = make_test_sample(frame_index=33, timestamp_ms=1100.0, perclos=0.20)
        p_d2 = clf.classify(s_d2)
        assert p_d2.alert_triggered is True
        assert p_d2.sustained_fatigue_ms == 1100.0

        # Returning to ACTIVE resets alert
        s_active = make_test_sample(frame_index=34, timestamp_ms=1133.3, perclos=0.05)
        p_active = clf.classify(s_active)
        assert p_active.label == FatigueLabel.ACTIVE
        assert p_active.alert_triggered is False
        assert p_active.sustained_fatigue_ms == 0.0

    def test_missing_features_handled_cleanly_without_fabrication(self) -> None:
        clf = RuleBasedBaseline()

        # Face missing
        s_no_face = make_test_sample(
            face_detected=False,
            landmarks_valid=False,
            ear_avg=None,
            is_eye_closed=None,
        )
        p_no_face = clf.classify(s_no_face)
        assert p_no_face.label == FatigueLabel.ACTIVE
        assert p_no_face.reason == "missing_features"
        assert p_no_face.closure_duration_ms == 0.0
        assert p_no_face.alert_triggered is False

        # Face detected but landmarks invalid
        s_invalid_lm = make_test_sample(
            face_detected=True,
            landmarks_valid=False,
            ear_avg=None,
            is_eye_closed=None,
        )
        p_invalid_lm = clf.classify(s_invalid_lm)
        assert p_invalid_lm.label == FatigueLabel.ACTIVE
        assert p_invalid_lm.reason == "missing_features"

    def test_predictions_never_use_ground_truth_labels(self) -> None:
        """Verify ground truth label has zero impact on predicted state."""
        clf = RuleBasedBaseline()

        # Feature indicates normal open eyes (ACTIVE)
        # Even if ground truth label is SLEEPING (label_id=2), prediction must be ACTIVE!
        sample_spoofed = make_test_sample(
            ear_avg=0.35,
            is_eye_closed=False,
            perclos=0.02,
            label="SLEEPING",
            label_id=2,
        )
        pred = clf.classify(sample_spoofed)
        assert pred.label == FatigueLabel.ACTIVE
        assert pred.label_id == 0

        # Feature indicates SLEEPING (ear closed, prolonged)
        # Even if ground truth label is ACTIVE (label_id=0), prediction must be SLEEPING!
        clf.reset()
        clf.classify(make_test_sample(frame_index=0, timestamp_ms=0.0, ear_avg=0.10, is_eye_closed=True))
        sample_sleep = make_test_sample(
            frame_index=60,
            timestamp_ms=2000.0,
            ear_avg=0.10,
            is_eye_closed=True,
            label="ACTIVE",
            label_id=0,
        )
        pred_sleep = clf.classify(sample_sleep)
        assert pred_sleep.label == FatigueLabel.SLEEPING
        assert pred_sleep.label_id == 2

    def test_session_boundary_resets_temporal_accumulation(self) -> None:
        """Verify crossing session IDs resets temporal tracking to prevent leakage."""
        clf = RuleBasedBaseline(BaselineConfig(drowsy_closure_duration_ms=400.0))

        # Session A: eye closes at t=0ms, prolonged to 500ms (DROWSY)
        sA0 = make_test_sample(
            session_id="session_A",
            frame_index=0,
            timestamp_ms=0.0,
            ear_avg=0.15,
            is_eye_closed=True,
        )
        clf.classify(sA0)
        sA1 = make_test_sample(
            session_id="session_A",
            frame_index=15,
            timestamp_ms=500.0,
            ear_avg=0.15,
            is_eye_closed=True,
        )
        predA = clf.classify(sA1)
        assert predA.label == FatigueLabel.DROWSY

        # Session B starts: frame 0 at t=0ms in Session B. Should NOT accumulate from Session A!
        sB0 = make_test_sample(
            session_id="session_B",
            frame_index=0,
            timestamp_ms=0.0,
            ear_avg=0.15,
            is_eye_closed=True,
        )
        predB = clf.classify(sB0)
        # Session B restarted closure tracking; duration is 0ms, so ACTIVE (not drowsy)
        assert predB.label == FatigueLabel.ACTIVE
        assert predB.closure_duration_ms == 0.0

    def test_classify_session_helper(self) -> None:
        clf = RuleBasedBaseline()
        samples = [
            make_test_sample(frame_index=0, timestamp_ms=0.0, ear_avg=0.30),
            make_test_sample(frame_index=1, timestamp_ms=33.3, ear_avg=0.30),
        ]
        preds = clf.classify_session(samples)
        assert len(preds) == 2
        assert all(p.label == FatigueLabel.ACTIVE for p in preds)
