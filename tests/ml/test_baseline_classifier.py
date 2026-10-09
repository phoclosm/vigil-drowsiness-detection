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
        assert config.drowsy_closure_duration_ms == 500.0
        assert config.sleeping_closure_duration_ms == 1500.0
        assert config.drowsy_perclos_threshold == 0.20
        assert config.alert_sustain_window_ms == 1000.0
        assert config.immediate_sleep_alert is True

    def test_custom_valid_config(self) -> None:
        config = BaselineConfig(
            drowsy_closure_duration_ms=600.0,
            sleeping_closure_duration_ms=2000.0,
            drowsy_perclos_threshold=0.25,
            alert_sustain_window_ms=1500.0,
            immediate_sleep_alert=False,
            is_provisional=False,
        )
        assert config.is_provisional is False
        assert config.drowsy_closure_duration_ms == 600.0

    def test_invalid_parameters_rejected(self) -> None:
        with pytest.raises(ValueError, match="drowsy_closure_duration_ms must be positive"):
            BaselineConfig(drowsy_closure_duration_ms=-10.0)

        with pytest.raises(ValueError, match="sleeping_closure_duration_ms.*strictly greater"):
            BaselineConfig(
                drowsy_closure_duration_ms=1500.0,
                sleeping_closure_duration_ms=1000.0,
            )

        with pytest.raises(ValueError, match="drowsy_perclos_threshold"):
            BaselineConfig(drowsy_perclos_threshold=0.0)

        with pytest.raises(ValueError, match="drowsy_perclos_threshold"):
            BaselineConfig(drowsy_perclos_threshold=1.5)

        with pytest.raises(TypeError, match="must be numeric"):
            BaselineConfig(drowsy_closure_duration_ms=None)  # type: ignore[arg-type]


class TestRuleBasedBaselineClassification:
    """Tests for RuleBasedBaseline decision rules and contracts."""

    def test_active_state_on_normal_open_eyes(self) -> None:
        clf = RuleBasedBaseline()
        sample = make_test_sample(
            ear_avg=0.32,
            is_eye_closed=False,
            perclos=0.05,
        )
        pred = clf.classify(sample)

        assert isinstance(pred, BaselinePrediction)
        assert pred.has_prediction is True
        assert pred.label == FatigueLabel.ACTIVE
        assert pred.label_id == 0
        assert pred.reason == "normal_alert"
        assert pred.closure_duration_ms == 0.0
        assert pred.sustained_fatigue_ms == 0.0
        assert pred.alert_triggered is False
        assert pred.is_provisional is True

    def test_uses_supplied_m1_eye_closure_indicator_not_ear_avg(self) -> None:
        """Verify M2 strictly uses sample.is_eye_closed and does not re-derive from ear_avg."""
        clf = RuleBasedBaseline()

        # Case 1: ear_avg is very low (e.g. 0.10) but M1 says eye is NOT closed (is_eye_closed=False)
        # M2 must trust M1 and NOT treat eye as closed!
        sample_open_low_ear = make_test_sample(
            ear_avg=0.10,
            is_eye_closed=False,
            perclos=0.05,
        )
        pred1 = clf.classify(sample_open_low_ear)
        assert pred1.label == FatigueLabel.ACTIVE
        assert pred1.closure_duration_ms == 0.0

        # Case 2: ear_avg is high (0.35) but M1 says eye IS closed (is_eye_closed=True)
        # M2 must treat eye as closed based on M1's indicator
        clf.reset()
        sample_closed_high_ear = make_test_sample(
            frame_index=0,
            timestamp_ms=0.0,
            ear_avg=0.35,
            is_eye_closed=True,
            perclos=0.05,
        )
        pred2 = clf.classify(sample_closed_high_ear)
        # Closed at t=0ms -> duration is 0ms -> ACTIVE at frame 0, but closure tracking started
        assert pred2.label == FatigueLabel.ACTIVE
        assert clf._closure_start_timestamp_ms == 0.0

        # After 600ms of closure -> DROWSY
        sample_closed_later = make_test_sample(
            frame_index=18,
            timestamp_ms=600.0,
            ear_avg=0.35,
            is_eye_closed=True,
            perclos=0.05,
        )
        pred3 = clf.classify(sample_closed_later)
        assert pred3.label == FatigueLabel.DROWSY
        assert pred3.closure_duration_ms == 600.0

    def test_missing_features_return_abstention_never_active(self) -> None:
        """CRITICAL: Missing features must return explicit no-prediction, not ACTIVE."""
        clf = RuleBasedBaseline()

        # 1. Face not detected
        s_no_face = make_test_sample(
            face_detected=False,
            landmarks_valid=False,
            ear_avg=None,
            is_eye_closed=None,
        )
        p_no_face = clf.classify(s_no_face)
        assert p_no_face.has_prediction is False
        assert p_no_face.label is None
        assert p_no_face.label_id is None
        assert p_no_face.reason == "missing_features"
        assert p_no_face.closure_duration_ms == 0.0
        assert p_no_face.alert_triggered is False

        # 2. Face detected but landmarks invalid
        s_invalid_lm = make_test_sample(
            face_detected=True,
            landmarks_valid=False,
            ear_avg=None,
            is_eye_closed=None,
        )
        p_invalid_lm = clf.classify(s_invalid_lm)
        assert p_invalid_lm.has_prediction is False
        assert p_invalid_lm.label is None
        assert p_invalid_lm.label_id is None
        assert p_invalid_lm.reason == "missing_features"

        # 3. is_eye_closed is None (unavailable observation)
        s_no_closure = make_test_sample(
            face_detected=True,
            landmarks_valid=True,
            ear_avg=0.30,
            is_eye_closed=None,
        )
        p_no_closure = clf.classify(s_no_closure)
        assert p_no_closure.has_prediction is False
        assert p_no_closure.label is None
        assert p_no_closure.reason == "missing_features"

    def test_sleeping_state_requires_closed_eyes_and_duration_1500ms(self) -> None:
        clf = RuleBasedBaseline()

        # Start eye closure at t=0ms
        s0 = make_test_sample(frame_index=0, timestamp_ms=0.0, is_eye_closed=True)
        clf.classify(s0)

        # Eye stays closed until t=1500ms -> SLEEPING!
        s_sleep = make_test_sample(frame_index=45, timestamp_ms=1500.0, is_eye_closed=True)
        pred = clf.classify(s_sleep)

        assert pred.label == FatigueLabel.SLEEPING
        assert pred.label_id == 2
        assert pred.reason == "prolonged_closure_sleeping"
        assert pred.closure_duration_ms == 1500.0
        assert pred.alert_triggered is True

    def test_open_eyes_with_high_perclos_does_not_trigger_sleeping(self) -> None:
        """CRITICAL: SLEEPING is never triggered solely from PERCLOS when eyes are open."""
        clf = RuleBasedBaseline()

        # Eyes are open (is_eye_closed=False), but PERCLOS is very high (0.45 >= 0.20)
        # Must predict DROWSY, NEVER SLEEPING!
        s_high_perclos = make_test_sample(
            ear_avg=0.32,
            is_eye_closed=False,
            perclos=0.45,
        )
        pred = clf.classify(s_high_perclos)
        assert pred.label == FatigueLabel.DROWSY
        assert pred.label_id == 1
        assert pred.reason == "elevated_perclos_drowsy"

    def test_drowsy_state_via_perclos_threshold_20_percent(self) -> None:
        clf = RuleBasedBaseline()

        # PERCLOS = 0.19 (< 0.20) with open eyes -> ACTIVE
        s_below = make_test_sample(is_eye_closed=False, perclos=0.19)
        assert clf.classify(s_below).label == FatigueLabel.ACTIVE

        # PERCLOS = 0.20 (>= 0.20) with open eyes -> DROWSY
        s_at = make_test_sample(is_eye_closed=False, perclos=0.20)
        pred_at = clf.classify(s_at)
        assert pred_at.label == FatigueLabel.DROWSY
        assert pred_at.reason == "elevated_perclos_drowsy"

    def test_drowsy_state_via_continuous_closure_500ms(self) -> None:
        clf = RuleBasedBaseline()

        # Eyes close at t=0ms
        clf.classify(make_test_sample(frame_index=0, timestamp_ms=0.0, is_eye_closed=True))

        # At t=490ms (< 500ms): not yet drowsy by duration
        s_490 = make_test_sample(frame_index=15, timestamp_ms=490.0, is_eye_closed=True, perclos=0.05)
        assert clf.classify(s_490).label == FatigueLabel.ACTIVE

        # At t=500ms (>= 500ms): DROWSY!
        s_500 = make_test_sample(frame_index=16, timestamp_ms=500.0, is_eye_closed=True, perclos=0.05)
        pred_500 = clf.classify(s_500)
        assert pred_500.label == FatigueLabel.DROWSY
        assert pred_500.reason == "prolonged_closure_drowsy"
        assert pred_500.closure_duration_ms == 500.0

    def test_alert_controller_timing_and_reset(self) -> None:
        clf = RuleBasedBaseline(BaselineConfig(alert_sustain_window_ms=1000.0))

        # Drowsy at t=0ms: alert False
        s0 = make_test_sample(frame_index=0, timestamp_ms=0.0, perclos=0.25)
        p0 = clf.classify(s0)
        assert p0.label == FatigueLabel.DROWSY
        assert p0.alert_triggered is False

        # Sustained drowsy at t=990ms (< 1000ms): alert False
        s1 = make_test_sample(frame_index=30, timestamp_ms=990.0, perclos=0.25)
        p1 = clf.classify(s1)
        assert p1.alert_triggered is False

        # Sustained drowsy at t=1000ms (>= 1000ms): alert True!
        s2 = make_test_sample(frame_index=31, timestamp_ms=1000.0, perclos=0.25)
        p2 = clf.classify(s2)
        assert p2.alert_triggered is True
        assert p2.sustained_fatigue_ms == 1000.0

        # Returning to ACTIVE resets alert immediately
        s_active = make_test_sample(frame_index=32, timestamp_ms=1033.3, perclos=0.05)
        p_active = clf.classify(s_active)
        assert p_active.label == FatigueLabel.ACTIVE
        assert p_active.alert_triggered is False
        assert p_active.sustained_fatigue_ms == 0.0

    def test_ground_truth_labels_are_never_referenced(self) -> None:
        """Verify ground truth label has zero impact on inference."""
        clf = RuleBasedBaseline()

        # Open eyes (ACTIVE feature state) with spoofed ground-truth SLEEPING
        s_spoof = make_test_sample(
            is_eye_closed=False,
            perclos=0.05,
            label="SLEEPING",
            label_id=2,
        )
        p = clf.classify(s_spoof)
        assert p.label == FatigueLabel.ACTIVE

    def test_session_boundary_resets_temporal_state(self) -> None:
        clf = RuleBasedBaseline()

        # Session A: eye closed at t=0ms, then 600ms (DROWSY)
        sA0 = make_test_sample(session_id="session_A", frame_index=0, timestamp_ms=0.0, is_eye_closed=True)
        clf.classify(sA0)
        sA1 = make_test_sample(session_id="session_A", frame_index=18, timestamp_ms=600.0, is_eye_closed=True)
        assert clf.classify(sA1).label == FatigueLabel.DROWSY

        # Session B starts: eye closed at t=0ms. Should NOT accumulate from Session A!
        sB0 = make_test_sample(session_id="session_B", frame_index=0, timestamp_ms=0.0, is_eye_closed=True)
        pB = clf.classify(sB0)
        assert pB.label == FatigueLabel.ACTIVE
        assert pB.closure_duration_ms == 0.0
