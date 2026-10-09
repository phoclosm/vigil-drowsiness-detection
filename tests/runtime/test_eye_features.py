"""Deterministic tests for EAR, blink, and temporal eye features."""

import pytest

from vigil.runtime.eye_features import (
    BlinkTracker,
    BlinkTrackerConfig,
    EyeFeatureError,
    EyeFeatureStream,
    EyeFeatureStreamConfig,
    calculate_eye_aspect_ratio,
    calculate_eye_aspect_ratios,
)
from vigil.runtime.landmarks import EyeLandmarks, NormalizedLandmark


def make_eye(ear: float) -> tuple[NormalizedLandmark, ...]:
    """Build six ordered points with the requested two-dimensional EAR."""
    half_vertical = 2.0 * ear
    return (
        NormalizedLandmark(0.0, 0.0, 0.0),
        NormalizedLandmark(1.0, half_vertical, 0.0),
        NormalizedLandmark(3.0, half_vertical, 0.0),
        NormalizedLandmark(4.0, 0.0, 0.0),
        NormalizedLandmark(3.0, -half_vertical, 0.0),
        NormalizedLandmark(1.0, -half_vertical, 0.0),
    )


def make_eyes(left: float, right: float | None = None) -> EyeLandmarks:
    """Build a two-eye landmark payload with known EAR values."""
    return EyeLandmarks(
        left=make_eye(left),
        right=make_eye(left if right is None else right),
    )


def make_stream(*, window: int = 3) -> EyeFeatureStream:
    return EyeFeatureStream(
        EyeFeatureStreamConfig(
            closure_threshold_ear=0.20,
            perclos_window_frames=window,
            minimum_blink_duration_ms=50.0,
        )
    )


def test_calculates_known_eye_aspect_ratio() -> None:
    assert calculate_eye_aspect_ratio(make_eye(0.25)) == pytest.approx(0.25)


def test_calculates_each_eye_and_mean() -> None:
    ratios = calculate_eye_aspect_ratios(make_eyes(0.20, 0.30))

    assert ratios.left == pytest.approx(0.20)
    assert ratios.right == pytest.approx(0.30)
    assert ratios.average == pytest.approx(0.25)


def test_ear_requires_six_points() -> None:
    with pytest.raises(ValueError, match="exactly six"):
        calculate_eye_aspect_ratio(make_eye(0.25)[:5])


def test_ear_rejects_coincident_eye_corners() -> None:
    points = list(make_eye(0.25))
    points[3] = points[0]

    with pytest.raises(EyeFeatureError, match="corner distance"):
        calculate_eye_aspect_ratio(points)


@pytest.mark.parametrize("duration", [-1.0, float("nan"), True])
def test_blink_config_rejects_invalid_minimum_duration(duration: object) -> None:
    with pytest.raises(ValueError, match="minimum_blink_duration_ms"):
        BlinkTrackerConfig(duration)  # type: ignore[arg-type]


def test_blink_tracker_counts_completed_event_and_reports_duration() -> None:
    tracker = BlinkTracker(BlinkTrackerConfig(minimum_blink_duration_ms=50.0))

    start = tracker.update(True, 0.0)
    active = tracker.update(True, 80.0)
    completed = tracker.update(False, 120.0)

    assert start.blink_duration_ms == 0.0
    assert active.blink_duration_ms == 80.0
    assert active.is_closure_active
    assert completed.blink_count == 1
    assert completed.blink_duration_ms == 120.0
    assert completed.blink_completed


def test_short_closure_does_not_count_as_blink() -> None:
    tracker = BlinkTracker(BlinkTrackerConfig(minimum_blink_duration_ms=100.0))

    tracker.update(True, 0.0)
    state = tracker.update(False, 50.0)

    assert state.blink_count == 0
    assert state.blink_duration_ms == 0.0
    assert not state.blink_completed


def test_missing_observation_interrupts_closure_without_synthetic_blink() -> None:
    tracker = BlinkTracker()

    tracker.update(True, 0.0)
    missing = tracker.update(None, 40.0)
    reopened = tracker.update(False, 80.0)

    assert not missing.is_closure_active
    assert reopened.blink_count == 0
    assert not reopened.blink_completed


def test_blink_tracker_requires_strictly_increasing_timestamps() -> None:
    tracker = BlinkTracker()
    tracker.update(False, 0.0)

    with pytest.raises(ValueError, match="increase strictly"):
        tracker.update(False, 0.0)


def test_blink_tracker_reset_clears_session_state() -> None:
    tracker = BlinkTracker()
    tracker.update(True, 0.0)
    tracker.update(False, 10.0)

    tracker.reset()

    state = tracker.update(False, 0.0)
    assert state.blink_count == 0
    assert state.blink_duration_ms == 0.0


@pytest.mark.parametrize(
    "config",
    [
        {"closure_threshold_ear": 0.0, "perclos_window_frames": 3},
        {"closure_threshold_ear": 0.61, "perclos_window_frames": 3},
        {"closure_threshold_ear": 0.20, "perclos_window_frames": 0},
        {"closure_threshold_ear": 0.20, "perclos_window_frames": True},
    ],
)
def test_stream_config_rejects_invalid_values(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EyeFeatureStreamConfig(**config)  # type: ignore[arg-type]


def test_stream_emits_contract_fields_and_completed_blink() -> None:
    stream = make_stream()

    first = stream.update(
        frame_index=0,
        timestamp_ms=0.0,
        face_detected=True,
        eyes=make_eyes(0.10),
    )
    second = stream.update(
        frame_index=1,
        timestamp_ms=100.0,
        face_detected=True,
        eyes=make_eyes(0.30),
    )

    assert first.landmarks_valid
    assert first.ear_left == pytest.approx(0.10)
    assert first.ear_right == pytest.approx(0.10)
    assert first.ear_avg == pytest.approx(0.10)
    assert first.is_eye_closed is True
    assert first.perclos is None
    assert second.is_eye_closed is False
    assert second.blink_count == 1
    assert second.blink_duration_ms == 100.0
    assert second.blink_completed


def test_perclos_is_trailing_and_requires_complete_valid_window() -> None:
    stream = make_stream(window=3)

    frames = [
        stream.update(
            frame_index=0,
            timestamp_ms=0.0,
            face_detected=True,
            eyes=make_eyes(0.30),
        ),
        stream.update(
            frame_index=1,
            timestamp_ms=10.0,
            face_detected=True,
            eyes=make_eyes(0.10),
        ),
        stream.update(
            frame_index=2,
            timestamp_ms=20.0,
            face_detected=True,
            eyes=make_eyes(0.10),
        ),
    ]

    assert frames[0].perclos is None
    assert frames[1].perclos is None
    assert frames[2].perclos == pytest.approx(2 / 3)

    missing = stream.update(
        frame_index=3,
        timestamp_ms=30.0,
        face_detected=False,
        eyes=None,
    )
    assert missing.perclos is None
    assert missing.landmarks_valid is False
    assert missing.ear_left is None
    assert missing.ear_right is None
    assert missing.ear_avg is None
    assert missing.is_eye_closed is None

    stream.update(
        frame_index=4,
        timestamp_ms=40.0,
        face_detected=True,
        eyes=make_eyes(0.30),
    )
    stream.update(
        frame_index=5,
        timestamp_ms=50.0,
        face_detected=True,
        eyes=make_eyes(0.30),
    )
    recovered = stream.update(
        frame_index=6,
        timestamp_ms=60.0,
        face_detected=True,
        eyes=make_eyes(0.30),
    )
    assert recovered.perclos == 0.0


def test_invalid_eye_geometry_emits_missing_features() -> None:
    points = list(make_eye(0.25))
    points[3] = points[0]
    eyes = EyeLandmarks(left=tuple(points), right=make_eye(0.25))

    frame = make_stream().update(
        frame_index=0,
        timestamp_ms=0.0,
        face_detected=True,
        eyes=eyes,
    )

    assert frame.face_detected is True
    assert frame.landmarks_valid is False
    assert frame.ear_avg is None
    assert frame.is_eye_closed is None


def test_out_of_contract_ear_emits_missing_features() -> None:
    frame = make_stream().update(
        frame_index=0,
        timestamp_ms=0.0,
        face_detected=True,
        eyes=make_eyes(0.61),
    )

    assert frame.landmarks_valid is False
    assert frame.ear_avg is None


def test_stream_rejects_landmarks_when_face_is_not_detected() -> None:
    with pytest.raises(ValueError, match="require face_detected"):
        make_stream().update(
            frame_index=0,
            timestamp_ms=0.0,
            face_detected=False,
            eyes=make_eyes(0.30),
        )


def test_stream_requires_first_frame_zero_without_consuming_state() -> None:
    stream = make_stream()

    with pytest.raises(ValueError, match="first feature frame"):
        stream.update(
            frame_index=1,
            timestamp_ms=10.0,
            face_detected=False,
            eyes=None,
        )

    frame = stream.update(
        frame_index=0,
        timestamp_ms=0.0,
        face_detected=False,
        eyes=None,
    )
    assert frame.frame_index == 0


def test_stream_allows_frame_gaps_but_rejects_duplicate_positions() -> None:
    stream = make_stream()
    stream.update(
        frame_index=0,
        timestamp_ms=0.0,
        face_detected=False,
        eyes=None,
    )
    frame = stream.update(
        frame_index=5,
        timestamp_ms=50.0,
        face_detected=False,
        eyes=None,
    )
    assert frame.frame_index == 5

    with pytest.raises(ValueError, match="frame_index must increase"):
        stream.update(
            frame_index=5,
            timestamp_ms=60.0,
            face_detected=False,
            eyes=None,
        )
    with pytest.raises(ValueError, match="timestamp_ms must increase"):
        stream.update(
            frame_index=6,
            timestamp_ms=50.0,
            face_detected=False,
            eyes=None,
        )


def test_stream_reset_accepts_new_session_frame_zero() -> None:
    stream = make_stream()
    stream.update(
        frame_index=0,
        timestamp_ms=0.0,
        face_detected=True,
        eyes=make_eyes(0.10),
    )

    stream.reset()

    frame = stream.update(
        frame_index=0,
        timestamp_ms=0.0,
        face_detected=True,
        eyes=make_eyes(0.30),
    )
    assert frame.blink_count == 0
    assert frame.perclos is None
