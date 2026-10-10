"""Tests for the isolated, timed vision preprocessing pipeline."""

from collections.abc import Iterator

import pytest

from vigil.runtime.capture import CapturedFrame, FrameTiming
from vigil.runtime.eye_features import EyeFeatureStream, EyeFeatureStreamConfig
from vigil.runtime.landmarks import (
    LEFT_EYE_LANDMARK_INDICES,
    RIGHT_EYE_LANDMARK_INDICES,
    FaceLandmarks,
    NormalizedLandmark,
)
from vigil.runtime.preprocessing import (
    EYE_FEATURE_STAGE_NAME,
    FACE_LANDMARK_STAGE_NAME,
    VisionPreprocessingError,
    VisionPreprocessor,
)


class FakeDetector:
    def __init__(self, results: list[tuple[FaceLandmarks, ...]]) -> None:
        self.results = list(results)
        self.images: list[object] = []

    def detect(self, frame: object) -> tuple[FaceLandmarks, ...]:
        self.images.append(frame)
        return self.results.pop(0)

    def close(self) -> None:
        pass


def make_eye(ear: float) -> tuple[NormalizedLandmark, ...]:
    half_vertical = 2.0 * ear
    return (
        NormalizedLandmark(0.0, 0.0, 0.0),
        NormalizedLandmark(1.0, half_vertical, 0.0),
        NormalizedLandmark(3.0, half_vertical, 0.0),
        NormalizedLandmark(4.0, 0.0, 0.0),
        NormalizedLandmark(3.0, -half_vertical, 0.0),
        NormalizedLandmark(1.0, -half_vertical, 0.0),
    )


def make_face(ear: float = 0.30) -> FaceLandmarks:
    points = [NormalizedLandmark(0.0, 0.0, 0.0) for _ in range(468)]
    for index, point in zip(LEFT_EYE_LANDMARK_INDICES, make_eye(ear)):
        points[index] = point
    for index, point in zip(RIGHT_EYE_LANDMARK_INDICES, make_eye(ear)):
        points[index] = point
    return FaceLandmarks(points=tuple(points))


def make_frame(index: int, timestamp_seconds: float) -> CapturedFrame:
    return CapturedFrame(
        image=f"frame-{index}",
        index=index,
        timing=FrameTiming(
            timestamp_seconds=timestamp_seconds,
            delta_seconds=None if index == 0 else 0.1,
            frames_per_second=None if index == 0 else 10.0,
        ),
    )


def make_stream() -> EyeFeatureStream:
    return EyeFeatureStream(
        EyeFeatureStreamConfig(
            closure_threshold_ear=0.20,
            perclos_window_frames=3,
            minimum_blink_duration_ms=50.0,
        )
    )


def test_preprocessor_emits_features_and_measured_stage_boundaries() -> None:
    clock: Iterator[float] = iter([1.0, 1.002, 2.0, 2.003])
    detector = FakeDetector([(make_face(),)])
    preprocessor = VisionPreprocessor(
        detector,
        make_stream(),
        clock=lambda: next(clock),
    )

    result = preprocessor.process(make_frame(0, 100.0))

    assert detector.images == ["frame-0"]
    assert result.eye_features.frame_index == 0
    assert result.eye_features.timestamp_ms == 0.0
    assert result.eye_features.face_detected is True
    assert result.eye_features.landmarks_valid is True
    assert result.eye_features.ear_avg == pytest.approx(0.30)
    assert result.eye_features.is_eye_closed is False
    assert result.latency.stage(FACE_LANDMARK_STAGE_NAME).duration_ms == pytest.approx(
        2.0
    )
    assert result.latency.stage(EYE_FEATURE_STAGE_NAME).duration_ms == pytest.approx(
        3.0
    )


def test_preprocessor_uses_session_relative_timestamps_and_missing_face_contract(
) -> None:
    clock = iter([0.0, 0.001, 0.002, 0.003, 1.0, 1.001, 1.002, 1.003])
    preprocessor = VisionPreprocessor(
        FakeDetector([(make_face(),), ()]),
        make_stream(),
        clock=lambda: next(clock),
    )
    preprocessor.process(make_frame(0, 250.0))

    result = preprocessor.process(make_frame(1, 250.125))

    assert result.eye_features.timestamp_ms == pytest.approx(125.0)
    assert result.eye_features.face_detected is False
    assert result.eye_features.landmarks_valid is False
    assert result.eye_features.ear_left is None
    assert result.eye_features.ear_right is None
    assert result.eye_features.ear_avg is None
    assert result.eye_features.is_eye_closed is None


def test_incomplete_landmarks_preserve_detected_face_but_emit_missing_features(
) -> None:
    incomplete = FaceLandmarks(points=(NormalizedLandmark(0.0, 0.0, 0.0),))
    clock = iter([0.0, 0.001, 0.002, 0.003])
    preprocessor = VisionPreprocessor(
        FakeDetector([(incomplete,)]),
        make_stream(),
        clock=lambda: next(clock),
    )

    result = preprocessor.process(make_frame(0, 1.0))

    assert result.eye_features.face_detected is True
    assert result.eye_features.landmarks_valid is False
    assert result.eye_features.ear_avg is None


def test_preprocessor_rejects_multiple_faces_for_stable_driver_identity() -> None:
    clock = iter([0.0, 0.001, 0.002, 0.003])
    preprocessor = VisionPreprocessor(
        FakeDetector([(make_face(), make_face())]),
        make_stream(),
        clock=lambda: next(clock),
    )

    with pytest.raises(VisionPreprocessingError, match="at most one"):
        preprocessor.process(make_frame(0, 1.0))


def test_preprocessor_requires_frame_zero_at_session_start() -> None:
    preprocessor = VisionPreprocessor(FakeDetector([]), make_stream())

    with pytest.raises(VisionPreprocessingError, match="index 0"):
        preprocessor.process(make_frame(2, 1.0))


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_preprocessor_rejects_invalid_capture_index(index: object) -> None:
    preprocessor = VisionPreprocessor(FakeDetector([]), make_stream())
    frame = make_frame(0, 1.0)
    malformed = CapturedFrame(
        image=frame.image,
        index=index,  # type: ignore[arg-type]
        timing=frame.timing,
    )

    with pytest.raises(VisionPreprocessingError, match="non-negative integer"):
        preprocessor.process(malformed)


@pytest.mark.parametrize("timestamp", [-1.0, float("nan"), float("inf")])
def test_preprocessor_rejects_invalid_capture_timestamp(timestamp: float) -> None:
    preprocessor = VisionPreprocessor(FakeDetector([]), make_stream())

    with pytest.raises(VisionPreprocessingError, match="finite and non-negative"):
        preprocessor.process(make_frame(0, timestamp))


def test_preprocessor_reset_starts_a_new_feature_session() -> None:
    clock = iter([0.0, 0.001, 0.002, 0.003, 1.0, 1.001, 1.002, 1.003])
    preprocessor = VisionPreprocessor(
        FakeDetector([(make_face(0.10),), (make_face(0.30),)]),
        make_stream(),
        clock=lambda: next(clock),
    )
    preprocessor.process(make_frame(0, 10.0))

    preprocessor.reset()
    result = preprocessor.process(make_frame(0, 20.0))

    assert result.eye_features.timestamp_ms == 0.0
    assert result.eye_features.blink_count == 0
    assert result.eye_features.perclos is None
