"""Reusable vision preprocessing boundary for Vigil's live runtime."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from vigil.runtime.capture import CapturedFrame
from vigil.runtime.eye_features import EyeFeatureFrame, EyeFeatureStream
from vigil.runtime.landmarks import (
    FaceLandmarkDetector,
    LandmarkOutputError,
    extract_eye_landmarks,
)
from vigil.runtime.performance import Clock, FrameLatency, StageLatencyRecorder


FACE_LANDMARK_STAGE_NAME = "face_landmarks"
EYE_FEATURE_STAGE_NAME = "eye_features"


class VisionPreprocessingError(RuntimeError):
    """Raised when a frame cannot satisfy the preprocessing contract."""


@dataclass(frozen=True, slots=True)
class VisionPreprocessingResult:
    """Eye features and measured stage latency for one captured frame."""

    eye_features: EyeFeatureFrame
    latency: FrameLatency


class VisionPreprocessor:
    """Convert captured frames into M1's stable eye-feature stream."""

    def __init__(
        self,
        detector: FaceLandmarkDetector,
        eye_feature_stream: EyeFeatureStream,
        *,
        clock: Clock = perf_counter,
    ) -> None:
        self._detector = detector
        self._eye_feature_stream = eye_feature_stream
        self._clock = clock
        self._session_start_seconds: float | None = None

    def reset(self) -> None:
        """Reset session-relative time and temporal eye-feature state."""
        self._session_start_seconds = None
        self._eye_feature_stream.reset()

    def process(self, frame: CapturedFrame) -> VisionPreprocessingResult:
        """Process one captured frame through explicit, timed CV stages."""
        if not isinstance(frame, CapturedFrame):
            raise TypeError("frame must be a CapturedFrame")
        if (
            isinstance(frame.index, bool)
            or not isinstance(frame.index, int)
            or frame.index < 0
        ):
            raise VisionPreprocessingError(
                "captured frame index must be a non-negative integer"
            )
        timestamp_seconds = frame.timing.timestamp_seconds
        if (
            isinstance(timestamp_seconds, bool)
            or not isinstance(timestamp_seconds, (int, float))
            or not isfinite(timestamp_seconds)
            or timestamp_seconds < 0.0
        ):
            raise VisionPreprocessingError(
                "captured frame timestamp must be finite and non-negative"
            )
        if self._session_start_seconds is None:
            if frame.index != 0:
                raise VisionPreprocessingError(
                    "the first preprocessing frame must have index 0"
                )
            self._session_start_seconds = timestamp_seconds

        timestamp_ms = (timestamp_seconds - self._session_start_seconds) * 1000.0
        recorder = StageLatencyRecorder(self._clock)
        with recorder.measure(FACE_LANDMARK_STAGE_NAME):
            faces = self._detector.detect(frame.image)

        with recorder.measure(EYE_FEATURE_STAGE_NAME):
            if len(faces) > 1:
                raise VisionPreprocessingError(
                    "vision preprocessing requires at most one detected face"
                )
            eyes = None
            if faces:
                try:
                    eyes = extract_eye_landmarks(faces[0])
                except LandmarkOutputError:
                    eyes = None
            eye_features = self._eye_feature_stream.update(
                frame_index=frame.index,
                timestamp_ms=timestamp_ms,
                face_detected=bool(faces),
                eyes=eyes,
            )

        return VisionPreprocessingResult(
            eye_features=eye_features,
            latency=recorder.snapshot(frame.index),
        )
