"""Ocular geometry and temporal eye features for Vigil's runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence

from vigil.runtime.landmarks import EyeLandmarks, NormalizedLandmark


class EyeFeatureError(RuntimeError):
    """Raised when eye features cannot be computed from supplied landmarks."""


@dataclass(frozen=True, slots=True)
class EyeAspectRatios:
    """Left, right, and mean eye aspect ratios for one frame."""

    left: float
    right: float
    average: float

    def __post_init__(self) -> None:
        for name, value in (
            ("left", self.left),
            ("right", self.right),
            ("average", self.average),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} EAR must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BlinkTrackerConfig:
    """Configuration for deterministic blink event tracking."""

    minimum_blink_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        value = self.minimum_blink_duration_ms
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                "minimum_blink_duration_ms must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class BlinkState:
    """Cumulative and current state after processing one eye observation."""

    blink_count: int
    blink_duration_ms: float
    is_closure_active: bool
    blink_completed: bool


class BlinkTracker:
    """Track completed closed-to-open eye events using monotonic timestamps."""

    def __init__(self, config: BlinkTrackerConfig | None = None) -> None:
        self.config = config or BlinkTrackerConfig()
        self.reset()

    def reset(self) -> None:
        """Clear all session-local blink state."""
        self._blink_count = 0
        self._closure_start_ms: float | None = None
        self._last_blink_duration_ms = 0.0
        self._last_timestamp_ms: float | None = None

    @property
    def blink_count(self) -> int:
        """Return the number of completed blinks in this session."""
        return self._blink_count

    def update(
        self,
        is_eye_closed: bool | None,
        timestamp_ms: float,
    ) -> BlinkState:
        """Advance tracking with one valid or missing eye observation."""
        self._validate_timestamp(timestamp_ms)
        if is_eye_closed is not None and not isinstance(is_eye_closed, bool):
            raise TypeError("is_eye_closed must be a boolean or None")

        timestamp = float(timestamp_ms)
        self._last_timestamp_ms = timestamp

        if is_eye_closed is None:
            self._closure_start_ms = None
            return self._state(
                duration_ms=self._last_blink_duration_ms,
                is_closure_active=False,
                blink_completed=False,
            )

        if is_eye_closed:
            if self._closure_start_ms is None:
                self._closure_start_ms = timestamp
            return self._state(
                duration_ms=timestamp - self._closure_start_ms,
                is_closure_active=True,
                blink_completed=False,
            )

        blink_completed = False
        if self._closure_start_ms is not None:
            duration_ms = timestamp - self._closure_start_ms
            if duration_ms >= self.config.minimum_blink_duration_ms:
                self._blink_count += 1
                self._last_blink_duration_ms = duration_ms
                blink_completed = True
            self._closure_start_ms = None

        return self._state(
            duration_ms=self._last_blink_duration_ms,
            is_closure_active=False,
            blink_completed=blink_completed,
        )

    def _validate_timestamp(self, timestamp_ms: float) -> None:
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, (int, float))
            or not isfinite(timestamp_ms)
            or timestamp_ms < 0.0
        ):
            raise ValueError("timestamp_ms must be finite and non-negative")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError("timestamp_ms must increase strictly")

    def _state(
        self,
        *,
        duration_ms: float,
        is_closure_active: bool,
        blink_completed: bool,
    ) -> BlinkState:
        return BlinkState(
            blink_count=self._blink_count,
            blink_duration_ms=duration_ms,
            is_closure_active=is_closure_active,
            blink_completed=blink_completed,
        )


@dataclass(frozen=True, slots=True)
class EyeFeatureStreamConfig:
    """Configuration for closure and trailing frame-based PERCLOS features."""

    closure_threshold_ear: float
    perclos_window_frames: int
    minimum_blink_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        threshold = self.closure_threshold_ear
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not isfinite(threshold)
            or not 0.0 < threshold <= 0.60
        ):
            raise ValueError("closure_threshold_ear must be within (0.0, 0.60]")
        if (
            isinstance(self.perclos_window_frames, bool)
            or not isinstance(self.perclos_window_frames, int)
            or self.perclos_window_frames < 1
        ):
            raise ValueError("perclos_window_frames must be a positive integer")
        BlinkTrackerConfig(self.minimum_blink_duration_ms)


@dataclass(frozen=True, slots=True)
class EyeFeatureFrame:
    """Observable eye-feature values emitted for one runtime frame."""

    frame_index: int
    timestamp_ms: float
    face_detected: bool
    landmarks_valid: bool
    ear_left: float | None
    ear_right: float | None
    ear_avg: float | None
    is_eye_closed: bool | None
    blink_count: int
    blink_duration_ms: float
    perclos: float | None
    blink_completed: bool


class EyeFeatureStream:
    """Convert ordered eye landmarks into a causal temporal feature stream."""

    def __init__(self, config: EyeFeatureStreamConfig) -> None:
        if not isinstance(config, EyeFeatureStreamConfig):
            raise TypeError("config must be an EyeFeatureStreamConfig")
        self.config = config
        self._blink_tracker = BlinkTracker(
            BlinkTrackerConfig(config.minimum_blink_duration_ms)
        )
        self._closure_window: deque[bool | None] = deque(
            maxlen=config.perclos_window_frames
        )
        self._last_frame_index: int | None = None
        self._last_timestamp_ms: float | None = None

    def reset(self) -> None:
        """Clear temporal state for the start of a new capture session."""
        self._blink_tracker.reset()
        self._closure_window.clear()
        self._last_frame_index = None
        self._last_timestamp_ms = None

    def update(
        self,
        *,
        frame_index: int,
        timestamp_ms: float,
        face_detected: bool,
        eyes: EyeLandmarks | None,
    ) -> EyeFeatureFrame:
        """Process one frame while preserving missing-observation semantics."""
        self._validate_position(frame_index, timestamp_ms)
        if not isinstance(face_detected, bool):
            raise TypeError("face_detected must be a boolean")
        if eyes is not None and not isinstance(eyes, EyeLandmarks):
            raise TypeError("eyes must be EyeLandmarks or None")
        if not face_detected and eyes is not None:
            raise ValueError("eye landmarks require face_detected to be True")

        timestamp = float(timestamp_ms)
        ratios = self._valid_ratios(eyes)
        if ratios is None:
            is_eye_closed = None
            blink = self._blink_tracker.update(None, timestamp)
            self._closure_window.append(None)
        else:
            is_eye_closed = ratios.average < self.config.closure_threshold_ear
            blink = self._blink_tracker.update(is_eye_closed, timestamp)
            self._closure_window.append(is_eye_closed)

        self._last_frame_index = frame_index
        self._last_timestamp_ms = timestamp
        landmarks_valid = ratios is not None
        return EyeFeatureFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp,
            face_detected=face_detected,
            landmarks_valid=landmarks_valid,
            ear_left=ratios.left if ratios is not None else None,
            ear_right=ratios.right if ratios is not None else None,
            ear_avg=ratios.average if ratios is not None else None,
            is_eye_closed=is_eye_closed,
            blink_count=blink.blink_count,
            blink_duration_ms=blink.blink_duration_ms,
            perclos=self._perclos(),
            blink_completed=blink.blink_completed,
        )

    def _validate_position(self, frame_index: int, timestamp_ms: float) -> None:
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, (int, float))
            or not isfinite(timestamp_ms)
            or timestamp_ms < 0.0
        ):
            raise ValueError("timestamp_ms must be finite and non-negative")

        if self._last_frame_index is None:
            if frame_index != 0 or timestamp_ms != 0.0:
                raise ValueError(
                    "the first feature frame must use frame_index 0 and "
                    "timestamp_ms 0.0"
                )
            return

        if frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase strictly")
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must increase strictly")

    @staticmethod
    def _valid_ratios(eyes: EyeLandmarks | None) -> EyeAspectRatios | None:
        if eyes is None:
            return None
        try:
            ratios = calculate_eye_aspect_ratios(eyes)
        except EyeFeatureError:
            return None
        if max(ratios.left, ratios.right, ratios.average) > 0.60:
            return None
        return ratios

    def _perclos(self) -> float | None:
        if len(self._closure_window) < self.config.perclos_window_frames:
            return None
        if any(value is None for value in self._closure_window):
            return None
        closed_frames = sum(value is True for value in self._closure_window)
        return closed_frames / self.config.perclos_window_frames


def _distance(first: NormalizedLandmark, second: NormalizedLandmark) -> float:
    """Return two-dimensional Euclidean distance in normalized image space."""
    return hypot(first.x - second.x, first.y - second.y)


def calculate_eye_aspect_ratio(
    points: Sequence[NormalizedLandmark],
) -> float:
    """Calculate EAR from six ordered eye landmarks.

    Point order is ``corner, upper, upper, corner, lower, lower``. The z-axis
    is intentionally excluded because EAR is defined in the image plane.
    """
    if len(points) != 6:
        raise ValueError("EAR requires exactly six ordered eye landmarks")

    horizontal = _distance(points[0], points[3])
    if horizontal <= 0.0:
        raise EyeFeatureError("eye corner distance must be positive")

    vertical_outer = _distance(points[1], points[5])
    vertical_inner = _distance(points[2], points[4])
    ear = (vertical_outer + vertical_inner) / (2.0 * horizontal)
    if not isfinite(ear):
        raise EyeFeatureError("calculated EAR must be finite")
    return ear


def calculate_eye_aspect_ratios(eyes: EyeLandmarks) -> EyeAspectRatios:
    """Calculate left, right, and mean EAR from both eyes."""
    left = calculate_eye_aspect_ratio(eyes.left)
    right = calculate_eye_aspect_ratio(eyes.right)
    return EyeAspectRatios(
        left=left,
        right=right,
        average=(left + right) / 2.0,
    )
