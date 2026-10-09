"""Ocular geometry and temporal eye features for Vigil's runtime."""

from __future__ import annotations

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
