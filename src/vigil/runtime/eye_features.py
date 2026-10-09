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
