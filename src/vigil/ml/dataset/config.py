"""Configuration and contract constants for Vigil M2 temporal windowing."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Exact 8-channel feature ordering locked by M2 D4 contract
LOCKED_FEATURE_CHANNELS: tuple[str, ...] = (
    "ear_avg",
    "is_eye_closed",
    "blink_duration_ms",
    "perclos",
    "ear_avg_valid",
    "is_eye_closed_valid",
    "blink_duration_valid",
    "perclos_valid",
)

LOCKED_WINDOW_LENGTH: int = 60
LOCKED_STRIDE: int = 15

CONTINUOUS_FEATURE_INDICES: tuple[int, ...] = (0, 2, 3)  # ear_avg, blink_duration_ms, perclos
BINARY_INDICATOR_INDICES: tuple[int, ...] = (1,)         # is_eye_closed
VALIDITY_MASK_INDICES: tuple[int, ...] = (4, 5, 6, 7)    # masks


@dataclass(frozen=True)
class WindowConfig:
    """Locked parameters for sliding window segmentation.

    Attributes:
        window_length: Sequence length W (number of recorded samples per window). Locked to 60.
        stride: Temporal stride S between consecutive windows. Locked to 15.
        max_frame_gap: Maximum allowed frame index gap between consecutive frames
            within a continuous segment. Default 1 (strictly consecutive, no dropped frames).
            Discontinuities > max_frame_gap trigger a segment break.
        max_time_gap_ms: Maximum allowed timestamp delta (ms) before triggering a segment break
            due to capture stalls or pause. Default 200.0 ms.
    """

    window_length: int = LOCKED_WINDOW_LENGTH
    stride: int = LOCKED_STRIDE
    max_frame_gap: int = 1
    max_time_gap_ms: float = 200.0

    def __post_init__(self) -> None:
        """Validate window configuration constraints."""
        if not isinstance(self.window_length, int) or isinstance(self.window_length, bool):
            raise TypeError(f"window_length must be an integer, got {type(self.window_length).__name__}.")
        if self.window_length != LOCKED_WINDOW_LENGTH:
            raise ValueError(
                f"window_length is locked to {LOCKED_WINDOW_LENGTH} by the M2 D4 contract, got {self.window_length}."
            )

        if not isinstance(self.stride, int) or isinstance(self.stride, bool):
            raise TypeError(f"stride must be an integer, got {type(self.stride).__name__}.")
        if self.stride != LOCKED_STRIDE:
            raise ValueError(
                f"stride is locked to {LOCKED_STRIDE} by the M2 D4 contract, got {self.stride}."
            )

        if not isinstance(self.max_frame_gap, int) or isinstance(self.max_frame_gap, bool):
            raise TypeError(f"max_frame_gap must be an integer, got {type(self.max_frame_gap).__name__}.")
        if self.max_frame_gap <= 0:
            raise ValueError(f"max_frame_gap must be positive, got {self.max_frame_gap}.")

        if not isinstance(self.max_time_gap_ms, (int, float)) or isinstance(self.max_time_gap_ms, bool):
            raise TypeError(f"max_time_gap_ms must be numeric, got {type(self.max_time_gap_ms).__name__}.")
        if not math.isfinite(self.max_time_gap_ms) or self.max_time_gap_ms <= 0.0:
            raise ValueError(f"max_time_gap_ms must be finite and positive, got {self.max_time_gap_ms}.")
