"""Configuration for Vigil M2 rule-based fatigue baseline classifier.

This module provides configurable threshold and alert parameters for the
classical rule-based fatigue baseline.

Note:
    All default thresholds defined herein are PROVISIONAL.
    Empirical calibration against recorded driver datasets is pending.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vigil.ml.data.schema import FatigueLabel


@dataclass(frozen=True)
class BaselineConfig:
    """Configurable thresholds for heuristic rule-based fatigue classification.

    Attributes:
        ear_close_threshold: Eye aspect ratio threshold below which eyes are
            considered closed. Default 0.20. (Provisional).
        drowsy_closure_duration_ms: Continuous eye closure duration in ms
            required to enter DROWSY state. Default 400.0 ms. (Provisional).
        sleeping_closure_duration_ms: Continuous eye closure duration in ms
            required to enter SLEEPING state. Default 1500.0 ms. (Provisional).
        drowsy_perclos_threshold: Trailing PERCLOS ratio (0.0 to 1.0) required
            to trigger DROWSY state. Default 0.15 (15%). (Provisional).
        sleeping_perclos_threshold: Trailing PERCLOS ratio (0.0 to 1.0) required
            to trigger SLEEPING state. Default 0.35 (35%). (Provisional).
        drowsy_blink_duration_ms: Single blink duration in ms indicating a
            sluggish blink associated with fatigue. Default 350.0 ms. (Provisional).
        alert_sustain_window_ms: Continuous duration of sustained fatigue (ms)
            required before triggering an audible/visible alert. Default 1000.0 ms.
        immediate_sleep_alert: If True, trigger an alert immediately upon entering
            the SLEEPING state without waiting for alert_sustain_window_ms.
        missing_feature_fallback: Default class to output when face is not detected
            or landmarks are invalid. Default FatigueLabel.ACTIVE.
        is_provisional: Flag indicating whether thresholds are provisional literature
            heuristics rather than empirically calibrated values.
        calibration_notes: Explanatory notes on threshold calibration status.
    """

    ear_close_threshold: float = 0.20
    drowsy_closure_duration_ms: float = 400.0
    sleeping_closure_duration_ms: float = 1500.0
    drowsy_perclos_threshold: float = 0.15
    sleeping_perclos_threshold: float = 0.35
    drowsy_blink_duration_ms: float = 350.0
    alert_sustain_window_ms: float = 1000.0
    immediate_sleep_alert: bool = True
    missing_feature_fallback: FatigueLabel = FatigueLabel.ACTIVE
    is_provisional: bool = True
    calibration_notes: str = "Empirical calibration is pending recorded driver datasets."

    def __post_init__(self) -> None:
        """Validate config parameters."""
        self.validate()

    def validate(self) -> None:
        """Enforce parameter domain and consistency constraints."""
        for name, val in [
            ("ear_close_threshold", self.ear_close_threshold),
            ("drowsy_closure_duration_ms", self.drowsy_closure_duration_ms),
            ("sleeping_closure_duration_ms", self.sleeping_closure_duration_ms),
            ("drowsy_perclos_threshold", self.drowsy_perclos_threshold),
            ("sleeping_perclos_threshold", self.sleeping_perclos_threshold),
            ("drowsy_blink_duration_ms", self.drowsy_blink_duration_ms),
            ("alert_sustain_window_ms", self.alert_sustain_window_ms),
        ]:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"{name} must be numeric, got {type(val).__name__}.")
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}.")

        if not (0.0 < self.ear_close_threshold < 0.60):
            raise ValueError(
                f"ear_close_threshold must be in (0.0, 0.60), got {self.ear_close_threshold}."
            )

        if self.drowsy_closure_duration_ms <= 0.0:
            raise ValueError(
                f"drowsy_closure_duration_ms must be positive, got {self.drowsy_closure_duration_ms}."
            )

        if self.sleeping_closure_duration_ms <= self.drowsy_closure_duration_ms:
            raise ValueError(
                f"sleeping_closure_duration_ms ({self.sleeping_closure_duration_ms}) must be "
                f"strictly greater than drowsy_closure_duration_ms ({self.drowsy_closure_duration_ms})."
            )

        if not (0.0 < self.drowsy_perclos_threshold <= 1.0):
            raise ValueError(
                f"drowsy_perclos_threshold must be in (0.0, 1.0], got {self.drowsy_perclos_threshold}."
            )

        if not (0.0 < self.sleeping_perclos_threshold <= 1.0):
            raise ValueError(
                f"sleeping_perclos_threshold must be in (0.0, 1.0], got {self.sleeping_perclos_threshold}."
            )

        if self.sleeping_perclos_threshold <= self.drowsy_perclos_threshold:
            raise ValueError(
                f"sleeping_perclos_threshold ({self.sleeping_perclos_threshold}) must be "
                f"strictly greater than drowsy_perclos_threshold ({self.drowsy_perclos_threshold})."
            )

        if self.drowsy_blink_duration_ms < 0.0:
            raise ValueError(
                f"drowsy_blink_duration_ms must be >= 0.0, got {self.drowsy_blink_duration_ms}."
            )

        if self.alert_sustain_window_ms < 0.0:
            raise ValueError(
                f"alert_sustain_window_ms must be >= 0.0, got {self.alert_sustain_window_ms}."
            )

        if self.missing_feature_fallback not in (
            FatigueLabel.ACTIVE,
            FatigueLabel.DROWSY,
            FatigueLabel.SLEEPING,
        ):
            raise ValueError(
                f"missing_feature_fallback must be a canonical class (ACTIVE, DROWSY, SLEEPING), "
                f"got {self.missing_feature_fallback}."
            )
