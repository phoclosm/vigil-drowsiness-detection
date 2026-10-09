"""Configuration for Vigil M2 rule-based fatigue baseline classifier.

This module provides configurable threshold and alert parameters for the
classical rule-based fatigue baseline.

Note:
    All default thresholds defined herein are PROVISIONAL literature heuristics.
    Empirical calibration against recorded driver datasets is pending.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    """Configurable thresholds for heuristic rule-based fatigue classification.

    Decision hierarchy and provisional thresholds follow the M2 D3 contract:
        - SLEEPING = 2: Current eyes closed and continuous closure duration >= sleeping_closure_duration_ms (1500 ms).
        - DROWSY = 1: Causal PERCLOS >= drowsy_perclos_threshold (0.20), or continuous closure duration >= drowsy_closure_duration_ms (500 ms).
        - ACTIVE = 0: Otherwise.
        - Abstention (no prediction): If landmarks are invalid or is_eye_closed is unavailable.

    Attributes:
        drowsy_closure_duration_ms: Continuous eye closure duration in ms
            required to enter DROWSY state. Default 500.0 ms. (Provisional).
        sleeping_closure_duration_ms: Continuous eye closure duration in ms
            required to enter SLEEPING state when eyes are closed. Default 1500.0 ms. (Provisional).
        drowsy_perclos_threshold: Trailing PERCLOS ratio (0.0 to 1.0) required
            to trigger DROWSY state. Default 0.20 (20%). (Provisional).
        alert_sustain_window_ms: Continuous duration of sustained fatigue (ms)
            required before triggering an audible/visible alert. Default 1000.0 ms.
        immediate_sleep_alert: If True, trigger an alert immediately upon entering
            the SLEEPING state without waiting for alert_sustain_window_ms.
        is_provisional: Flag indicating whether thresholds are provisional literature
            heuristics rather than empirically calibrated values.
        calibration_notes: Explanatory notes on threshold calibration status.
    """

    drowsy_closure_duration_ms: float = 500.0
    sleeping_closure_duration_ms: float = 1500.0
    drowsy_perclos_threshold: float = 0.20
    alert_sustain_window_ms: float = 1000.0
    immediate_sleep_alert: bool = True
    is_provisional: bool = True
    calibration_notes: str = "Empirical calibration is pending recorded driver datasets."

    def __post_init__(self) -> None:
        """Validate config parameters."""
        self.validate()

    def validate(self) -> None:
        """Enforce parameter domain and consistency constraints."""
        for name, val in [
            ("drowsy_closure_duration_ms", self.drowsy_closure_duration_ms),
            ("sleeping_closure_duration_ms", self.sleeping_closure_duration_ms),
            ("drowsy_perclos_threshold", self.drowsy_perclos_threshold),
            ("alert_sustain_window_ms", self.alert_sustain_window_ms),
        ]:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"{name} must be numeric, got {type(val).__name__}.")
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}.")

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

        if self.alert_sustain_window_ms < 0.0:
            raise ValueError(
                f"alert_sustain_window_ms must be >= 0.0, got {self.alert_sustain_window_ms}."
            )
