"""Adapter between M1 eye features and M2 temporal model inference."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Protocol

from vigil.ml.data import LiveInputFrame
from vigil.ml.inference import (
    CANONICAL_CLASS_LABELS,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_INVALID_INPUT,
    STATUS_OK,
    InferenceResult,
    TemporalInferenceEngine,
)
from vigil.runtime.eye_features import EyeFeatureFrame


_INFERENCE_STATUSES = frozenset(
    {
        STATUS_OK,
        STATUS_INSUFFICIENT_HISTORY,
        STATUS_INVALID_INPUT,
        STATUS_ERROR,
    }
)


class ModelInferenceError(RuntimeError):
    """Raised when live model inference cannot be completed safely."""


class ModelLoadError(ModelInferenceError):
    """Raised when an exported model bundle cannot be loaded."""


class InferenceContractError(ModelInferenceError):
    """Raised when runtime input or model output violates its contract."""


class InferenceEngine(Protocol):
    """Structural boundary used by the runtime and deterministic test doubles."""

    def process_sample(self, sample: LiveInputFrame) -> InferenceResult:
        """Process one unlabeled runtime frame."""

    def reset(self) -> None:
        """Clear temporal inference state."""


@dataclass(frozen=True, slots=True)
class LivePrediction:
    """One inference response plus display-safe fatigue-state fields."""

    result: InferenceResult
    state: str | None
    confidence: float | None
    source_frame_index: int | None
    is_fresh: bool

    @property
    def is_available(self) -> bool:
        """Return whether a genuine model prediction is available for display."""
        return self.state is not None


class ModelInferenceAdapter:
    """Convert ordered M1 eye features into M2 live inference frames."""

    def __init__(self, engine: InferenceEngine, *, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not callable(getattr(engine, "process_sample", None)):
            raise TypeError("engine must provide process_sample")
        if not callable(getattr(engine, "reset", None)):
            raise TypeError("engine must provide reset")

        self._engine = engine
        self._session_id = session_id
        self._last_timestamp_ms: float | None = None
        self._cached_state: str | None = None
        self._cached_confidence: float | None = None
        self._cached_source_frame_index: int | None = None

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        session_id: str,
    ) -> ModelInferenceAdapter:
        """Load a validated M2 model bundle and construct its live adapter."""
        path = Path(bundle_path)
        if not path.is_file():
            raise ModelLoadError(f"model bundle file does not exist: {path}")

        try:
            engine = TemporalInferenceEngine.from_bundle(path)
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise ModelLoadError(f"could not load model bundle {path}: {exc}") from exc
        return cls(engine, session_id=session_id)

    @property
    def session_id(self) -> str:
        """Return the explicit identifier for this live capture session."""
        return self._session_id

    def reset(self) -> None:
        """Clear adapter timing and the model's temporal window."""
        try:
            self._engine.reset()
        finally:
            self._last_timestamp_ms = None
            self._clear_display_cache()

    def process(self, features: EyeFeatureFrame) -> LivePrediction:
        """Forward one eye-feature frame through the temporal model contract."""
        if not isinstance(features, EyeFeatureFrame):
            raise TypeError("features must be an EyeFeatureFrame")

        if self._last_timestamp_ms is None and features.frame_index != 0:
            self._clear_display_cache()
            raise InferenceContractError(
                "the first inference frame must have frame_index 0"
            )
        frame_delta_ms = (
            None
            if features.frame_index == 0
            else features.timestamp_ms - self._last_timestamp_ms
        )

        try:
            live_frame = LiveInputFrame(
                session_id=self._session_id,
                frame_index=features.frame_index,
                timestamp_ms=features.timestamp_ms,
                frame_delta_ms=frame_delta_ms,
                face_detected=features.face_detected,
                landmarks_valid=features.landmarks_valid,
                ear_left=features.ear_left,
                ear_right=features.ear_right,
                ear_avg=features.ear_avg,
                is_eye_closed=features.is_eye_closed,
                blink_count=features.blink_count,
                blink_duration_ms=features.blink_duration_ms,
                perclos=features.perclos,
                fps=None,
            )
        except (TypeError, ValueError) as exc:
            self._clear_display_cache()
            raise InferenceContractError(
                f"eye features cannot form a live inference frame: {exc}"
            ) from exc

        try:
            result = self._engine.process_sample(live_frame)
        except Exception as exc:  # The adapter adds runtime context to backend errors.
            self._clear_display_cache()
            raise ModelInferenceError(
                f"model inference failed for frame {features.frame_index}: {exc}"
            ) from exc

        try:
            _validate_result(
                result,
                session_id=self._session_id,
                frame_index=features.frame_index,
            )
        except InferenceContractError:
            self._clear_display_cache()
            raise
        self._last_timestamp_ms = features.timestamp_ms

        if result.status == STATUS_OK:
            self._cached_state = result.predicted_class
            self._cached_confidence = result.confidence
            self._cached_source_frame_index = features.frame_index
            return LivePrediction(
                result=result,
                state=self._cached_state,
                confidence=self._cached_confidence,
                source_frame_index=self._cached_source_frame_index,
                is_fresh=True,
            )
        if result.status == STATUS_INSUFFICIENT_HISTORY and result.reason == "stride_skip":
            return LivePrediction(
                result=result,
                state=self._cached_state,
                confidence=self._cached_confidence,
                source_frame_index=self._cached_source_frame_index,
                is_fresh=False,
            )

        self._clear_display_cache()
        return LivePrediction(
            result=result,
            state=None,
            confidence=None,
            source_frame_index=None,
            is_fresh=False,
        )

    def _clear_display_cache(self) -> None:
        self._cached_state = None
        self._cached_confidence = None
        self._cached_source_frame_index = None


def _validate_result(
    result: InferenceResult,
    *,
    session_id: str,
    frame_index: int,
) -> None:
    """Reject backend output that could mislabel the live overlay."""
    if not isinstance(result, InferenceResult):
        raise InferenceContractError("engine must return an InferenceResult")
    if result.status not in _INFERENCE_STATUSES:
        raise InferenceContractError(f"unknown inference status: {result.status!r}")
    if not isinstance(result.reason, str) or not result.reason:
        raise InferenceContractError("inference reason must be a non-empty string")
    if (
        isinstance(result.latency_ms, bool)
        or not isinstance(result.latency_ms, (int, float))
        or not isfinite(result.latency_ms)
        or result.latency_ms < 0.0
    ):
        raise InferenceContractError(
            "inference latency_ms must be finite and non-negative"
        )
    if (
        isinstance(result.buffer_size, bool)
        or not isinstance(result.buffer_size, int)
        or result.buffer_size < 0
    ):
        raise InferenceContractError("inference buffer_size must be non-negative")
    if result.session_id != session_id:
        raise InferenceContractError("inference result session_id does not match input")
    if result.end_frame_index != frame_index:
        raise InferenceContractError(
            "inference result end_frame_index does not match input"
        )

    prediction_fields = (
        result.predicted_class,
        result.class_id,
        result.probabilities,
        result.confidence,
    )
    if result.status != STATUS_OK:
        if any(value is not None for value in prediction_fields):
            raise InferenceContractError(
                "non-OK inference results must not contain a prediction"
            )
        return

    class_id_by_name = {
        class_name: class_id
        for class_id, class_name in CANONICAL_CLASS_LABELS.items()
    }
    if result.predicted_class not in class_id_by_name:
        raise InferenceContractError("prediction must use a canonical fatigue state")
    if result.class_id != class_id_by_name[result.predicted_class]:
        raise InferenceContractError("prediction class_id does not match its state")
    if not isinstance(result.probabilities, dict):
        raise InferenceContractError("prediction probabilities must be a dictionary")
    if set(result.probabilities) != set(class_id_by_name):
        raise InferenceContractError(
            "prediction probabilities must contain all canonical states"
        )
    probability_values = tuple(result.probabilities.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in probability_values
    ):
        raise InferenceContractError(
            "prediction probabilities must be finite values within [0.0, 1.0]"
        )
    if not isclose(sum(probability_values), 1.0, abs_tol=1e-5):
        raise InferenceContractError("prediction probabilities must sum to 1.0")
    if (
        isinstance(result.confidence, bool)
        or not isinstance(result.confidence, (int, float))
        or not isfinite(result.confidence)
        or not 0.0 <= result.confidence <= 1.0
    ):
        raise InferenceContractError(
            "prediction confidence must be finite and within [0.0, 1.0]"
        )
    predicted_probability = result.probabilities[result.predicted_class]
    if not isclose(result.confidence, predicted_probability, abs_tol=1e-7):
        raise InferenceContractError(
            "prediction confidence must match the predicted-state probability"
        )
    if not isclose(result.confidence, max(probability_values), abs_tol=1e-7):
        raise InferenceContractError(
            "prediction confidence must be the maximum class probability"
        )
