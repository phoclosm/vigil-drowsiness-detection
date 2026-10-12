"""Stateful rolling window inference engine for Vigil M2 temporal sequence models."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from vigil.ml.data.schema import LiveInputFrame, SampleRecord
from vigil.ml.dataset.config import WindowConfig
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.dataset.windowing import build_sample_feature_vector
from vigil.ml.models.export import LoadedModelBundle, load_model_bundle
from vigil.ml.models.gru import GRUTemporalClassifier

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Published status codes (§5.3 of docs/m2-data-contract.md)
# ---------------------------------------------------------------------------

STATUS_OK = "STATUS_OK"
"""Inference succeeded; a valid prediction and probability distribution were produced."""

STATUS_INSUFFICIENT_HISTORY = "STATUS_INSUFFICIENT_HISTORY"
"""Temporal buffer is filling (warm-up phase) or waiting for the next stride step."""

STATUS_INVALID_INPUT = "STATUS_INVALID_INPUT"
"""Input frame is structurally invalid, or the sequence was reset due to a
discontinuity; no prediction can be produced."""

STATUS_ERROR = "STATUS_ERROR"
"""An internal inference failure prevented a valid prediction."""

# Mapping from internal reason strings to published status codes.
_REASON_TO_STATUS: dict[str, str] = {
    "predicted": STATUS_OK,
    "buffer_warming_up": STATUS_INSUFFICIENT_HISTORY,
    "stride_skip": STATUS_INSUFFICIENT_HISTORY,
    "session_reset": STATUS_INSUFFICIENT_HISTORY,
    "discontinuity_frame_gap": STATUS_INSUFFICIENT_HISTORY,
    "discontinuity_timing_gap": STATUS_INSUFFICIENT_HISTORY,
    "invalid_input": STATUS_INVALID_INPUT,
    "inference_error": STATUS_ERROR,
}

CANONICAL_CLASS_LABELS: dict[int, str] = {
    0: "ACTIVE",
    1: "DROWSY",
    2: "SLEEPING",
}

# Union type for accepted inference inputs.
InputFrame = SampleRecord | LiveInputFrame


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceResult:
    """Structured M2→M1 inference result conforming to docs/m2-data-contract.md §5.

    Required contract fields
    ------------------------
    predicted_class : str | None
        Top-1 predicted class name (``'ACTIVE'``, ``'DROWSY'``, ``'SLEEPING'``)
        when ``status == STATUS_OK``, otherwise ``None``.
    class_id : int | None
        Canonical class index (``0``, ``1``, ``2``) when ``status == STATUS_OK``,
        otherwise ``None``.
    probabilities : dict[str, float] | None
        Softmax probability distribution over canonical classes when
        ``status == STATUS_OK``, otherwise ``None``.
    confidence : float | None
        Top-1 confidence score (``max(probabilities)``) when
        ``status == STATUS_OK``, otherwise ``None``.
    status : str
        One of ``STATUS_OK``, ``STATUS_INSUFFICIENT_HISTORY``,
        ``STATUS_INVALID_INPUT``, or ``STATUS_ERROR``.  Present on every
        response.
    latency_ms : float
        Wall-clock execution time for feature preprocessing and neural
        inference in milliseconds.  Measured from just before feature
        extraction to just after ``torch.softmax`` returns.  Zero for
        non-prediction steps where no inference is performed.  Always
        finite and non-negative.

    Diagnostic fields (supplemental, do not replace contract fields)
    ----------------------------------------------------------------
    reason : str
        Internal status code providing more granular context:
        ``'predicted'``, ``'buffer_warming_up'``, ``'stride_skip'``,
        ``'session_reset'``, ``'discontinuity_frame_gap'``,
        ``'discontinuity_timing_gap'``, ``'inference_error'``.
    session_id : str | None
        Active session identifier.
    buffer_size : int
        Number of valid frames currently held in the rolling buffer.
    start_frame_index : int | None
        Oldest frame index in the active window.
    end_frame_index : int | None
        Newest frame index in the active window.
    start_timestamp_ms : float | None
        Timestamp of the oldest frame in the active window.
    end_timestamp_ms : float | None
        Timestamp of the newest frame in the active window.
    """

    # --- Required contract fields ---
    predicted_class: str | None
    class_id: int | None
    probabilities: dict[str, float] | None
    confidence: float | None
    status: str
    latency_ms: float

    # --- Diagnostic fields ---
    reason: str
    session_id: str | None
    buffer_size: int
    start_frame_index: int | None
    end_frame_index: int | None
    start_timestamp_ms: float | None
    end_timestamp_ms: float | None

    # ------------------------------------------------------------------
    # Back-compat aliases — map old RuntimeInferenceResult field names.
    # ------------------------------------------------------------------

    @property
    def has_prediction(self) -> bool:
        """``True`` when ``status == STATUS_OK``."""
        return self.status == STATUS_OK

    @property
    def label(self) -> str | None:
        """Alias for ``predicted_class`` (backward compat)."""
        return self.predicted_class

    @property
    def label_id(self) -> int | None:
        """Alias for ``class_id`` (backward compat)."""
        return self.class_id


# Backward-compatible alias so existing import sites keep working.
RuntimeInferenceResult = InferenceResult


# ---------------------------------------------------------------------------
# Feature-vector adapter
# ---------------------------------------------------------------------------


def _build_feature_vector(
    frame: InputFrame,
    normalizer: FeatureNormalizer,
) -> list[float]:
    """Delegate feature extraction for both SampleRecord and LiveInputFrame.

    ``build_sample_feature_vector`` only requires the fields that
    ``LiveInputFrame`` exposes (``landmarks_valid``, ``ear_avg``,
    ``is_eye_closed``, ``blink_duration_ms``, ``perclos``), so we can pass
    either type directly once mypy is satisfied via duck typing.
    """
    # Both types share the exact attribute names consumed by the builder.
    return build_sample_feature_vector(frame, normalizer)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TemporalInferenceEngine:
    """Stateful, rolling-window runtime inference engine for Vigil M2 GRU models.

    Invariants and Contracts
    ------------------------
    - Accepts both :class:`SampleRecord` (labelled training records) and
      :class:`LiveInputFrame` (unlabelled live camera frames) as input.
    - Maintains a bounded rolling buffer of 60 valid timesteps.
    - Transforms inputs into locked 8-channel features using the fitted
      normalizer.
    - Emits no prediction until 60 continuous valid frames have accumulated.
    - Emits subsequent predictions every S=15 valid frames across the latest
      60 frames.
    - Detects frame-index gaps (> 1) and timing gaps (> 200 ms); resets
      buffer on discontinuity.
    - Resets buffer when encountering session boundary changes.
    - Rejects duplicate frame indices or non-monotonic timestamps with a
      clear ``ValueError``.
    - Every response contains all six required contract fields; ``None`` is
      used for prediction-dependent values when no prediction exists.
    - ``latency_ms`` is measured from just before feature extraction to just
      after ``torch.softmax`` returns; it is zero for non-prediction steps.
    - Missing ground-truth annotation on a ``LiveInputFrame`` is **never**
      treated as invalid sensor data.
    """

    def __init__(
        self,
        model: GRUTemporalClassifier,
        normalizer: FeatureNormalizer,
        config: WindowConfig | None = None,
    ) -> None:
        """Initialize the temporal inference engine with a trained model and normalizer.

        Args:
            model: Trained GRUTemporalClassifier instance.
            normalizer: Fitted FeatureNormalizer instance.
            config: Optional WindowConfig (default enforces W=60, S=15).

        Raises:
            ImportError: If PyTorch is unavailable.
            TypeError: If model or normalizer have invalid types.
            ValueError: If normalizer is not fitted.
        """
        if not HAS_TORCH or torch is None:
            raise ImportError(
                "PyTorch is required for TemporalInferenceEngine. "
                "Install PyTorch with 'pip install torch'."
            )

        if not isinstance(model, GRUTemporalClassifier):
            raise TypeError(f"model must be a GRUTemporalClassifier, got {type(model).__name__}.")

        if not isinstance(normalizer, FeatureNormalizer):
            raise TypeError(f"normalizer must be a FeatureNormalizer, got {type(normalizer).__name__}.")

        if not normalizer.is_fitted:
            raise ValueError("TemporalInferenceEngine requires a fitted FeatureNormalizer.")

        self.model = model
        self.model.eval()
        self.normalizer = normalizer
        self.config = config if config is not None else WindowConfig()

        self._buffer: deque[InputFrame] = deque(maxlen=self.config.window_length)
        self._prev_frame: InputFrame | None = None
        self._samples_since_last_prediction = 0
        self._has_ever_predicted = False
        self._current_session_id: str | None = None

    @classmethod
    def from_bundle(
        cls,
        bundle_or_path: LoadedModelBundle | str | Path,
    ) -> TemporalInferenceEngine:
        """Factory creating an inference engine directly from a model bundle or file path.

        Args:
            bundle_or_path: LoadedModelBundle instance or path to a serialized `.pt` bundle.

        Returns:
            Configured and initialized TemporalInferenceEngine.
        """
        if isinstance(bundle_or_path, (str, Path)):
            bundle = load_model_bundle(bundle_or_path)
        elif isinstance(bundle_or_path, LoadedModelBundle):
            bundle = bundle_or_path
        else:
            raise TypeError(
                f"Expected LoadedModelBundle, str, or Path, got {type(bundle_or_path).__name__}."
            )

        return cls(
            model=bundle.model,
            normalizer=bundle.normalizer,
            config=bundle.window_config,
        )

    @property
    def buffer_size(self) -> int:
        """Return the current number of valid frames in the rolling buffer."""
        return len(self._buffer)

    @property
    def current_session_id(self) -> str | None:
        """Return the active recording session identifier, if any."""
        return self._current_session_id

    def reset(self) -> None:
        """Reset the internal rolling window buffer and temporal sequence state."""
        self._buffer.clear()
        self._prev_frame = None
        self._samples_since_last_prediction = 0
        self._has_ever_predicted = False
        self._current_session_id = None

    def process_sample(self, sample: InputFrame) -> InferenceResult:
        """Process an incoming frame (alias for step)."""
        return self.step(sample)

    def step(self, sample: InputFrame) -> InferenceResult:
        """Process a single incoming frame and emit a contract-conforming inference result.

        Accepts both :class:`SampleRecord` (labelled training records used in
        offline testing) and :class:`LiveInputFrame` (unlabelled live camera
        frames from M1).  Missing ground-truth annotation on a
        ``LiveInputFrame`` is **never** treated as invalid sensor data.

        Args:
            sample: Incoming :class:`SampleRecord` or :class:`LiveInputFrame`.

        Returns:
            :class:`InferenceResult` containing all six required contract
            fields and supplemental diagnostic information.

        Raises:
            TypeError: If sample is neither a SampleRecord nor a LiveInputFrame.
            ValueError: If schema validation fails or sequence monotonicity is
                violated within the same session.
        """
        if not isinstance(sample, (SampleRecord, LiveInputFrame)):
            raise TypeError(
                f"sample must be a SampleRecord or LiveInputFrame, got {type(sample).__name__}."
            )

        # Enforce schema validation for whichever type was supplied. A
        # structurally invalid frame is a contract response, not a prediction.
        try:
            sample.validate()
        except (TypeError, ValueError):
            return InferenceResult(
                predicted_class=None,
                class_id=None,
                probabilities=None,
                confidence=None,
                status=STATUS_INVALID_INPUT,
                latency_ms=0.0,
                reason="invalid_input",
                session_id=getattr(sample, "session_id", None),
                buffer_size=len(self._buffer),
                start_frame_index=self._buffer[0].frame_index if self._buffer else None,
                end_frame_index=getattr(sample, "frame_index", None),
                start_timestamp_ms=self._buffer[0].timestamp_ms if self._buffer else None,
                end_timestamp_ms=getattr(sample, "timestamp_ms", None),
            )

        # Helper: build a no-prediction InferenceResult with zero latency.
        def _no_pred(reason: str, buf_len: int, first: InputFrame | None) -> InferenceResult:
            return InferenceResult(
                predicted_class=None,
                class_id=None,
                probabilities=None,
                confidence=None,
                status=_REASON_TO_STATUS[reason],
                latency_ms=0.0,
                reason=reason,
                session_id=sample.session_id,
                buffer_size=buf_len,
                start_frame_index=first.frame_index if first is not None else sample.frame_index,
                end_frame_index=sample.frame_index,
                start_timestamp_ms=first.timestamp_ms if first is not None else sample.timestamp_ms,
                end_timestamp_ms=sample.timestamp_ms,
            )

        # 1. Session boundary check
        if self._current_session_id is not None and sample.session_id != self._current_session_id:
            self.reset()
            self._current_session_id = sample.session_id
            self._buffer.append(sample)
            self._prev_frame = sample
            self._samples_since_last_prediction = 1
            return _no_pred("session_reset", 1, sample)

        self._current_session_id = sample.session_id

        # 2. Sequence integrity and gap checks
        if self._prev_frame is not None:
            # Monotonicity checks
            if sample.frame_index <= self._prev_frame.frame_index:
                raise ValueError(
                    f"Duplicate or decreasing frame index in session '{sample.session_id}': "
                    f"previous frame {self._prev_frame.frame_index} followed by {sample.frame_index}."
                )
            if sample.timestamp_ms <= self._prev_frame.timestamp_ms:
                raise ValueError(
                    f"Non-monotonic timestamp detected in session '{sample.session_id}': "
                    f"frame {self._prev_frame.frame_index} ({self._prev_frame.timestamp_ms}ms) "
                    f"followed by frame {sample.frame_index} ({sample.timestamp_ms}ms)."
                )

            frame_gap = sample.frame_index - self._prev_frame.frame_index
            time_gap_ms = sample.timestamp_ms - self._prev_frame.timestamp_ms

            # Discontinuity checks (frame drop or capture pause)
            if frame_gap > self.config.max_frame_gap:
                self._buffer.clear()
                self._samples_since_last_prediction = 1
                self._has_ever_predicted = False
                self._buffer.append(sample)
                self._prev_frame = sample
                return _no_pred("discontinuity_frame_gap", 1, sample)

            if time_gap_ms > self.config.max_time_gap_ms:
                self._buffer.clear()
                self._samples_since_last_prediction = 1
                self._has_ever_predicted = False
                self._buffer.append(sample)
                self._prev_frame = sample
                return _no_pred("discontinuity_timing_gap", 1, sample)

        # 3. Append to buffer
        self._buffer.append(sample)
        self._prev_frame = sample
        self._samples_since_last_prediction += 1

        buf_len = len(self._buffer)
        first_frame = self._buffer[0]

        # 4. Warm-up check (fewer than 60 frames)
        if buf_len < self.config.window_length:
            return _no_pred("buffer_warming_up", buf_len, first_frame)

        # 5. Stride check (S=15)
        if self._has_ever_predicted and self._samples_since_last_prediction < self.config.stride:
            return _no_pred("stride_skip", buf_len, first_frame)

        # 6. Execute forward inference — latency measurement covers steps 6 only.
        _t_start = time.perf_counter()
        try:
            feature_rows = [_build_feature_vector(s, self.normalizer) for s in self._buffer]
            tensor_x = torch.tensor([feature_rows], dtype=torch.float32)

            with torch.no_grad():
                logits = self.model(tensor_x)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_id = int(torch.argmax(logits, dim=-1)[0].item())

        except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
            latency_ms = max(0.0, (time.perf_counter() - _t_start) * 1000.0)
            return InferenceResult(
                predicted_class=None,
                class_id=None,
                probabilities=None,
                confidence=None,
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                reason="inference_error",
                session_id=sample.session_id,
                buffer_size=buf_len,
                start_frame_index=first_frame.frame_index,
                end_frame_index=sample.frame_index,
                start_timestamp_ms=first_frame.timestamp_ms,
                end_timestamp_ms=sample.timestamp_ms,
            )

        latency_ms = max(0.0, (time.perf_counter() - _t_start) * 1000.0)
        probabilities = [float(value.item()) for value in probs]
        if (
            len(probabilities) != len(CANONICAL_CLASS_LABELS)
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
            or pred_id not in CANONICAL_CLASS_LABELS
            or not math.isfinite(latency_ms)
        ):
            return InferenceResult(
                predicted_class=None,
                class_id=None,
                probabilities=None,
                confidence=None,
                status=STATUS_ERROR,
                latency_ms=latency_ms if math.isfinite(latency_ms) and latency_ms >= 0.0 else 0.0,
                reason="inference_error",
                session_id=sample.session_id,
                buffer_size=buf_len,
                start_frame_index=first_frame.frame_index,
                end_frame_index=sample.frame_index,
                start_timestamp_ms=first_frame.timestamp_ms,
                end_timestamp_ms=sample.timestamp_ms,
            )

        self._has_ever_predicted = True
        self._samples_since_last_prediction = 0

        pred_label = CANONICAL_CLASS_LABELS[pred_id]
        prob_dict = {
            CANONICAL_CLASS_LABELS[i]: probabilities[i] for i in range(3)
        }
        confidence = probabilities[pred_id]

        return InferenceResult(
            predicted_class=pred_label,
            class_id=pred_id,
            probabilities=prob_dict,
            confidence=confidence,
            status=STATUS_OK,
            latency_ms=latency_ms,
            reason="predicted",
            session_id=sample.session_id,
            buffer_size=buf_len,
            start_frame_index=first_frame.frame_index,
            end_frame_index=sample.frame_index,
            start_timestamp_ms=first_frame.timestamp_ms,
            end_timestamp_ms=sample.timestamp_ms,
        )
