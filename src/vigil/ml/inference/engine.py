"""Stateful rolling window inference engine for Vigil M2 temporal sequence models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from vigil.ml.data.schema import SampleRecord
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

CANONICAL_CLASS_LABELS: dict[int, str] = {
    0: "ACTIVE",
    1: "DROWSY",
    2: "SLEEPING",
}


@dataclass(frozen=True)
class RuntimeInferenceResult:
    """Structured result returned by the temporal inference engine for each processed sample.

    Attributes:
        has_prediction: Whether a model prediction was produced on this step.
        label: Predicted class name ('ACTIVE', 'DROWSY', 'SLEEPING') if predicted, else None.
        label_id: Canonical integer class ID (0, 1, 2) if predicted, else None.
        probabilities: Mapping from class name to predicted probability if predicted, else None.
        confidence: Probability of the predicted class if predicted, else None.
        reason: Explanatory status code:
            - 'predicted': A valid 60-frame window was evaluated.
            - 'buffer_warming_up': Fewer than 60 continuous samples in buffer.
            - 'stride_skip': Buffer is full, but waiting for next stride step (S=15).
            - 'discontinuity_frame_gap': Sequence reset due to dropped frames (delta > 1).
            - 'discontinuity_timing_gap': Sequence reset due to capture stall (delta > 200ms).
            - 'session_reset': Sequence reset due to session boundary change.
        session_id: Active session identifier.
        buffer_size: Number of valid frames currently held in the rolling window.
        start_frame_index: First frame index of the current window/buffer.
        end_frame_index: Latest frame index in the current window/buffer.
        start_timestamp_ms: Session-relative timestamp of the earliest frame.
        end_timestamp_ms: Session-relative timestamp of the latest frame.
    """

    has_prediction: bool
    label: str | None
    label_id: int | None
    probabilities: dict[str, float] | None
    confidence: float | None
    reason: str
    session_id: str | None
    buffer_size: int
    start_frame_index: int | None
    end_frame_index: int | None
    start_timestamp_ms: float | None
    end_timestamp_ms: float | None


class TemporalInferenceEngine:
    """Stateful, rolling-window runtime inference engine for Vigil M2 GRU models.

    Invariants and Contracts:
        - Maintains a bounded rolling buffer of 60 valid timesteps.
        - Transforms raw SampleRecords into locked 8-channel features using the fitted normalizer.
        - Emits no prediction until 60 continuous valid samples have accumulated.
        - Emits subsequent predictions every S=15 valid frames across the latest 60 frames.
        - Detects frame-index gaps (> 1) and timing gaps (> 200ms); resets buffer on discontinuity.
        - Resets buffer when encountering session boundary changes.
        - Rejects duplicate frame indices or non-monotonic timestamps with a clear ValueError.
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

        self._buffer: deque[SampleRecord] = deque(maxlen=self.config.window_length)
        self._prev_sample: SampleRecord | None = None
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
        """Return the current number of valid sample records in the rolling buffer."""
        return len(self._buffer)

    @property
    def current_session_id(self) -> str | None:
        """Return the active recording session identifier, if any."""
        return self._current_session_id

    def reset(self) -> None:
        """Reset the internal rolling window buffer and temporal sequence state."""
        self._buffer.clear()
        self._prev_sample = None
        self._samples_since_last_prediction = 0
        self._has_ever_predicted = False
        self._current_session_id = None

    def process_sample(self, sample: SampleRecord) -> RuntimeInferenceResult:
        """Process an incoming SampleRecord (alias for step)."""
        return self.step(sample)

    def step(self, sample: SampleRecord) -> RuntimeInferenceResult:
        """Process a single incoming SampleRecord and emit an inference result.

        Args:
            sample: Incoming SampleRecord from M1.

        Returns:
            RuntimeInferenceResult detailing prediction or abstention reason.

        Raises:
            TypeError: If sample is not a SampleRecord.
            ValueError: If sample schema validation fails or sequence monotonicity is violated.
        """
        if not isinstance(sample, SampleRecord):
            raise TypeError(f"sample must be a SampleRecord, got {type(sample).__name__}.")

        # Enforce canonical schema validation
        sample.validate()

        # 1. Session boundary check
        if self._current_session_id is not None and sample.session_id != self._current_session_id:
            self.reset()
            self._current_session_id = sample.session_id
            self._buffer.append(sample)
            self._prev_sample = sample
            self._samples_since_last_prediction = 1
            return RuntimeInferenceResult(
                has_prediction=False,
                label=None,
                label_id=None,
                probabilities=None,
                confidence=None,
                reason="session_reset",
                session_id=sample.session_id,
                buffer_size=1,
                start_frame_index=sample.frame_index,
                end_frame_index=sample.frame_index,
                start_timestamp_ms=sample.timestamp_ms,
                end_timestamp_ms=sample.timestamp_ms,
            )

        self._current_session_id = sample.session_id

        # 2. Sequence integrity and gap checks
        if self._prev_sample is not None:
            # Monotonicity checks
            if sample.frame_index <= self._prev_sample.frame_index:
                raise ValueError(
                    f"Duplicate or decreasing frame index in session '{sample.session_id}': "
                    f"previous frame {self._prev_sample.frame_index} followed by {sample.frame_index}."
                )
            if sample.timestamp_ms <= self._prev_sample.timestamp_ms:
                raise ValueError(
                    f"Non-monotonic timestamp detected in session '{sample.session_id}': "
                    f"frame {self._prev_sample.frame_index} ({self._prev_sample.timestamp_ms}ms) "
                    f"followed by frame {sample.frame_index} ({sample.timestamp_ms}ms)."
                )

            frame_gap = sample.frame_index - self._prev_sample.frame_index
            time_gap_ms = sample.timestamp_ms - self._prev_sample.timestamp_ms

            # Discontinuity checks (frame drop or capture pause)
            if frame_gap > self.config.max_frame_gap:
                self._buffer.clear()
                self._samples_since_last_prediction = 1
                self._has_ever_predicted = False
                self._buffer.append(sample)
                self._prev_sample = sample
                return RuntimeInferenceResult(
                    has_prediction=False,
                    label=None,
                    label_id=None,
                    probabilities=None,
                    confidence=None,
                    reason="discontinuity_frame_gap",
                    session_id=sample.session_id,
                    buffer_size=1,
                    start_frame_index=sample.frame_index,
                    end_frame_index=sample.frame_index,
                    start_timestamp_ms=sample.timestamp_ms,
                    end_timestamp_ms=sample.timestamp_ms,
                )

            if time_gap_ms > self.config.max_time_gap_ms:
                self._buffer.clear()
                self._samples_since_last_prediction = 1
                self._has_ever_predicted = False
                self._buffer.append(sample)
                self._prev_sample = sample
                return RuntimeInferenceResult(
                    has_prediction=False,
                    label=None,
                    label_id=None,
                    probabilities=None,
                    confidence=None,
                    reason="discontinuity_timing_gap",
                    session_id=sample.session_id,
                    buffer_size=1,
                    start_frame_index=sample.frame_index,
                    end_frame_index=sample.frame_index,
                    start_timestamp_ms=sample.timestamp_ms,
                    end_timestamp_ms=sample.timestamp_ms,
                )

        # 3. Append to buffer
        self._buffer.append(sample)
        self._prev_sample = sample
        self._samples_since_last_prediction += 1

        buf_len = len(self._buffer)
        first_frame = self._buffer[0]

        # 4. Warm-up check (fewer than 60 frames)
        if buf_len < self.config.window_length:
            return RuntimeInferenceResult(
                has_prediction=False,
                label=None,
                label_id=None,
                probabilities=None,
                confidence=None,
                reason="buffer_warming_up",
                session_id=sample.session_id,
                buffer_size=buf_len,
                start_frame_index=first_frame.frame_index,
                end_frame_index=sample.frame_index,
                start_timestamp_ms=first_frame.timestamp_ms,
                end_timestamp_ms=sample.timestamp_ms,
            )

        # 5. Stride check (S=15)
        if self._has_ever_predicted and self._samples_since_last_prediction < self.config.stride:
            return RuntimeInferenceResult(
                has_prediction=False,
                label=None,
                label_id=None,
                probabilities=None,
                confidence=None,
                reason="stride_skip",
                session_id=sample.session_id,
                buffer_size=buf_len,
                start_frame_index=first_frame.frame_index,
                end_frame_index=sample.frame_index,
                start_timestamp_ms=first_frame.timestamp_ms,
                end_timestamp_ms=sample.timestamp_ms,
            )

        # 6. Execute forward inference
        feature_rows = [build_sample_feature_vector(s, self.normalizer) for s in self._buffer]
        tensor_x = torch.tensor([feature_rows], dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(logits, dim=-1)[0].item())

        self._has_ever_predicted = True
        self._samples_since_last_prediction = 0

        pred_label = CANONICAL_CLASS_LABELS[pred_id]
        prob_dict = {
            CANONICAL_CLASS_LABELS[i]: float(probs[i].item()) for i in range(3)
        }
        confidence = float(probs[pred_id].item())

        return RuntimeInferenceResult(
            has_prediction=True,
            label=pred_label,
            label_id=pred_id,
            probabilities=prob_dict,
            confidence=confidence,
            reason="predicted",
            session_id=sample.session_id,
            buffer_size=buf_len,
            start_frame_index=first_frame.frame_index,
            end_frame_index=sample.frame_index,
            start_timestamp_ms=first_frame.timestamp_ms,
            end_timestamp_ms=sample.timestamp_ms,
        )
