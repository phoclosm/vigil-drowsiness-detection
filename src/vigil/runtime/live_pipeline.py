"""Composition boundary for vision preprocessing and temporal inference."""

from __future__ import annotations

from dataclasses import dataclass

from vigil.runtime.capture import CapturedFrame
from vigil.runtime.eye_features import EyeFeatureFrame
from vigil.runtime.inference import LivePrediction, ModelInferenceAdapter
from vigil.runtime.performance import FrameLatency, StageLatency
from vigil.runtime.preprocessing import VisionPreprocessor


MODEL_INFERENCE_STAGE_NAME = "model_inference"


class LivePipelineError(RuntimeError):
    """Raised when composed runtime stages return inconsistent frame data."""


@dataclass(frozen=True, slots=True)
class LivePipelineResult:
    """Captured frame, derived features, prediction, and measured latency."""

    frame: CapturedFrame
    eye_features: EyeFeatureFrame
    prediction: LivePrediction
    latency: FrameLatency


class LiveInferencePipeline:
    """Run one captured frame through M1 vision and M2 model inference."""

    def __init__(
        self,
        preprocessor: VisionPreprocessor,
        inference: ModelInferenceAdapter,
    ) -> None:
        if not callable(getattr(preprocessor, "process", None)):
            raise TypeError("preprocessor must provide process")
        if not callable(getattr(preprocessor, "reset", None)):
            raise TypeError("preprocessor must provide reset")
        if not callable(getattr(inference, "process", None)):
            raise TypeError("inference must provide process")
        if not callable(getattr(inference, "reset", None)):
            raise TypeError("inference must provide reset")
        self._preprocessor = preprocessor
        self._inference = inference

    def reset(self) -> None:
        """Start a new vision and inference session."""
        self._preprocessor.reset()
        self._inference.reset()

    def process(self, frame: CapturedFrame) -> LivePipelineResult:
        """Process one frame without owning capture, display, or GUI events."""
        if not isinstance(frame, CapturedFrame):
            raise TypeError("frame must be a CapturedFrame")

        preprocessed = self._preprocessor.process(frame)
        if preprocessed.eye_features.frame_index != frame.index:
            raise LivePipelineError(
                "preprocessing feature index does not match captured frame"
            )
        if preprocessed.latency.frame_index != frame.index:
            raise LivePipelineError(
                "preprocessing latency index does not match captured frame"
            )

        prediction = self._inference.process(preprocessed.eye_features)
        inference_latency = StageLatency(
            name=MODEL_INFERENCE_STAGE_NAME,
            duration_ms=prediction.result.latency_ms,
        )
        latency = FrameLatency(
            frame_index=frame.index,
            stages=(*preprocessed.latency.stages, inference_latency),
        )
        return LivePipelineResult(
            frame=frame,
            eye_features=preprocessed.eye_features,
            prediction=prediction,
            latency=latency,
        )
