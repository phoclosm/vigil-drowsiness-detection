"""Tests for composing vision preprocessing with temporal inference."""

import pytest

from vigil.ml.inference import STATUS_INSUFFICIENT_HISTORY, InferenceResult
from vigil.runtime.capture import CapturedFrame, FrameTiming
from vigil.runtime.eye_features import EyeFeatureFrame
from vigil.runtime.inference import LivePrediction
from vigil.runtime.live_pipeline import (
    MODEL_INFERENCE_STAGE_NAME,
    LiveInferencePipeline,
    LivePipelineError,
)
from vigil.runtime.performance import FrameLatency, StageLatency
from vigil.runtime.preprocessing import (
    EYE_FEATURE_STAGE_NAME,
    FACE_LANDMARK_STAGE_NAME,
    VisionPreprocessingResult,
)


class FakePreprocessor:
    def __init__(
        self,
        result: VisionPreprocessingResult,
        events: list[str],
    ) -> None:
        self.result = result
        self.events = events

    def process(self, frame: CapturedFrame) -> VisionPreprocessingResult:
        self.events.append(f"preprocess:{frame.index}")
        return self.result

    def reset(self) -> None:
        self.events.append("preprocess:reset")


class FakeInferenceAdapter:
    def __init__(self, prediction: LivePrediction, events: list[str]) -> None:
        self.prediction = prediction
        self.events = events

    def process(self, features: EyeFeatureFrame) -> LivePrediction:
        self.events.append(f"inference:{features.frame_index}")
        return self.prediction

    def reset(self) -> None:
        self.events.append("inference:reset")


def make_frame(index: int = 0) -> CapturedFrame:
    return CapturedFrame(
        image=object(),
        index=index,
        timing=FrameTiming(
            timestamp_seconds=float(index),
            delta_seconds=None if index == 0 else 1.0,
            frames_per_second=None if index == 0 else 1.0,
        ),
    )


def make_features(index: int = 0) -> EyeFeatureFrame:
    return EyeFeatureFrame(
        frame_index=index,
        timestamp_ms=float(index * 40),
        face_detected=True,
        landmarks_valid=True,
        ear_left=0.30,
        ear_right=0.30,
        ear_avg=0.30,
        is_eye_closed=False,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=None,
        blink_completed=False,
    )


def make_preprocessing_result(index: int = 0) -> VisionPreprocessingResult:
    return VisionPreprocessingResult(
        eye_features=make_features(index),
        latency=FrameLatency(
            frame_index=index,
            stages=(
                StageLatency(FACE_LANDMARK_STAGE_NAME, 2.0),
                StageLatency(EYE_FEATURE_STAGE_NAME, 1.0),
            ),
        ),
    )


def make_prediction(index: int = 0) -> LivePrediction:
    result = InferenceResult(
        predicted_class=None,
        class_id=None,
        probabilities=None,
        confidence=None,
        status=STATUS_INSUFFICIENT_HISTORY,
        latency_ms=0.0,
        reason="buffer_warming_up",
        session_id="live-session",
        buffer_size=index + 1,
        start_frame_index=0,
        end_frame_index=index,
        start_timestamp_ms=0.0,
        end_timestamp_ms=float(index * 40),
    )
    return LivePrediction(
        result=result,
        state=None,
        confidence=None,
        source_frame_index=None,
        is_fresh=False,
    )


def test_pipeline_runs_preprocessing_before_inference() -> None:
    events: list[str] = []
    frame = make_frame()
    preprocessing = make_preprocessing_result()
    prediction = make_prediction()
    pipeline = LiveInferencePipeline(
        FakePreprocessor(preprocessing, events),  # type: ignore[arg-type]
        FakeInferenceAdapter(prediction, events),  # type: ignore[arg-type]
    )

    result = pipeline.process(frame)

    assert events == ["preprocess:0", "inference:0"]
    assert result.frame is frame
    assert result.eye_features is preprocessing.eye_features
    assert result.prediction is prediction
    assert [stage.name for stage in result.latency.stages] == [
        FACE_LANDMARK_STAGE_NAME,
        EYE_FEATURE_STAGE_NAME,
        MODEL_INFERENCE_STAGE_NAME,
    ]
    assert result.latency.stage(MODEL_INFERENCE_STAGE_NAME).duration_ms == 0.0
    assert result.latency.measured_latency_ms == pytest.approx(3.0)


def test_pipeline_uses_only_measured_model_latency() -> None:
    events: list[str] = []
    prediction = make_prediction()
    measured = InferenceResult(
        predicted_class=prediction.result.predicted_class,
        class_id=prediction.result.class_id,
        probabilities=prediction.result.probabilities,
        confidence=prediction.result.confidence,
        status=prediction.result.status,
        latency_ms=4.5,
        reason=prediction.result.reason,
        session_id=prediction.result.session_id,
        buffer_size=prediction.result.buffer_size,
        start_frame_index=prediction.result.start_frame_index,
        end_frame_index=prediction.result.end_frame_index,
        start_timestamp_ms=prediction.result.start_timestamp_ms,
        end_timestamp_ms=prediction.result.end_timestamp_ms,
    )
    prediction = LivePrediction(
        result=measured,
        state=None,
        confidence=None,
        source_frame_index=None,
        is_fresh=False,
    )
    pipeline = LiveInferencePipeline(
        FakePreprocessor(make_preprocessing_result(), events),  # type: ignore[arg-type]
        FakeInferenceAdapter(prediction, events),  # type: ignore[arg-type]
    )

    result = pipeline.process(make_frame())

    assert result.latency.stage(
        MODEL_INFERENCE_STAGE_NAME
    ).duration_ms == pytest.approx(4.5)


def test_pipeline_reset_propagates_in_stage_order() -> None:
    events: list[str] = []
    pipeline = LiveInferencePipeline(
        FakePreprocessor(make_preprocessing_result(), events),  # type: ignore[arg-type]
        FakeInferenceAdapter(make_prediction(), events),  # type: ignore[arg-type]
    )

    pipeline.reset()

    assert events == ["preprocess:reset", "inference:reset"]


def test_pipeline_rejects_mismatched_feature_index() -> None:
    events: list[str] = []
    pipeline = LiveInferencePipeline(
        FakePreprocessor(make_preprocessing_result(index=1), events),  # type: ignore[arg-type]
        FakeInferenceAdapter(make_prediction(index=1), events),  # type: ignore[arg-type]
    )

    with pytest.raises(LivePipelineError, match="feature index"):
        pipeline.process(make_frame(index=0))

    assert events == ["preprocess:0"]
