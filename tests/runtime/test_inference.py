"""Tests for the M1-to-M2 live inference adapter."""

from pathlib import Path

import pytest

from vigil.ml.data import LiveInputFrame
from vigil.ml.inference import (
    STATUS_ERROR,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_INVALID_INPUT,
    STATUS_OK,
    InferenceResult,
)
from vigil.runtime.eye_features import EyeFeatureFrame
from vigil.runtime.inference import (
    InferenceContractError,
    ModelInferenceAdapter,
    ModelInferenceError,
    ModelLoadError,
)


class FakeEngine:
    """Record adapter inputs and return deterministic contract results."""

    def __init__(self, results: list[InferenceResult]) -> None:
        self.results = list(results)
        self.samples: list[LiveInputFrame] = []
        self.reset_count = 0

    def process_sample(self, sample: LiveInputFrame) -> InferenceResult:
        self.samples.append(sample)
        return self.results.pop(0)

    def reset(self) -> None:
        self.reset_count += 1


def make_features(
    frame_index: int = 0,
    timestamp_ms: float = 0.0,
    *,
    landmarks_valid: bool = True,
    face_detected: bool | None = None,
) -> EyeFeatureFrame:
    ear = 0.30 if landmarks_valid else None
    return EyeFeatureFrame(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        face_detected=(
            landmarks_valid if face_detected is None else face_detected
        ),
        landmarks_valid=landmarks_valid,
        ear_left=ear,
        ear_right=ear,
        ear_avg=ear,
        is_eye_closed=False if landmarks_valid else None,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=0.0 if landmarks_valid else None,
        blink_completed=False,
    )


def make_result(
    *,
    frame_index: int,
    status: str = STATUS_INSUFFICIENT_HISTORY,
    reason: str = "buffer_warming_up",
) -> InferenceResult:
    has_prediction = status == STATUS_OK
    return InferenceResult(
        predicted_class="ACTIVE" if has_prediction else None,
        class_id=0 if has_prediction else None,
        probabilities=(
            {"ACTIVE": 0.8, "DROWSY": 0.15, "SLEEPING": 0.05}
            if has_prediction
            else None
        ),
        confidence=0.8 if has_prediction else None,
        status=status,
        latency_ms=1.25 if has_prediction else 0.0,
        reason=reason,
        session_id="live-session",
        buffer_size=frame_index + 1,
        start_frame_index=0,
        end_frame_index=frame_index,
        start_timestamp_ms=0.0,
        end_timestamp_ms=float(frame_index * 40),
    )


def test_adapter_maps_eye_features_to_unlabeled_live_input() -> None:
    engine = FakeEngine([make_result(frame_index=0)])
    adapter = ModelInferenceAdapter(engine, session_id="live-session")

    prediction = adapter.process(make_features())

    assert not prediction.is_available
    assert prediction.result.status == STATUS_INSUFFICIENT_HISTORY
    assert len(engine.samples) == 1
    sample = engine.samples[0]
    assert sample == LiveInputFrame(
        session_id="live-session",
        frame_index=0,
        timestamp_ms=0.0,
        frame_delta_ms=None,
        face_detected=True,
        landmarks_valid=True,
        ear_left=0.30,
        ear_right=0.30,
        ear_avg=0.30,
        is_eye_closed=False,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=0.0,
        fps=None,
    )
    assert not hasattr(sample, "subject_id")
    assert not hasattr(sample, "label")
    assert not hasattr(sample, "label_id")


def test_adapter_derives_frame_delta_from_feature_timestamps() -> None:
    engine = FakeEngine(
        [make_result(frame_index=0), make_result(frame_index=1)]
    )
    adapter = ModelInferenceAdapter(engine, session_id="live-session")

    adapter.process(make_features())
    adapter.process(make_features(frame_index=1, timestamp_ms=40.0))

    assert engine.samples[0].frame_delta_ms is None
    assert engine.samples[1].frame_delta_ms == pytest.approx(40.0)


def test_adapter_exposes_a_genuine_canonical_prediction() -> None:
    engine = FakeEngine(
        [make_result(frame_index=0, status=STATUS_OK, reason="predicted")]
    )
    adapter = ModelInferenceAdapter(engine, session_id="live-session")

    prediction = adapter.process(make_features())

    assert prediction.state == "ACTIVE"
    assert prediction.confidence == pytest.approx(0.8)
    assert prediction.source_frame_index == 0
    assert prediction.is_fresh
    assert prediction.is_available
    assert prediction.result.latency_ms == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (STATUS_INSUFFICIENT_HISTORY, "buffer_warming_up"),
        (STATUS_INVALID_INPUT, "invalid_input"),
        (STATUS_ERROR, "inference_error"),
    ],
)
def test_adapter_never_invents_a_state_for_no_prediction_statuses(
    status: str,
    reason: str,
) -> None:
    engine = FakeEngine(
        [make_result(frame_index=0, status=status, reason=reason)]
    )
    adapter = ModelInferenceAdapter(engine, session_id="live-session")

    prediction = adapter.process(make_features())

    assert prediction.result.status == status
    assert prediction.state is None
    assert prediction.confidence is None
    assert prediction.source_frame_index is None
    assert not prediction.is_fresh


@pytest.mark.parametrize("face_detected", [False, True])
def test_missing_landmarks_are_forwarded_without_synthetic_eye_values(
    face_detected: bool,
) -> None:
    engine = FakeEngine([make_result(frame_index=0)])
    adapter = ModelInferenceAdapter(engine, session_id="live-session")

    prediction = adapter.process(
        make_features(
            landmarks_valid=False,
            face_detected=face_detected,
        )
    )

    assert prediction.result.status == STATUS_INSUFFICIENT_HISTORY
    assert len(engine.samples) == 1
    sample = engine.samples[0]
    assert sample.face_detected is face_detected
    assert sample.landmarks_valid is False
    assert sample.ear_left is None
    assert sample.ear_right is None
    assert sample.ear_avg is None
    assert sample.is_eye_closed is None


def test_stride_skip_with_missing_landmarks_retains_last_genuine_prediction(
) -> None:
    engine = FakeEngine(
        [
            make_result(frame_index=0, status=STATUS_OK, reason="predicted"),
            make_result(
                frame_index=1,
                status=STATUS_INSUFFICIENT_HISTORY,
                reason="stride_skip",
            ),
        ]
    )
    adapter = ModelInferenceAdapter(engine, session_id="live-session")
    fresh = adapter.process(make_features())

    retained = adapter.process(
        make_features(
            frame_index=1,
            timestamp_ms=40.0,
            landmarks_valid=False,
            face_detected=False,
        )
    )

    assert fresh.is_fresh
    assert retained.result.status == STATUS_INSUFFICIENT_HISTORY
    assert retained.result.reason == "stride_skip"
    assert retained.state == "ACTIVE"
    assert retained.confidence == pytest.approx(0.8)
    assert retained.source_frame_index == 0
    assert retained.is_available
    assert not retained.is_fresh
    assert len(engine.samples) == 2
    assert engine.samples[1].landmarks_valid is False


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (STATUS_INSUFFICIENT_HISTORY, "buffer_warming_up"),
        (STATUS_INSUFFICIENT_HISTORY, "session_reset"),
        (STATUS_INSUFFICIENT_HISTORY, "discontinuity_frame_gap"),
        (STATUS_INSUFFICIENT_HISTORY, "discontinuity_timing_gap"),
        (STATUS_INVALID_INPUT, "invalid_input"),
        (STATUS_ERROR, "inference_error"),
    ],
)
def test_unusable_or_reset_result_clears_cached_prediction(
    status: str,
    reason: str,
) -> None:
    engine = FakeEngine(
        [
            make_result(frame_index=0, status=STATUS_OK, reason="predicted"),
            make_result(frame_index=1, status=status, reason=reason),
            make_result(
                frame_index=2,
                status=STATUS_INSUFFICIENT_HISTORY,
                reason="stride_skip",
            ),
        ]
    )
    adapter = ModelInferenceAdapter(engine, session_id="live-session")
    adapter.process(make_features())

    unavailable = adapter.process(make_features(frame_index=1, timestamp_ms=40.0))
    later_stride = adapter.process(make_features(frame_index=2, timestamp_ms=80.0))

    assert not unavailable.is_available
    assert not later_stride.is_available


def test_reset_clears_cached_prediction() -> None:
    engine = FakeEngine(
        [
            make_result(frame_index=0, status=STATUS_OK, reason="predicted"),
            make_result(
                frame_index=0,
                status=STATUS_INSUFFICIENT_HISTORY,
                reason="stride_skip",
            ),
        ]
    )
    adapter = ModelInferenceAdapter(engine, session_id="live-session")
    adapter.process(make_features())

    adapter.reset()
    after_reset = adapter.process(make_features())

    assert not after_reset.is_available
    assert not after_reset.is_fresh


def test_adapter_rejects_a_noncanonical_prediction() -> None:
    result = make_result(frame_index=0, status=STATUS_OK, reason="predicted")
    malformed = InferenceResult(
        predicted_class="TIRED",
        class_id=result.class_id,
        probabilities=result.probabilities,
        confidence=result.confidence,
        status=result.status,
        latency_ms=result.latency_ms,
        reason=result.reason,
        session_id=result.session_id,
        buffer_size=result.buffer_size,
        start_frame_index=result.start_frame_index,
        end_frame_index=result.end_frame_index,
        start_timestamp_ms=result.start_timestamp_ms,
        end_timestamp_ms=result.end_timestamp_ms,
    )
    adapter = ModelInferenceAdapter(
        FakeEngine([malformed]),
        session_id="live-session",
    )

    with pytest.raises(InferenceContractError, match="canonical fatigue state"):
        adapter.process(make_features())


def test_adapter_wraps_unexpected_backend_errors() -> None:
    class FailingEngine(FakeEngine):
        def process_sample(self, sample: LiveInputFrame) -> InferenceResult:
            raise RuntimeError("synthetic backend failure")

    adapter = ModelInferenceAdapter(FailingEngine([]), session_id="live-session")

    with pytest.raises(ModelInferenceError, match="frame 0"):
        adapter.process(make_features())


def test_adapter_reset_clears_timing_and_resets_engine() -> None:
    engine = FakeEngine(
        [make_result(frame_index=0), make_result(frame_index=0)]
    )
    adapter = ModelInferenceAdapter(engine, session_id="live-session")
    adapter.process(make_features())

    adapter.reset()
    adapter.process(make_features())

    assert engine.reset_count == 1
    assert [sample.frame_delta_ms for sample in engine.samples] == [None, None]


def test_adapter_requires_frame_zero_at_session_start() -> None:
    adapter = ModelInferenceAdapter(FakeEngine([]), session_id="live-session")

    with pytest.raises(InferenceContractError, match="frame_index 0"):
        adapter.process(make_features(frame_index=1, timestamp_ms=40.0))


def test_from_bundle_reports_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="does not exist"):
        ModelInferenceAdapter.from_bundle(
            tmp_path / "missing.pt",
            session_id="live-session",
        )


def test_from_bundle_delegates_to_m2_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "model.pt"
    bundle_path.touch()
    engine = FakeEngine([])
    received: list[Path] = []

    def load_bundle(path: Path) -> FakeEngine:
        received.append(path)
        return engine

    monkeypatch.setattr(
        "vigil.runtime.inference.TemporalInferenceEngine.from_bundle",
        load_bundle,
    )

    adapter = ModelInferenceAdapter.from_bundle(
        bundle_path,
        session_id="live-session",
    )

    assert adapter.session_id == "live-session"
    assert received == [bundle_path]
