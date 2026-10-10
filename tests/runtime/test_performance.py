"""Deterministic tests for measured runtime performance records."""

from collections.abc import Iterator

import pytest

from vigil.runtime.performance import (
    FrameLatency,
    PerformanceMeasurementError,
    StageLatency,
    StageLatencyRecorder,
)


def clock_from(values: list[float]) -> Iterator[float]:
    return iter(values)


def test_recorder_measures_ordered_stage_latencies() -> None:
    clock = clock_from([1.0, 1.025, 2.0, 2.005])
    recorder = StageLatencyRecorder(clock=lambda: next(clock))

    with recorder.measure("face_landmarks"):
        pass
    with recorder.measure("eye_features"):
        pass

    frame = recorder.snapshot(frame_index=7)
    assert frame.frame_index == 7
    assert tuple(stage.name for stage in frame.stages) == (
        "face_landmarks",
        "eye_features",
    )
    assert frame.stage("face_landmarks").duration_ms == pytest.approx(25.0)
    assert frame.stage("eye_features").duration_ms == pytest.approx(5.0)
    assert frame.measured_latency_ms == pytest.approx(30.0)


def test_missing_stage_lookup_is_explicit() -> None:
    frame = FrameLatency(frame_index=0, stages=())

    with pytest.raises(KeyError, match="not measured"):
        frame.stage("face_landmarks")


def test_recorder_rejects_duplicate_stage_names() -> None:
    clock = clock_from([0.0, 0.1])
    recorder = StageLatencyRecorder(clock=lambda: next(clock))
    with recorder.measure("eye_features"):
        pass

    with pytest.raises(ValueError, match="already been measured"):
        with recorder.measure("eye_features"):
            pass


def test_recorder_rejects_nested_measurements() -> None:
    clock = clock_from([0.0, 0.1])
    recorder = StageLatencyRecorder(clock=lambda: next(clock))

    with recorder.measure("outer"):
        with pytest.raises(PerformanceMeasurementError, match="overlap or nest"):
            with recorder.measure("inner"):
                pass


def test_recorder_rejects_backwards_clock() -> None:
    clock = clock_from([2.0, 1.0])
    recorder = StageLatencyRecorder(clock=lambda: next(clock))

    with pytest.raises(PerformanceMeasurementError, match="moved backwards"):
        with recorder.measure("face_landmarks"):
            pass


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0"])
def test_recorder_rejects_invalid_clock_values(value: object) -> None:
    recorder = StageLatencyRecorder(clock=lambda: value)  # type: ignore[arg-type]

    with pytest.raises(PerformanceMeasurementError, match="finite numeric"):
        with recorder.measure("face_landmarks"):
            pass


@pytest.mark.parametrize("duration", [-1.0, float("nan"), True])
def test_stage_latency_rejects_invalid_duration(duration: object) -> None:
    with pytest.raises(ValueError, match="duration_ms"):
        StageLatency("stage", duration)  # type: ignore[arg-type]


def test_frame_latency_requires_unique_stage_names() -> None:
    duplicate = StageLatency("stage", 1.0)

    with pytest.raises(ValueError, match="unique"):
        FrameLatency(frame_index=0, stages=(duplicate, duplicate))
