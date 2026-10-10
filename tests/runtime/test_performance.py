"""Deterministic tests for measured runtime performance records."""

from collections.abc import Iterator

import pytest

from vigil.runtime.performance import (
    CAPTURE_STAGE_NAME,
    BenchmarkRunner,
    FrameLatency,
    PerformanceMeasurementError,
    StageLatency,
    StageLatencyRecorder,
    format_benchmark_result,
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


def test_benchmark_runner_measures_fps_and_aggregates_stages() -> None:
    clock = clock_from([0.0, 0.001, 0.003, 0.010, 0.014, 0.050])
    source = iter([0, 1])
    runner = BenchmarkRunner(clock=lambda: next(clock))

    result = runner.run(
        read_frame=lambda: next(source),
        process_frame=lambda index: FrameLatency(
            frame_index=index,
            stages=(StageLatency("vision_preprocessing", 10.0 + index * 10.0),),
        ),
        max_frames=2,
    )

    assert result.frames_processed == 2
    assert result.elapsed_seconds == pytest.approx(0.050)
    assert result.frames_per_second == pytest.approx(40.0)
    capture = result.stage(CAPTURE_STAGE_NAME)
    assert capture.sample_count == 2
    assert capture.minimum_ms == pytest.approx(2.0)
    assert capture.mean_ms == pytest.approx(3.0)
    assert capture.maximum_ms == pytest.approx(4.0)
    preprocessing = result.stage("vision_preprocessing")
    assert preprocessing.mean_ms == pytest.approx(15.0)


def test_benchmark_runner_reports_unavailable_rate_for_empty_source() -> None:
    clock = clock_from([0.0, 0.001, 0.002, 0.003])
    runner = BenchmarkRunner(clock=lambda: next(clock))

    result = runner.run(
        read_frame=lambda: None,
        process_frame=lambda _: FrameLatency(frame_index=0, stages=()),
        max_frames=5,
    )

    assert result.frames_processed == 0
    assert result.frames_per_second is None
    assert result.stage_summaries == ()
    assert "Measured FPS: unavailable" in format_benchmark_result(result)


@pytest.mark.parametrize("max_frames", [0, -1, True, 1.5])
def test_benchmark_runner_requires_positive_frame_limit(max_frames: object) -> None:
    runner = BenchmarkRunner()

    with pytest.raises(ValueError, match="positive integer"):
        runner.run(
            read_frame=lambda: None,
            process_frame=lambda _: FrameLatency(frame_index=0, stages=()),
            max_frames=max_frames,  # type: ignore[arg-type]
        )


def test_benchmark_runner_rejects_backwards_capture_clock() -> None:
    clock = clock_from([0.0, 2.0, 1.0])
    runner = BenchmarkRunner(clock=lambda: next(clock))

    with pytest.raises(PerformanceMeasurementError, match="moved backwards"):
        runner.run(
            read_frame=lambda: object(),
            process_frame=lambda _: FrameLatency(frame_index=0, stages=()),
            max_frames=1,
        )


def test_benchmark_output_contains_only_measured_summary_values() -> None:
    clock = clock_from([0.0, 0.001, 0.003, 0.050])
    runner = BenchmarkRunner(clock=lambda: next(clock))
    result = runner.run(
        read_frame=lambda: "frame",
        process_frame=lambda _: FrameLatency(
            frame_index=0,
            stages=(StageLatency("vision_preprocessing", 12.5),),
        ),
        max_frames=1,
    )

    output = format_benchmark_result(result)
    assert "Frames processed: 1" in output
    assert "Elapsed seconds: 0.050000" in output
    assert "Measured FPS: 20.000" in output
    assert "capture: 2.000 / 2.000 / 2.000 (n=1)" in output
    assert "vision_preprocessing: 12.500 / 12.500 / 12.500 (n=1)" in output
