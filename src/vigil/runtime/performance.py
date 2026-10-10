"""Measured runtime performance records for Vigil's vision pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Callable, Iterator, TypeVar


Clock = Callable[[], float]
FrameInput = TypeVar("FrameInput")
CAPTURE_STAGE_NAME = "capture"


class PerformanceMeasurementError(RuntimeError):
    """Raised when a timing source cannot produce a valid measurement."""


@dataclass(frozen=True, slots=True)
class StageLatency:
    """One measured pipeline-stage duration in milliseconds."""

    name: str
    duration_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("stage name must be a non-empty string")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, (int, float))
            or not isfinite(self.duration_ms)
            or self.duration_ms < 0.0
        ):
            raise ValueError("stage duration_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FrameLatency:
    """Ordered stage measurements for one processed frame."""

    frame_index: int
    stages: tuple[StageLatency, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique within a frame")

    @property
    def measured_latency_ms(self) -> float:
        """Return the sum of measured stage durations without extrapolation."""
        return sum(stage.duration_ms for stage in self.stages)

    def stage(self, name: str) -> StageLatency:
        """Return a named measurement or raise when the stage was not recorded."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"stage was not measured: {name}")


class StageLatencyRecorder:
    """Measure sequential pipeline stages with an injectable monotonic clock."""

    def __init__(self, clock: Clock = perf_counter) -> None:
        self._clock = clock
        self._stages: list[StageLatency] = []
        self._active_stage: str | None = None

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure one named stage around a block of real work."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be a non-empty string")
        if self._active_stage is not None:
            raise PerformanceMeasurementError(
                "stage measurements must not overlap or nest"
            )
        if any(stage.name == name for stage in self._stages):
            raise ValueError(f"stage has already been measured: {name}")

        self._active_stage = name
        start = self._read_clock()
        try:
            yield
        finally:
            finish = self._read_clock()
            self._active_stage = None
            duration_seconds = finish - start
            if duration_seconds < 0.0:
                raise PerformanceMeasurementError("performance clock moved backwards")
            self._stages.append(
                StageLatency(name=name, duration_ms=duration_seconds * 1000.0)
            )

    def snapshot(self, frame_index: int) -> FrameLatency:
        """Freeze the measurements collected for one frame."""
        if self._active_stage is not None:
            raise PerformanceMeasurementError(
                "cannot snapshot while a stage measurement is active"
            )
        return FrameLatency(frame_index=frame_index, stages=tuple(self._stages))

    def _read_clock(self) -> float:
        return _read_clock(self._clock)


@dataclass(frozen=True, slots=True)
class StageLatencySummary:
    """Aggregate measured latency statistics for one named stage."""

    name: str
    sample_count: int
    minimum_ms: float
    mean_ms: float
    maximum_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measured throughput and stage latencies from one benchmark run."""

    frame_latencies: tuple[FrameLatency, ...]
    elapsed_seconds: float
    frames_per_second: float | None
    stage_summaries: tuple[StageLatencySummary, ...]

    @property
    def frames_processed(self) -> int:
        return len(self.frame_latencies)

    def stage(self, name: str) -> StageLatencySummary:
        """Return aggregate measurements for a named stage."""
        for summary in self.stage_summaries:
            if summary.name == name:
                return summary
        raise KeyError(f"stage was not measured: {name}")


class BenchmarkRunner:
    """Run a bounded capture-and-processing benchmark using measured time."""

    def __init__(self, clock: Clock = perf_counter) -> None:
        self._clock = clock

    def run(
        self,
        *,
        read_frame: Callable[[], FrameInput | None],
        process_frame: Callable[[FrameInput], FrameLatency],
        max_frames: int,
    ) -> BenchmarkResult:
        """Measure at most ``max_frames`` from a pull-based frame source."""
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or max_frames < 1
        ):
            raise ValueError("max_frames must be a positive integer")

        run_start = _read_clock(self._clock)
        frames: list[FrameLatency] = []
        for _ in range(max_frames):
            capture_start = _read_clock(self._clock)
            frame = read_frame()
            capture_finish = _read_clock(self._clock)
            capture_seconds = capture_finish - capture_start
            if capture_seconds < 0.0:
                raise PerformanceMeasurementError("performance clock moved backwards")
            if frame is None:
                break

            processed = process_frame(frame)
            capture_latency = StageLatency(
                name=CAPTURE_STAGE_NAME,
                duration_ms=capture_seconds * 1000.0,
            )
            frames.append(
                FrameLatency(
                    frame_index=processed.frame_index,
                    stages=(capture_latency, *processed.stages),
                )
            )

        run_finish = _read_clock(self._clock)
        elapsed_seconds = run_finish - run_start
        if elapsed_seconds < 0.0:
            raise PerformanceMeasurementError("performance clock moved backwards")
        frames_per_second = (
            len(frames) / elapsed_seconds if frames and elapsed_seconds > 0.0 else None
        )
        return BenchmarkResult(
            frame_latencies=tuple(frames),
            elapsed_seconds=elapsed_seconds,
            frames_per_second=frames_per_second,
            stage_summaries=_summarize_stages(frames),
        )


def format_benchmark_result(result: BenchmarkResult) -> str:
    """Format measured results for terminal output without adding estimates."""
    fps = (
        f"{result.frames_per_second:.3f}"
        if result.frames_per_second is not None
        else "unavailable"
    )
    lines = [
        f"Frames processed: {result.frames_processed}",
        f"Elapsed seconds: {result.elapsed_seconds:.6f}",
        f"Measured FPS: {fps}",
        "Stage latency ms (min / mean / max):",
    ]
    if not result.stage_summaries:
        lines.append("  unavailable")
    for stage in result.stage_summaries:
        lines.append(
            f"  {stage.name}: {stage.minimum_ms:.3f} / {stage.mean_ms:.3f} / "
            f"{stage.maximum_ms:.3f} (n={stage.sample_count})"
        )
    return "\n".join(lines)


def _summarize_stages(
    frames: list[FrameLatency],
) -> tuple[StageLatencySummary, ...]:
    samples: dict[str, list[float]] = {}
    for frame in frames:
        for stage in frame.stages:
            samples.setdefault(stage.name, []).append(stage.duration_ms)
    return tuple(
        StageLatencySummary(
            name=name,
            sample_count=len(durations),
            minimum_ms=min(durations),
            mean_ms=sum(durations) / len(durations),
            maximum_ms=max(durations),
        )
        for name, durations in samples.items()
    )


def _read_clock(clock: Clock) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise PerformanceMeasurementError(
            "performance clock must return a finite numeric value"
        )
    return float(value)
