"""Measured runtime performance records for Vigil's vision pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Callable, Iterator


Clock = Callable[[], float]


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
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise PerformanceMeasurementError(
                "performance clock must return a finite numeric value"
            )
        return float(value)
