"""Video capture interfaces for Vigil's real-time runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Protocol


class CaptureError(RuntimeError):
    """Base exception for capture failures."""


class CaptureDependencyError(CaptureError):
    """Raised when the OpenCV capture dependency is unavailable."""


class CaptureOpenError(CaptureError):
    """Raised when a configured capture source cannot be opened."""


class CaptureReadError(CaptureError):
    """Raised when a frame cannot be read from an open webcam."""


class CaptureBackend(Protocol):
    """Small subset of ``cv2.VideoCapture`` used by the runtime."""

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API compatibility
        """Return whether the source opened successfully."""

    def read(self) -> tuple[bool, Any]:
        """Read one frame from the source."""

    def release(self) -> None:
        """Release the underlying capture resource."""


CaptureSource = int | Path
BackendSource = int | str
CaptureFactory = Callable[[BackendSource], CaptureBackend]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Configuration for a webcam device or recorded video file."""

    source: CaptureSource = 0

    def __post_init__(self) -> None:
        if isinstance(self.source, bool):
            raise TypeError("capture source must be a webcam index or video path")
        if isinstance(self.source, int):
            if self.source < 0:
                raise ValueError("webcam device index must be non-negative")
            return
        if not isinstance(self.source, Path):
            raise TypeError("capture source must be a webcam index or pathlib.Path")

    @classmethod
    def webcam(cls, index: int = 0) -> CaptureConfig:
        """Create configuration for a webcam device index."""
        return cls(source=index)

    @classmethod
    def video_file(cls, path: str | Path) -> CaptureConfig:
        """Create configuration for a recorded video path."""
        if isinstance(path, str) and not path.strip():
            raise ValueError("video path must not be empty")
        return cls(source=Path(path).expanduser())

    @property
    def is_webcam(self) -> bool:
        """Return whether this configuration targets a webcam."""
        return isinstance(self.source, int)

    @property
    def is_video_file(self) -> bool:
        """Return whether this configuration targets a recorded video."""
        return isinstance(self.source, Path)

    @property
    def backend_source(self) -> BackendSource:
        """Return the source representation expected by OpenCV."""
        if isinstance(self.source, Path):
            return str(self.source)
        return self.source

    @property
    def source_label(self) -> str:
        """Return a human-readable source label for diagnostics."""
        if self.is_webcam:
            return f"webcam device {self.source}"
        return f"video file '{self.source}'"


@dataclass(frozen=True, slots=True)
class FrameTiming:
    """Timing measurements associated with one captured frame."""

    timestamp_seconds: float
    delta_seconds: float | None
    frames_per_second: float | None


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """A captured image together with its sequence and timing metadata."""

    image: Any
    index: int
    timing: FrameTiming


class FrameTimer:
    """Measure monotonic inter-frame timing without averaging or fabrication."""

    def __init__(self, clock: Clock = perf_counter) -> None:
        self._clock = clock
        self._previous_timestamp: float | None = None

    def reset(self) -> None:
        """Forget prior timing so the next frame starts a new sequence."""
        self._previous_timestamp = None

    def measure(self) -> FrameTiming:
        """Measure the current timestamp, frame interval, and instantaneous FPS."""
        timestamp = float(self._clock())
        if self._previous_timestamp is None:
            delta = None
        else:
            delta = timestamp - self._previous_timestamp
            if delta < 0:
                raise CaptureError("frame clock moved backwards")

        frames_per_second = None if not delta else 1.0 / delta
        self._previous_timestamp = timestamp
        return FrameTiming(
            timestamp_seconds=timestamp,
            delta_seconds=delta,
            frames_per_second=frames_per_second,
        )


def _open_opencv_capture(source: BackendSource) -> CaptureBackend:
    """Create an OpenCV capture backend without importing it at module load."""
    try:
        import cv2
    except ImportError as error:
        raise CaptureDependencyError(
            "OpenCV is required for video capture; install project dependencies first"
        ) from error

    return cv2.VideoCapture(source)


class CaptureStream:
    """Own the lifecycle of one webcam or recorded-video backend."""

    def __init__(
        self,
        config: CaptureConfig,
        *,
        capture_factory: CaptureFactory | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        self.config = config
        self._capture_factory = capture_factory or _open_opencv_capture
        self._backend: CaptureBackend | None = None
        self._timer = FrameTimer(clock)
        self._frame_index = 0

    @property
    def is_open(self) -> bool:
        """Return whether this stream currently owns an open backend."""
        return self._backend is not None and self._backend.isOpened()

    def open(self) -> CaptureStream:
        """Open the configured source and return this stream."""
        if self._backend is not None:
            return self

        source = self.config.source
        if isinstance(source, Path) and not source.is_file():
            raise CaptureOpenError(f"video file does not exist: '{source}'")

        backend = self._capture_factory(self.config.backend_source)
        if not backend.isOpened():
            backend.release()
            raise CaptureOpenError(f"unable to open {self.config.source_label}")

        self._backend = backend
        self._timer.reset()
        self._frame_index = 0
        return self

    def read(self) -> CapturedFrame | None:
        """Read a timed frame, returning ``None`` at the end of a video file."""
        if self._backend is None:
            raise CaptureError("capture stream must be opened before reading")

        success, frame = self._backend.read()
        if not success or frame is None:
            if self.config.is_video_file:
                return None
            raise CaptureReadError(f"unable to read from {self.config.source_label}")

        captured_frame = CapturedFrame(
            image=frame,
            index=self._frame_index,
            timing=self._timer.measure(),
        )
        self._frame_index += 1
        return captured_frame

    def frames(self) -> Iterator[CapturedFrame]:
        """Yield frames until a video ends or a webcam read fails."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        """Release the capture backend; repeated calls are safe."""
        if self._backend is not None:
            self._backend.release()
            self._backend = None

    def __enter__(self) -> CaptureStream:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()
