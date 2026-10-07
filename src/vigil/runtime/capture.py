"""Video capture interfaces for Vigil's real-time runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


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


CaptureFactory = Callable[[int], CaptureBackend]


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Configuration for a local webcam source."""

    source: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.source, bool) or not isinstance(self.source, int):
            raise TypeError("webcam source must be an integer device index")
        if self.source < 0:
            raise ValueError("webcam device index must be non-negative")

    @classmethod
    def webcam(cls, index: int = 0) -> CaptureConfig:
        """Create configuration for a webcam device index."""
        return cls(source=index)


def _open_opencv_capture(source: int) -> CaptureBackend:
    """Create an OpenCV capture backend without importing it at module load."""
    try:
        import cv2
    except ImportError as error:
        raise CaptureDependencyError(
            "OpenCV is required for video capture; install project dependencies first"
        ) from error

    return cv2.VideoCapture(source)


class CaptureStream:
    """Own the lifecycle of one webcam capture backend."""

    def __init__(
        self,
        config: CaptureConfig,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.config = config
        self._capture_factory = capture_factory or _open_opencv_capture
        self._backend: CaptureBackend | None = None

    @property
    def is_open(self) -> bool:
        """Return whether this stream currently owns an open backend."""
        return self._backend is not None and self._backend.isOpened()

    def open(self) -> CaptureStream:
        """Open the configured webcam and return this stream."""
        if self._backend is not None:
            return self

        backend = self._capture_factory(self.config.source)
        if not backend.isOpened():
            backend.release()
            raise CaptureOpenError(
                f"unable to open webcam device {self.config.source}"
            )

        self._backend = backend
        return self

    def read(self) -> Any:
        """Read one webcam frame or raise an explicit capture error."""
        if self._backend is None:
            raise CaptureError("capture stream must be opened before reading")

        success, frame = self._backend.read()
        if not success or frame is None:
            raise CaptureReadError(
                f"unable to read from webcam device {self.config.source}"
            )
        return frame

    def close(self) -> None:
        """Release the capture backend; repeated calls are safe."""
        if self._backend is not None:
            self._backend.release()
            self._backend = None

    def __enter__(self) -> CaptureStream:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()
