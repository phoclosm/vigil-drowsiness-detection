"""Face landmark detection interfaces for Vigil's runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Protocol


class LandmarkError(RuntimeError):
    """Base exception for face landmark failures."""


class LandmarkDependencyError(LandmarkError):
    """Raised when OpenCV or MediaPipe is unavailable."""


class LandmarkOpenError(LandmarkError):
    """Raised when the face landmark backend cannot be created."""


class LandmarkOutputError(LandmarkError):
    """Raised when a detector returns malformed landmark output."""


@dataclass(frozen=True, slots=True)
class NormalizedLandmark:
    """One finite 3D point in MediaPipe's normalized image coordinates."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("landmark coordinates must be finite")


@dataclass(frozen=True, slots=True)
class FaceLandmarks:
    """Ordered normalized landmarks for one detected face."""

    points: tuple[NormalizedLandmark, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("a detected face must contain at least one landmark")

    def point(self, index: int) -> NormalizedLandmark:
        """Return a landmark by model index with a domain-specific error."""
        if index < 0 or index >= len(self.points):
            raise LandmarkOutputError(
                f"landmark index {index} is unavailable; detector returned "
                f"{len(self.points)} points"
            )
        return self.points[index]


@dataclass(frozen=True, slots=True)
class FaceLandmarkConfig:
    """MediaPipe Face Mesh settings for real-time frame processing."""

    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_num_faces, bool)
            or not isinstance(self.max_num_faces, int)
            or self.max_num_faces < 1
        ):
            raise ValueError("max_num_faces must be a positive integer")
        for name, value in (
            ("min_detection_confidence", self.min_detection_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between 0.0 and 1.0")


class FaceMeshBackend(Protocol):
    """Subset of the MediaPipe Face Mesh object used by Vigil."""

    def process(self, image: Any) -> Any:
        """Detect face landmarks in one RGB frame."""

    def close(self) -> None:
        """Release backend resources."""


class FaceLandmarkDetector(Protocol):
    """Stable detector contract consumed by the Vigil runtime."""

    def detect(self, frame: Any) -> tuple[FaceLandmarks, ...]:
        """Return all faces detected in one BGR frame."""

    def close(self) -> None:
        """Release detector resources."""


BackendFactory = Callable[[FaceLandmarkConfig], FaceMeshBackend]
ColorConverter = Callable[[Any], Any]


def _create_mediapipe_backend(config: FaceLandmarkConfig) -> FaceMeshBackend:
    """Create MediaPipe Face Mesh without importing it at module load."""
    try:
        import mediapipe as mp
    except ImportError as error:
        raise LandmarkDependencyError(
            "MediaPipe is required for face landmark detection"
        ) from error

    try:
        face_mesh = mp.solutions.face_mesh
    except AttributeError as error:
        raise LandmarkDependencyError(
            "the installed MediaPipe package does not provide Face Mesh"
        ) from error

    return face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=config.max_num_faces,
        refine_landmarks=config.refine_landmarks,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )


def _bgr_to_rgb(frame: Any) -> Any:
    """Convert an OpenCV BGR frame to the RGB format MediaPipe expects."""
    try:
        import cv2
    except ImportError as error:
        raise LandmarkDependencyError(
            "OpenCV is required to prepare frames for landmark detection"
        ) from error

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class MediaPipeFaceLandmarkDetector:
    """Lifecycle-managed adapter around MediaPipe Face Mesh."""

    def __init__(
        self,
        config: FaceLandmarkConfig | None = None,
        *,
        backend_factory: BackendFactory | None = None,
        color_converter: ColorConverter | None = None,
    ) -> None:
        self.config = config or FaceLandmarkConfig()
        self._backend_factory = backend_factory or _create_mediapipe_backend
        self._color_converter = color_converter or _bgr_to_rgb
        self._backend: FaceMeshBackend | None = None

    @property
    def is_open(self) -> bool:
        """Return whether the detector currently owns a backend."""
        return self._backend is not None

    def open(self) -> MediaPipeFaceLandmarkDetector:
        """Create the configured detector backend."""
        if self._backend is not None:
            return self

        try:
            self._backend = self._backend_factory(self.config)
        except LandmarkError:
            raise
        except Exception as error:
            raise LandmarkOpenError(
                "unable to initialize the face landmark detector"
            ) from error
        return self

    def detect(self, frame: Any) -> tuple[FaceLandmarks, ...]:
        """Detect and normalize all face landmarks in one BGR frame."""
        if self._backend is None:
            raise LandmarkError("landmark detector must be opened before use")
        if frame is None:
            raise ValueError("frame must not be None")

        result = self._backend.process(self._color_converter(frame))
        raw_faces = result.multi_face_landmarks
        return tuple(self._convert_face(face) for face in raw_faces)

    @staticmethod
    def _convert_face(raw_face: Any) -> FaceLandmarks:
        try:
            raw_points = raw_face.landmark
            points = tuple(
                NormalizedLandmark(
                    x=float(point.x),
                    y=float(point.y),
                    z=float(point.z),
                )
                for point in raw_points
            )
            return FaceLandmarks(points=points)
        except (AttributeError, TypeError, ValueError) as error:
            raise LandmarkOutputError(
                "detector returned malformed face landmarks"
            ) from error

    def close(self) -> None:
        """Release the detector backend; repeated calls are safe."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __enter__(self) -> MediaPipeFaceLandmarkDetector:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()
