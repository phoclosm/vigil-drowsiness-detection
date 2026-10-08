"""Deterministic tests for face and eye landmark output contracts."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from vigil.runtime.landmarks import (
    LEFT_EYE_LANDMARK_INDICES,
    RIGHT_EYE_LANDMARK_INDICES,
    FaceLandmarkConfig,
    FaceLandmarks,
    LandmarkError,
    LandmarkOpenError,
    LandmarkOutputError,
    MediaPipeFaceLandmarkDetector,
    NormalizedLandmark,
    extract_eye_landmarks,
)


@dataclass
class FakePoint:
    x: float
    y: float
    z: float


@dataclass
class FakeFace:
    landmark: list[Any]


@dataclass
class FakeResult:
    multi_face_landmarks: list[FakeFace] | None


class FakeFaceMeshBackend:
    """In-memory replacement for MediaPipe Face Mesh."""

    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.processed_images: list[Any] = []
        self.closed = False

    def process(self, image: Any) -> Any:
        self.processed_images.append(image)
        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


class RecordingBackendFactory:
    """Return one fake backend and retain its received configuration."""

    def __init__(self, backend: FakeFaceMeshBackend) -> None:
        self.backend = backend
        self.config: FaceLandmarkConfig | None = None

    def __call__(self, config: FaceLandmarkConfig) -> FakeFaceMeshBackend:
        self.config = config
        return self.backend


def make_raw_face(count: int = 468) -> FakeFace:
    """Build ordered, finite points that resemble a Face Mesh result."""
    return FakeFace(
        landmark=[
            FakePoint(x=index / 1000, y=index / 2000, z=-index / 3000)
            for index in range(count)
        ]
    )


@pytest.mark.parametrize("max_num_faces", [0, -1, True, 1.5])
def test_config_requires_positive_integer_face_count(max_num_faces: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FaceLandmarkConfig(max_num_faces=max_num_faces)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_detection_confidence", -0.1),
        ("min_detection_confidence", 1.1),
        ("min_detection_confidence", True),
        ("min_tracking_confidence", -0.1),
        ("min_tracking_confidence", 1.1),
        ("min_tracking_confidence", "0.5"),
    ],
)
def test_config_rejects_invalid_confidence(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        FaceLandmarkConfig(**{field: value})  # type: ignore[arg-type]


def test_detect_requires_open_detector() -> None:
    detector = MediaPipeFaceLandmarkDetector(
        backend_factory=RecordingBackendFactory(FakeFaceMeshBackend([])),
        color_converter=lambda frame: frame,
    )

    with pytest.raises(LandmarkError, match="opened before use"):
        detector.detect("frame")


def test_open_wraps_backend_initialization_failure() -> None:
    def failing_factory(_: FaceLandmarkConfig) -> FakeFaceMeshBackend:
        raise RuntimeError("backend failure")

    detector = MediaPipeFaceLandmarkDetector(backend_factory=failing_factory)

    with pytest.raises(LandmarkOpenError, match="unable to initialize"):
        detector.open()


def test_detector_converts_normalized_face_output() -> None:
    backend = FakeFaceMeshBackend([FakeResult([make_raw_face()])])
    factory = RecordingBackendFactory(backend)
    config = FaceLandmarkConfig(max_num_faces=1, refine_landmarks=True)
    detector = MediaPipeFaceLandmarkDetector(
        config,
        backend_factory=factory,
        color_converter=lambda frame: f"rgb:{frame}",
    ).open()

    faces = detector.detect("bgr-frame")

    assert factory.config == config
    assert backend.processed_images == ["rgb:bgr-frame"]
    assert len(faces) == 1
    assert len(faces[0].points) == 468
    assert faces[0].point(33) == NormalizedLandmark(
        x=0.033,
        y=0.0165,
        z=-0.011,
    )


@pytest.mark.parametrize("raw_faces", [None, []])
def test_missing_face_returns_empty_tuple(raw_faces: list[FakeFace] | None) -> None:
    backend = FakeFaceMeshBackend([FakeResult(raw_faces)])
    detector = MediaPipeFaceLandmarkDetector(
        backend_factory=RecordingBackendFactory(backend),
        color_converter=lambda frame: frame,
    ).open()

    assert detector.detect("frame-without-face") == ()


def test_none_frame_is_rejected_before_processing() -> None:
    backend = FakeFaceMeshBackend([])
    detector = MediaPipeFaceLandmarkDetector(
        backend_factory=RecordingBackendFactory(backend),
        color_converter=lambda frame: frame,
    ).open()

    with pytest.raises(ValueError, match="must not be None"):
        detector.detect(None)

    assert backend.processed_images == []


def test_malformed_result_is_rejected() -> None:
    backend = FakeFaceMeshBackend([SimpleNamespace(unexpected=[])])
    detector = MediaPipeFaceLandmarkDetector(
        backend_factory=RecordingBackendFactory(backend),
        color_converter=lambda frame: frame,
    ).open()

    with pytest.raises(LandmarkOutputError, match="does not contain"):
        detector.detect("frame")


@pytest.mark.parametrize(
    "raw_face",
    [
        FakeFace(landmark=[]),
        FakeFace(landmark=[SimpleNamespace(x=0.1, y=0.2)]),
        FakeFace(landmark=[FakePoint(x=float("nan"), y=0.2, z=0.0)]),
    ],
)
def test_malformed_face_landmarks_are_rejected(raw_face: FakeFace) -> None:
    backend = FakeFaceMeshBackend([FakeResult([raw_face])])
    detector = MediaPipeFaceLandmarkDetector(
        backend_factory=RecordingBackendFactory(backend),
        color_converter=lambda frame: frame,
    ).open()

    with pytest.raises(LandmarkOutputError, match="malformed"):
        detector.detect("frame")


def test_eye_extraction_uses_documented_order() -> None:
    face = FaceLandmarks(
        points=tuple(
            NormalizedLandmark(x=index / 1000, y=0.0, z=0.0)
            for index in range(468)
        )
    )

    eyes = extract_eye_landmarks(face)

    assert tuple(round(point.x * 1000) for point in eyes.left) == (
        LEFT_EYE_LANDMARK_INDICES
    )
    assert tuple(round(point.x * 1000) for point in eyes.right) == (
        RIGHT_EYE_LANDMARK_INDICES
    )


def test_eye_extraction_rejects_incomplete_face_mesh() -> None:
    face = FaceLandmarks(points=(NormalizedLandmark(0.0, 0.0, 0.0),))

    with pytest.raises(LandmarkOutputError, match="is unavailable"):
        extract_eye_landmarks(face)


def test_context_manager_closes_backend() -> None:
    backend = FakeFaceMeshBackend([FakeResult(None)])

    with MediaPipeFaceLandmarkDetector(
        backend_factory=RecordingBackendFactory(backend),
        color_converter=lambda frame: frame,
    ) as detector:
        assert detector.is_open
        assert detector.detect("frame") == ()

    assert backend.closed
    assert not detector.is_open
