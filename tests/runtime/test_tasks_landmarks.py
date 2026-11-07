"""Tasks result adaptation, topology and lifecycle contract checks."""

from types import SimpleNamespace

import pytest

from vigil.runtime.landmarks import (
    LandmarkError,
    LandmarkOutputError,
    extract_eye_landmarks,
)
from vigil.runtime.tasks_landmarks import TasksFaceLandmarkDetector


class Backend:
    def __init__(self, faces):
        self.result = SimpleNamespace(face_landmarks=faces)
        self.closed = 0

    def detect(self, image):
        return self.result

    def close(self):
        self.closed += 1


def make_detector(tmp_path, faces):
    path = tmp_path / "model.task"
    path.touch()
    backend = Backend(faces)
    return TasksFaceLandmarkDetector(
        path, backend_factory=lambda *_: backend, image_converter=lambda x: x
    ), backend


def test_tasks_preserves_order_coordinates_and_eye_indices(tmp_path):
    points = [SimpleNamespace(x=i / 1000, y=i / 2000, z=-i / 3000) for i in range(478)]
    detector, backend = make_detector(tmp_path, [points])
    with detector:
        (face,) = detector.detect(object())
        eyes = extract_eye_landmarks(face)
        assert [p.x for p in eyes.left] == [
            i / 1000 for i in (362, 385, 387, 263, 373, 380)
        ]
        assert [p.x for p in eyes.right] == [
            i / 1000 for i in (33, 160, 158, 133, 153, 144)
        ]
        assert face.points[33].z == -33 / 3000
    detector.close()
    assert backend.closed == 1
    with pytest.raises(LandmarkError):
        detector.detect(object())


def test_tasks_no_face(tmp_path):
    detector, _ = make_detector(tmp_path, [])
    with detector:
        assert detector.detect(object()) == ()


@pytest.mark.parametrize(
    "points", [[], [object()] * 478, [SimpleNamespace(x=float("nan"), y=0, z=0)] * 478]
)
def test_tasks_rejects_malformed_output(tmp_path, points):
    detector, backend = make_detector(tmp_path, [points])
    with pytest.raises(LandmarkOutputError), detector:
        detector.detect(object())
    assert backend.closed == 1
