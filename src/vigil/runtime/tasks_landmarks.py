"""MediaPipe Tasks adapter exposing only Vigil's normalized face contract."""

from pathlib import Path
from typing import Any, Callable

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
)


class TasksFaceLandmarkDetector:
    """Synchronous IMAGE mode: one result per frame, without async frame drops.

    The official 478-point model retains the canonical 468-point face topology
    and adds iris points. Eye coordinates stay in normalized image space.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: FaceLandmarkConfig | None = None,
        *,
        backend_factory: Callable | None = None,
        image_converter: Callable | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = config or FaceLandmarkConfig()
        self._factory = backend_factory
        self._converter = image_converter
        self._backend: Any = None

    def open(self):
        """Load an explicitly supplied task asset; never download at runtime."""
        if self._backend is not None:
            return self
        if not self.model_path.is_file():
            raise LandmarkOpenError("face landmarker task asset is missing")
        try:
            if self._factory:
                self._backend = self._factory(self.model_path, self.config)
            else:
                import mediapipe as mp
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import (
                    FaceLandmarker,
                    FaceLandmarkerOptions,
                    RunningMode,
                )

                connections = mp.tasks.vision.FaceLandmarksConnections
                for indices, edges in (
                    (LEFT_EYE_LANDMARK_INDICES, connections.FACE_LANDMARKS_LEFT_EYE),
                    (RIGHT_EYE_LANDMARK_INDICES, connections.FACE_LANDMARKS_RIGHT_EYE),
                ):
                    vertices = {v for edge in edges for v in (edge.start, edge.end)}
                    if not set(indices) <= vertices:
                        raise LandmarkOpenError("unsupported eye landmark topology")
                self._backend = FaceLandmarker.create_from_options(
                    FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(self.model_path)),
                        running_mode=RunningMode.IMAGE,
                        num_faces=self.config.max_num_faces,
                        min_face_detection_confidence=self.config.min_detection_confidence,
                        min_tracking_confidence=self.config.min_tracking_confidence,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                )
        except LandmarkError:
            raise
        except Exception as exc:
            raise LandmarkOpenError(
                "unable to initialize Tasks face landmarker"
            ) from exc
        return self

    def detect(self, frame: Any) -> tuple[FaceLandmarks, ...]:
        if self._backend is None:
            raise LandmarkError("landmark detector must be opened before use")
        if frame is None:
            raise ValueError("frame must not be None")
        try:
            if self._converter:
                image = self._converter(frame)
            else:
                import cv2
                import mediapipe as mp
                import numpy as np

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                )
            result = self._backend.detect(image)
            faces = []
            for face in result.face_landmarks:
                if len(face) != 478:
                    raise ValueError("expected official 478-point model")
                faces.append(
                    FaceLandmarks(
                        tuple(
                            NormalizedLandmark(float(p.x), float(p.y), float(p.z))
                            for p in face
                        )
                    )
                )
            if len(faces) > self.config.max_num_faces:
                raise ValueError("too many faces")
            return tuple(faces)
        except Exception as exc:
            raise LandmarkOutputError("Tasks returned invalid face landmarks") from exc

    def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


def create_detector(backend: str, model_path: Path | None):
    """Select a vision adapter without exposing vendor results downstream."""
    if backend == "legacy":
        return MediaPipeFaceLandmarkDetector()
    if backend != "tasks" or model_path is None:
        raise LandmarkOpenError("Tasks backend requires --landmark-model PATH")
    return TasksFaceLandmarkDetector(model_path)
