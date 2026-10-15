"""OpenCV text overlay for live fatigue state and model confidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

from vigil.runtime.inference import LivePrediction


ACTIVE_COLOR_BGR = (0, 200, 0)
DROWSY_COLOR_BGR = (0, 165, 255)
SLEEPING_COLOR_BGR = (0, 0, 255)
UNAVAILABLE_COLOR_BGR = (180, 180, 180)

_STATE_COLORS = {
    "ACTIVE": ACTIVE_COLOR_BGR,
    "DROWSY": DROWSY_COLOR_BGR,
    "SLEEPING": SLEEPING_COLOR_BGR,
}


class OverlayError(RuntimeError):
    """Raised when prediction data cannot be rendered safely."""


class OverlayDependencyError(OverlayError):
    """Raised when the default OpenCV drawing backend is unavailable."""


class TextDrawingBackend(Protocol):
    """Small subset of the OpenCV API needed by the overlay."""

    FONT_HERSHEY_SIMPLEX: int
    LINE_AA: int

    def putText(  # noqa: N802 - mirrors the OpenCV method name
        self,
        image: Any,
        text: str,
        origin: tuple[int, int],
        font_face: int,
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int,
    ) -> Any:
        """Draw text on an image in place."""


@dataclass(frozen=True, slots=True)
class PredictionOverlayConfig:
    """Layout and typography for the two-line prediction overlay."""

    origin: tuple[int, int] = (20, 30)
    line_spacing: int = 28
    font_scale: float = 0.7
    thickness: int = 2

    def __post_init__(self) -> None:
        if (
            not isinstance(self.origin, tuple)
            or len(self.origin) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.origin
            )
        ):
            raise ValueError("origin must contain two non-negative integers")
        if (
            isinstance(self.line_spacing, bool)
            or not isinstance(self.line_spacing, int)
            or self.line_spacing < 1
        ):
            raise ValueError("line_spacing must be a positive integer")
        if (
            isinstance(self.font_scale, bool)
            or not isinstance(self.font_scale, (int, float))
            or not isfinite(self.font_scale)
            or self.font_scale <= 0.0
        ):
            raise ValueError("font_scale must be finite and positive")
        if (
            isinstance(self.thickness, bool)
            or not isinstance(self.thickness, int)
            or self.thickness < 1
        ):
            raise ValueError("thickness must be a positive integer")


class PredictionOverlay:
    """Draw the latest genuine model state and confidence onto a frame."""

    def __init__(
        self,
        config: PredictionOverlayConfig | None = None,
        *,
        drawing_backend: TextDrawingBackend | None = None,
    ) -> None:
        self.config = config or PredictionOverlayConfig()
        self._drawing_backend = drawing_backend

    def draw(self, image: Any, prediction: LivePrediction) -> Any:
        """Draw state and confidence in place, returning the same image object."""
        if image is None:
            raise OverlayError("cannot draw a prediction on an empty frame")
        if not isinstance(prediction, LivePrediction):
            raise TypeError("prediction must be a LivePrediction")

        if prediction.state is None:
            if prediction.confidence is not None:
                raise OverlayError("confidence requires an available fatigue state")
            state_text = "State: UNAVAILABLE"
            confidence_text = "Confidence: --"
            color = UNAVAILABLE_COLOR_BGR
        else:
            if prediction.state not in _STATE_COLORS:
                raise OverlayError(
                    f"cannot render unknown fatigue state: {prediction.state!r}"
                )
            confidence = prediction.confidence
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise OverlayError(
                    "available fatigue state requires confidence within [0.0, 1.0]"
                )
            state_text = f"State: {prediction.state}"
            confidence_text = f"Confidence: {confidence:.1%}"
            color = _STATE_COLORS[prediction.state]

        backend = self._drawing_backend or _load_opencv()
        x, y = self.config.origin
        self._draw_text(backend, image, state_text, (x, y), color)
        self._draw_text(
            backend,
            image,
            confidence_text,
            (x, y + self.config.line_spacing),
            color,
        )
        return image

    def _draw_text(
        self,
        backend: TextDrawingBackend,
        image: Any,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        backend.putText(
            image,
            text,
            origin,
            backend.FONT_HERSHEY_SIMPLEX,
            float(self.config.font_scale),
            color,
            self.config.thickness,
            backend.LINE_AA,
        )


def _load_opencv() -> TextDrawingBackend:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency availability varies
        raise OverlayDependencyError(
            "OpenCV is required for the default prediction overlay backend"
        ) from exc
    return cv2  # type: ignore[return-value]
