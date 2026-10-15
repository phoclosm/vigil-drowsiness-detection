"""Deterministic tests for the live state and confidence overlay."""

from typing import Any

import pytest

from vigil.ml.inference import (
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_OK,
    InferenceResult,
)
from vigil.runtime.inference import LivePrediction
from vigil.runtime.overlay import (
    ACTIVE_COLOR_BGR,
    DROWSY_COLOR_BGR,
    SLEEPING_COLOR_BGR,
    UNAVAILABLE_COLOR_BGR,
    OverlayError,
    PredictionOverlay,
    PredictionOverlayConfig,
)


class FakeDrawingBackend:
    FONT_HERSHEY_SIMPLEX = 42
    LINE_AA = 99

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

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
        self.calls.append(
            (
                image,
                text,
                origin,
                font_face,
                font_scale,
                color,
                thickness,
                line_type,
            )
        )
        return image


def make_prediction(
    state: str | None,
    confidence: float | None,
) -> LivePrediction:
    has_prediction = state is not None
    class_ids = {"ACTIVE": 0, "DROWSY": 1, "SLEEPING": 2}
    probabilities = (
        {name: confidence if name == state else (1.0 - confidence) / 2.0
         for name in class_ids}
        if has_prediction and confidence is not None
        else None
    )
    result = InferenceResult(
        predicted_class=state,
        class_id=class_ids.get(state) if state is not None else None,
        probabilities=probabilities,
        confidence=confidence,
        status=STATUS_OK if has_prediction else STATUS_INSUFFICIENT_HISTORY,
        latency_ms=1.0 if has_prediction else 0.0,
        reason="predicted" if has_prediction else "buffer_warming_up",
        session_id="live-session",
        buffer_size=60 if has_prediction else 1,
        start_frame_index=0,
        end_frame_index=59 if has_prediction else 0,
        start_timestamp_ms=0.0,
        end_timestamp_ms=2360.0 if has_prediction else 0.0,
    )
    return LivePrediction(
        result=result,
        state=state,
        confidence=confidence,
        source_frame_index=59 if has_prediction else None,
        is_fresh=has_prediction,
    )


@pytest.mark.parametrize(
    ("state", "color"),
    [
        ("ACTIVE", ACTIVE_COLOR_BGR),
        ("DROWSY", DROWSY_COLOR_BGR),
        ("SLEEPING", SLEEPING_COLOR_BGR),
    ],
)
def test_overlay_draws_canonical_state_and_confidence(
    state: str,
    color: tuple[int, int, int],
) -> None:
    backend = FakeDrawingBackend()
    overlay = PredictionOverlay(drawing_backend=backend)
    image = object()

    returned = overlay.draw(image, make_prediction(state, 0.8))

    assert returned is image
    assert [call[1] for call in backend.calls] == [
        f"State: {state}",
        "Confidence: 80.0%",
    ]
    assert [call[2] for call in backend.calls] == [(20, 30), (20, 58)]
    assert [call[5] for call in backend.calls] == [color, color]


def test_overlay_marks_unavailable_prediction_without_inventing_confidence() -> None:
    backend = FakeDrawingBackend()
    overlay = PredictionOverlay(drawing_backend=backend)

    overlay.draw(object(), make_prediction(None, None))

    assert [call[1] for call in backend.calls] == [
        "State: UNAVAILABLE",
        "Confidence: --",
    ]
    assert [call[5] for call in backend.calls] == [
        UNAVAILABLE_COLOR_BGR,
        UNAVAILABLE_COLOR_BGR,
    ]


def test_overlay_uses_configured_layout_and_typography() -> None:
    backend = FakeDrawingBackend()
    overlay = PredictionOverlay(
        PredictionOverlayConfig(
            origin=(5, 10),
            line_spacing=20,
            font_scale=1.0,
            thickness=3,
        ),
        drawing_backend=backend,
    )

    overlay.draw(object(), make_prediction("ACTIVE", 0.8))

    assert [call[2] for call in backend.calls] == [(5, 10), (5, 30)]
    assert all(call[3] == backend.FONT_HERSHEY_SIMPLEX for call in backend.calls)
    assert all(call[4] == 1.0 for call in backend.calls)
    assert all(call[6] == 3 for call in backend.calls)
    assert all(call[7] == backend.LINE_AA for call in backend.calls)


def test_overlay_rejects_unknown_state() -> None:
    prediction = make_prediction("ACTIVE", 0.8)
    malformed = LivePrediction(
        result=prediction.result,
        state="TIRED",
        confidence=0.8,
        source_frame_index=59,
        is_fresh=True,
    )

    with pytest.raises(OverlayError, match="unknown fatigue state"):
        PredictionOverlay(drawing_backend=FakeDrawingBackend()).draw(
            object(),
            malformed,
        )


def test_overlay_rejects_empty_frame() -> None:
    with pytest.raises(OverlayError, match="empty frame"):
        PredictionOverlay(drawing_backend=FakeDrawingBackend()).draw(
            None,
            make_prediction(None, None),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin": (-1, 0)},
        {"line_spacing": 0},
        {"font_scale": 0.0},
        {"thickness": 0},
    ],
)
def test_overlay_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PredictionOverlayConfig(**kwargs)  # type: ignore[arg-type]
