"""End-to-end OpenCV demo for the Vigil v0.1 prototype."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Protocol, Sequence

from vigil.runtime.capture import (
    CapturedFrame,
    CaptureConfig,
    CaptureError,
    CaptureStream,
)
from vigil.runtime.eye_features import EyeFeatureStream, EyeFeatureStreamConfig
from vigil.runtime.environment import opencv_version
from vigil.runtime.inference import ModelInferenceAdapter, ModelInferenceError
from vigil.runtime.landmarks import LandmarkError, MediaPipeFaceLandmarkDetector
from vigil.runtime.live_pipeline import (
    LiveInferencePipeline,
    LivePipelineError,
    LivePipelineResult,
)
from vigil.runtime.overlay import OverlayError, PredictionOverlay
from vigil.runtime.preprocessing import VisionPreprocessingError, VisionPreprocessor


DEMO_WINDOW_NAME = "Vigil v0.1"


class DemoError(RuntimeError):
    """Base exception for live demo failures."""


class DemoDisplayError(DemoError):
    """Raised when the OpenCV display window cannot be used."""


class FrameSource(Protocol):
    """Frame source consumed by the hardware-independent demo loop."""

    def read(self) -> CapturedFrame | None:
        """Return the next frame or ``None`` when a video is exhausted."""


class LiveFrameProcessor(Protocol):
    """Per-frame preprocessing and inference boundary."""

    def process(self, frame: CapturedFrame) -> LivePipelineResult:
        """Process one captured frame."""


class FrameOverlay(Protocol):
    """Prediction overlay boundary used by the demo loop."""

    def draw(self, image: Any, prediction: object) -> Any:
        """Draw prediction information onto an image."""


class DisplayBackend(Protocol):
    """Minimal display API required by the demo loop."""

    def show(self, window_name: str, image: Any) -> None:
        """Display the current frame."""

    def read_key(self, delay_ms: int) -> int:
        """Return the latest key code or a negative value when none was pressed."""

    def close(self) -> None:
        """Close display resources."""


class OpenCVDisplayBackend:
    """Lazy OpenCV adapter for window display and keyboard events."""

    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DemoDisplayError(
                "OpenCV is required for the live demo display"
            ) from exc
        self._cv2 = cv2

    def show(self, window_name: str, image: Any) -> None:
        try:
            self._cv2.imshow(window_name, image)
        except Exception as exc:
            raise DemoDisplayError("unable to display the live demo window") from exc

    def read_key(self, delay_ms: int) -> int:
        try:
            return int(self._cv2.waitKey(delay_ms)) & 0xFF
        except Exception as exc:
            raise DemoDisplayError("unable to read live demo keyboard input") from exc

    def close(self) -> None:
        try:
            self._cv2.destroyAllWindows()
        except Exception as exc:
            raise DemoDisplayError("unable to close the live demo window") from exc


@dataclass(frozen=True, slots=True)
class DemoSummary:
    """Non-performance summary of one completed demo session."""

    frames_processed: int
    stop_reason: str


class DemoLoop:
    """Display processed frames until input ends or the user stops the demo."""

    def __init__(
        self,
        *,
        overlay: FrameOverlay,
        display: DisplayBackend,
        max_frames: int | None = None,
        window_name: str = DEMO_WINDOW_NAME,
    ) -> None:
        if (
            max_frames is not None
            and (
                isinstance(max_frames, bool)
                or not isinstance(max_frames, int)
                or max_frames < 1
            )
        ):
            raise ValueError("max_frames must be a positive integer or None")
        if not isinstance(window_name, str) or not window_name.strip():
            raise ValueError("window_name must be a non-empty string")
        self._overlay = overlay
        self._display = display
        self._max_frames = max_frames
        self._window_name = window_name

    def run(
        self,
        source: FrameSource,
        pipeline: LiveFrameProcessor,
    ) -> DemoSummary:
        """Run the UI loop without calculating or printing performance claims."""
        frames_processed = 0
        stop_reason = "source_exhausted"
        try:
            while True:
                if (
                    self._max_frames is not None
                    and frames_processed >= self._max_frames
                ):
                    stop_reason = "frame_limit"
                    break

                frame = source.read()
                if frame is None:
                    stop_reason = "source_exhausted"
                    break

                result = pipeline.process(frame)
                self._overlay.draw(result.frame.image, result.prediction)
                self._display.show(self._window_name, result.frame.image)
                frames_processed += 1

                key = self._display.read_key(1)
                if key in (ord("q"), ord("Q"), 27):
                    stop_reason = "user_requested"
                    break
        finally:
            self._display.close()

        return DemoSummary(
            frames_processed=frames_processed,
            stop_reason=stop_reason,
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _ear_threshold(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 < parsed <= 0.60:
        raise argparse.ArgumentTypeError("EAR threshold must be within (0.0, 0.60]")
    return parsed


def _non_empty(value: str) -> str:
    parsed = value.strip()
    if not parsed:
        raise argparse.ArgumentTypeError("value must not be empty")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the demo parser without opening a camera, model, or UI dependency."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Vigil v0.1 learning prototype. This is not a production "
            "driver-safety or medical system."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--webcam", type=_non_negative_int, metavar="INDEX")
    source.add_argument("--video", type=Path, metavar="PATH")
    parser.add_argument("--model", type=Path, required=True, metavar="BUNDLE")
    parser.add_argument("--session-id", type=_non_empty, required=True)
    parser.add_argument(
        "--closure-threshold-ear",
        type=_ear_threshold,
        required=True,
    )
    parser.add_argument(
        "--perclos-window-frames",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--minimum-blink-duration-ms",
        type=_non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        default=None,
        help="optional bounded frame count for a demo session",
    )
    return parser


def _capture_config(args: argparse.Namespace) -> CaptureConfig:
    if args.video is not None:
        return CaptureConfig.video_file(args.video)
    return CaptureConfig.webcam(args.webcam)


def _run_demo(args: argparse.Namespace) -> DemoSummary:
    capture_config = _capture_config(args)
    inference = ModelInferenceAdapter.from_bundle(
        args.model,
        session_id=args.session_id,
    )
    feature_stream = EyeFeatureStream(
        EyeFeatureStreamConfig(
            closure_threshold_ear=args.closure_threshold_ear,
            perclos_window_frames=args.perclos_window_frames,
            minimum_blink_duration_ms=args.minimum_blink_duration_ms,
        )
    )

    with CaptureStream(capture_config) as capture:
        with MediaPipeFaceLandmarkDetector() as detector:
            pipeline = LiveInferencePipeline(
                VisionPreprocessor(detector, feature_stream),
                inference,
            )
            loop = DemoLoop(
                overlay=PredictionOverlay(),
                display=OpenCVDisplayBackend(),
                max_frames=args.max_frames,
            )
            return loop.run(capture, pipeline)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live demo and report only session completion metadata."""
    args = build_parser().parse_args(argv)
    capture_config = _capture_config(args)
    try:
        summary = _run_demo(args)
    except (
        CaptureError,
        DemoError,
        LandmarkError,
        LivePipelineError,
        ModelInferenceError,
        OverlayError,
        VisionPreprocessingError,
        ValueError,
    ) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Demo interrupted by user.", file=sys.stderr)
        return 130

    print(f"Source: {capture_config.source_label}")
    print(f"OpenCV version: {opencv_version()}")
    print(f"Frames processed: {summary.frames_processed}")
    print(f"Stop reason: {summary.stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
