"""Tests for the hardware-independent live demo loop and CLI contract."""

from pathlib import Path
from typing import Any

import pytest

import vigil.demo as demo
from vigil.ml.inference import STATUS_INSUFFICIENT_HISTORY, InferenceResult
from vigil.runtime.capture import CapturedFrame, FrameTiming
from vigil.runtime.eye_features import EyeFeatureFrame
from vigil.runtime.inference import LivePrediction, ModelLoadError
from vigil.runtime.live_pipeline import LivePipelineResult
from vigil.runtime.performance import FrameLatency


class FakeSource:
    def __init__(self, frames: list[CapturedFrame | None]) -> None:
        self.frames = list(frames)
        self.read_count = 0

    def read(self) -> CapturedFrame | None:
        self.read_count += 1
        return self.frames.pop(0)


class FakePipeline:
    def __init__(self, results: list[LivePipelineResult]) -> None:
        self.results = list(results)
        self.frames: list[CapturedFrame] = []

    def process(self, frame: CapturedFrame) -> LivePipelineResult:
        self.frames.append(frame)
        return self.results.pop(0)


class FailingPipeline:
    def process(self, frame: CapturedFrame) -> LivePipelineResult:
        raise RuntimeError(f"synthetic failure at frame {frame.index}")


class FakeOverlay:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, object]] = []

    def draw(self, image: Any, prediction: object) -> Any:
        self.calls.append((image, prediction))
        return image


class FakeDisplay:
    def __init__(self, keys: list[int]) -> None:
        self.keys = list(keys)
        self.shown: list[tuple[str, Any]] = []
        self.delays: list[int] = []
        self.closed = False

    def show(self, window_name: str, image: Any) -> None:
        self.shown.append((window_name, image))

    def read_key(self, delay_ms: int) -> int:
        self.delays.append(delay_ms)
        return self.keys.pop(0)

    def close(self) -> None:
        self.closed = True


def make_frame(index: int) -> CapturedFrame:
    return CapturedFrame(
        image=f"image-{index}",
        index=index,
        timing=FrameTiming(
            timestamp_seconds=float(index),
            delta_seconds=None if index == 0 else 1.0,
            frames_per_second=None if index == 0 else 1.0,
        ),
    )


def make_result(index: int) -> LivePipelineResult:
    frame = make_frame(index)
    features = EyeFeatureFrame(
        frame_index=index,
        timestamp_ms=float(index * 40),
        face_detected=False,
        landmarks_valid=False,
        ear_left=None,
        ear_right=None,
        ear_avg=None,
        is_eye_closed=None,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=None,
        blink_completed=False,
    )
    inference = InferenceResult(
        predicted_class=None,
        class_id=None,
        probabilities=None,
        confidence=None,
        status=STATUS_INSUFFICIENT_HISTORY,
        latency_ms=0.0,
        reason="buffer_warming_up",
        session_id="demo-session",
        buffer_size=index + 1,
        start_frame_index=0,
        end_frame_index=index,
        start_timestamp_ms=0.0,
        end_timestamp_ms=float(index * 40),
    )
    return LivePipelineResult(
        frame=frame,
        eye_features=features,
        prediction=LivePrediction(
            result=inference,
            state=None,
            confidence=None,
            source_frame_index=None,
            is_fresh=False,
        ),
        latency=FrameLatency(frame_index=index, stages=()),
    )


def common_args() -> list[str]:
    return [
        "--model",
        "model.pt",
        "--session-id",
        "demo-session",
        "--closure-threshold-ear",
        "0.20",
        "--perclos-window-frames",
        "60",
    ]


def test_parser_requires_exactly_one_capture_source() -> None:
    parser = demo.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(common_args())
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--webcam", "0", "--video", "drive.mp4", *common_args()]
        )


def test_parser_preserves_explicit_demo_configuration() -> None:
    args = demo.build_parser().parse_args(
        [
            "--video",
            "drive.mp4",
            *common_args(),
            "--minimum-blink-duration-ms",
            "75",
            "--max-frames",
            "120",
        ]
    )

    assert args.video == Path("drive.mp4")
    assert args.webcam is None
    assert args.model == Path("model.pt")
    assert args.session_id == "demo-session"
    assert args.closure_threshold_ear == pytest.approx(0.20)
    assert args.perclos_window_frames == 60
    assert args.minimum_blink_duration_ms == pytest.approx(75.0)
    assert args.max_frames == 120


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--webcam", "-1"),
        ("--session-id", "   "),
        ("--closure-threshold-ear", "0"),
        ("--closure-threshold-ear", "0.61"),
        ("--closure-threshold-ear", "nan"),
        ("--perclos-window-frames", "0"),
        ("--minimum-blink-duration-ms", "-1"),
        ("--max-frames", "0"),
    ],
)
def test_parser_rejects_invalid_configuration(option: str, value: str) -> None:
    values = {
        "--webcam": "0",
        "--model": "model.pt",
        "--session-id": "demo-session",
        "--closure-threshold-ear": "0.20",
        "--perclos-window-frames": "60",
        "--minimum-blink-duration-ms": "0",
        "--max-frames": "10",
    }
    values[option] = value
    argv: list[str] = []
    for name, configured in values.items():
        argv.extend((name, configured))

    with pytest.raises(SystemExit):
        demo.build_parser().parse_args(argv)


def test_demo_loop_processes_video_until_source_is_exhausted() -> None:
    source = FakeSource([make_frame(0), make_frame(1), None])
    pipeline = FakePipeline([make_result(0), make_result(1)])
    overlay = FakeOverlay()
    display = FakeDisplay(keys=[-1, -1])

    summary = demo.DemoLoop(overlay=overlay, display=display).run(
        source,
        pipeline,
    )

    assert summary.frames_processed == 2
    assert summary.stop_reason == "source_exhausted"
    assert [frame.index for frame in pipeline.frames] == [0, 1]
    assert [call[0] for call in overlay.calls] == ["image-0", "image-1"]
    assert [shown[0] for shown in display.shown] == [
        demo.DEMO_WINDOW_NAME,
        demo.DEMO_WINDOW_NAME,
    ]
    assert display.closed


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_demo_loop_stops_on_documented_key(key: int) -> None:
    source = FakeSource([make_frame(0), make_frame(1)])
    display = FakeDisplay(keys=[key])

    summary = demo.DemoLoop(
        overlay=FakeOverlay(),
        display=display,
    ).run(source, FakePipeline([make_result(0)]))

    assert summary.frames_processed == 1
    assert summary.stop_reason == "user_requested"
    assert source.read_count == 1
    assert display.closed


def test_demo_loop_honors_frame_limit_without_extra_capture() -> None:
    source = FakeSource([make_frame(0), make_frame(1)])
    display = FakeDisplay(keys=[-1])

    summary = demo.DemoLoop(
        overlay=FakeOverlay(),
        display=display,
        max_frames=1,
    ).run(source, FakePipeline([make_result(0)]))

    assert summary.frames_processed == 1
    assert summary.stop_reason == "frame_limit"
    assert source.read_count == 1
    assert display.closed


def test_demo_loop_closes_display_when_processing_fails() -> None:
    display = FakeDisplay(keys=[])

    with pytest.raises(RuntimeError, match="synthetic failure"):
        demo.DemoLoop(
            overlay=FakeOverlay(),
            display=display,
        ).run(FakeSource([make_frame(0)]), FailingPipeline())

    assert display.closed


def test_main_reports_completion_without_performance_claims(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        demo,
        "_run_demo",
        lambda _: demo.DemoSummary(12, "frame_limit"),
    )

    exit_code = demo.main(["--webcam", "0", *common_args()])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Source: webcam device 0" in captured.out
    assert "OpenCV version:" in captured.out
    assert "Frames processed: 12" in captured.out
    assert "Stop reason: frame_limit" in captured.out
    assert "FPS" not in captured.out
    assert "accuracy" not in captured.out.lower()


def test_main_reports_model_failure_without_opening_hardware(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_: object) -> demo.DemoSummary:
        raise ModelLoadError("bundle missing")

    monkeypatch.setattr(demo, "_run_demo", fail)

    exit_code = demo.main(["--webcam", "0", *common_args()])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Demo failed: bundle missing" in captured.err
