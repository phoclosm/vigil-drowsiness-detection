"""Tests for the bounded vision benchmark command-line interface."""

from pathlib import Path

import pytest

import vigil.benchmark as benchmark
from vigil.runtime.performance import (
    BenchmarkResult,
    FrameLatency,
    StageLatency,
    StageLatencySummary,
)
from vigil.runtime.preprocessing import VisionPreprocessingError


def test_parser_requires_exactly_one_capture_source() -> None:
    parser = benchmark.build_parser()
    common = [
        "--frames",
        "10",
        "--closure-threshold-ear",
        "0.20",
        "--perclos-window-frames",
        "30",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args(["--webcam", "0", "--video", "drive.mp4", *common])


def test_parser_preserves_explicit_benchmark_configuration() -> None:
    args = benchmark.build_parser().parse_args(
        [
            "--video",
            "drive.mp4",
            "--frames",
            "120",
            "--closure-threshold-ear",
            "0.21",
            "--perclos-window-frames",
            "60",
            "--minimum-blink-duration-ms",
            "75",
        ]
    )

    assert args.video == Path("drive.mp4")
    assert args.webcam is None
    assert args.frames == 120
    assert args.closure_threshold_ear == 0.21
    assert args.perclos_window_frames == 60
    assert args.minimum_blink_duration_ms == 75.0


@pytest.mark.parametrize(
    "option,value",
    [
        ("--frames", "0"),
        ("--webcam", "-1"),
        ("--closure-threshold-ear", "0"),
        ("--closure-threshold-ear", "0.61"),
        ("--perclos-window-frames", "0"),
        ("--minimum-blink-duration-ms", "-1"),
        ("--minimum-blink-duration-ms", "nan"),
        ("--minimum-blink-duration-ms", "inf"),
    ],
)
def test_parser_rejects_invalid_numeric_configuration(
    option: str,
    value: str,
) -> None:
    values = {
        "--webcam": "0",
        "--frames": "10",
        "--closure-threshold-ear": "0.20",
        "--perclos-window-frames": "30",
        "--minimum-blink-duration-ms": "0",
    }
    values[option] = value
    argv: list[str] = []
    for name, configured in values.items():
        argv.extend((name, configured))

    with pytest.raises(SystemExit):
        benchmark.build_parser().parse_args(argv)


def test_main_prints_only_returned_measurements(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frame = FrameLatency(
        frame_index=0,
        stages=(StageLatency("capture", 2.0),),
    )
    result = BenchmarkResult(
        frame_latencies=(frame,),
        elapsed_seconds=0.05,
        frames_per_second=20.0,
        stage_summaries=(
            StageLatencySummary("capture", 1, 2.0, 2.0, 2.0),
        ),
    )
    monkeypatch.setattr(benchmark, "_run_benchmark", lambda _: result)

    exit_code = benchmark.main(
        [
            "--webcam",
            "0",
            "--frames",
            "1",
            "--closure-threshold-ear",
            "0.20",
            "--perclos-window-frames",
            "30",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Source: webcam device 0" in captured.out
    assert "Requested frame limit: 1" in captured.out
    assert "Measured FPS: 20.000" in captured.out


def test_main_reports_runtime_failure_without_fake_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_: object) -> BenchmarkResult:
        raise VisionPreprocessingError("bad frame")

    monkeypatch.setattr(benchmark, "_run_benchmark", fail)

    exit_code = benchmark.main(
        [
            "--webcam",
            "0",
            "--frames",
            "1",
            "--closure-threshold-ear",
            "0.20",
            "--perclos-window-frames",
            "30",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Benchmark failed: bad frame" in captured.err
    assert "Measured FPS" not in captured.err
