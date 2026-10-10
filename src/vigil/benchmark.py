"""Command-line runner for measured Vigil vision benchmarks."""

from __future__ import annotations

import argparse
import sys
from math import isfinite
from pathlib import Path
from typing import Sequence

from vigil.runtime.capture import CaptureConfig, CaptureError, CaptureStream
from vigil.runtime.eye_features import EyeFeatureStream, EyeFeatureStreamConfig
from vigil.runtime.landmarks import LandmarkError, MediaPipeFaceLandmarkDetector
from vigil.runtime.performance import (
    BenchmarkResult,
    BenchmarkRunner,
    PerformanceMeasurementError,
    format_benchmark_result,
)
from vigil.runtime.preprocessing import VisionPreprocessingError, VisionPreprocessor


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
    if not 0.0 < parsed <= 0.60:
        raise argparse.ArgumentTypeError("EAR threshold must be within (0.0, 0.60]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser without opening runtime dependencies."""
    parser = argparse.ArgumentParser(
        description="Measure Vigil capture and vision-preprocessing performance."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--webcam", type=_non_negative_int, metavar="INDEX")
    source.add_argument("--video", type=Path, metavar="PATH")
    parser.add_argument("--frames", type=_positive_int, required=True)
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
    return parser


def _capture_config(args: argparse.Namespace) -> CaptureConfig:
    if args.video is not None:
        return CaptureConfig.video_file(args.video)
    return CaptureConfig.webcam(args.webcam)


def _run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    capture_config = _capture_config(args)
    feature_stream = EyeFeatureStream(
        EyeFeatureStreamConfig(
            closure_threshold_ear=args.closure_threshold_ear,
            perclos_window_frames=args.perclos_window_frames,
            minimum_blink_duration_ms=args.minimum_blink_duration_ms,
        )
    )
    with CaptureStream(capture_config) as capture:
        with MediaPipeFaceLandmarkDetector() as detector:
            preprocessor = VisionPreprocessor(detector, feature_stream)
            return BenchmarkRunner().run(
                read_frame=capture.read,
                process_frame=lambda frame: preprocessor.process(frame).latency,
                max_frames=args.frames,
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a bounded benchmark and print only measurements from that run."""
    args = build_parser().parse_args(argv)
    config = _capture_config(args)
    try:
        result = _run_benchmark(args)
    except (
        CaptureError,
        LandmarkError,
        PerformanceMeasurementError,
        VisionPreprocessingError,
        ValueError,
    ) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1

    print(f"Source: {config.source_label}")
    print(f"Requested frame limit: {args.frames}")
    print(format_benchmark_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
