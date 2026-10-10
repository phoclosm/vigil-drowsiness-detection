# M1 Vision Benchmark Instructions

## Purpose and scope

This benchmark measures the current Vigil capture and vision-preprocessing
path. It is a development diagnostic for the Vigil v0.1 prototype, not a
safety certification or evidence that the system is suitable for driving.

The repository does not publish baseline FPS or latency values yet. Add
results only after running the benchmark on identified hardware and retaining
the complete run configuration. Never copy the deterministic unit-test values
into a performance report; those values test calculations and are not
real-world measurements.

## Measured runtime boundary

The benchmark runner measures a bounded sequence of frames through these
stages:

1. `capture`: time spent obtaining each successfully returned frame from
   OpenCV.
2. `face_landmarks`: BGR-to-RGB conversion and MediaPipe face-landmark
   detection.
3. `eye_features`: canonical eye-point extraction, EAR calculation, eye
   closure state, blink tracking, and trailing frame-based PERCLOS.

Overall elapsed time starts before the first capture attempt and ends after
the final requested frame or video end. Measured FPS is:

`successfully processed frames / overall elapsed seconds`

Stage output reports the measured sample count and minimum, arithmetic mean,
and maximum latency in milliseconds. Stage sums exclude uninstrumented runner
overhead, so they must not be presented as identical to overall elapsed time.
If no frames are processed or elapsed time is zero, FPS is reported as
`unavailable` rather than estimated.

## Setup

Use Python 3.10 or newer in an isolated environment and install the project
with its runtime dependencies. Include development dependencies when running
the checks shown later.

```powershell
python -m pip install -e ".[dev]"
python -m vigil.benchmark --help
```

Do not add a webcam recording, source video, generated dataset, benchmark
dump, or local environment file to Git. A recorded input must have appropriate
consent and licensing.

## Run a bounded benchmark

The closure threshold and PERCLOS window are required arguments because they
are part of the feature contract. Use values approved for the dataset or
experiment; do not choose values merely to improve benchmark output.

For a webcam:

```powershell
python -m vigil.benchmark `
  --webcam 0 `
  --frames <positive-frame-count> `
  --closure-threshold-ear <approved-threshold> `
  --perclos-window-frames <approved-window-size> `
  --minimum-blink-duration-ms <approved-duration>
```

For a recorded video:

```powershell
python -m vigil.benchmark `
  --video <path-to-local-video> `
  --frames <positive-frame-count> `
  --closure-threshold-ear <approved-threshold> `
  --perclos-window-frames <approved-window-size> `
  --minimum-blink-duration-ms <approved-duration>
```

`--minimum-blink-duration-ms` is optional and defaults to `0.0`. That default
means every completed closed-to-open event is eligible for counting; it is a
configuration behavior, not a calibrated blink-duration claim. A video that
ends before the requested limit reports only the frames actually processed.

The command exits with a failure message instead of printing a partial
performance summary when capture, landmark detection, preprocessing, or the
measurement clock fails.

## Record enough context to reproduce a result

Before publishing any measured value, retain all of the following with the
result:

- Git revision from `git rev-parse HEAD` and branch name.
- Date and the exact benchmark command.
- Input kind, webcam model or anonymized video identifier, resolution, and
  requested and processed frame counts.
- CPU, GPU if used by the installed stack, RAM, operating system, Python
  version, and relevant OpenCV and MediaPipe versions.
- Closure threshold, PERCLOS window size, minimum blink duration, and
  MediaPipe detector configuration.
- Whether other material workloads were running and whether the run was a
  cold or warmed run.
- Complete stdout and any failure output kept outside version control unless
  a maintainer explicitly approves a small report for inclusion.

Choose the protocol before running it. Report failed runs and all comparable
runs rather than selecting only favorable output. Do not compare machines,
inputs, resolutions, configurations, or code revisions as though they were
the same experimental condition.

## Vision preprocessing contract

`VisionPreprocessor` is the reusable M1 boundary between capture and the
feature stream consumed by M2-owned recording, baseline, and model code. For
each frame it returns `VisionPreprocessingResult` with:

- `eye_features`: `EyeFeatureFrame` containing `frame_index`, session-relative
  `timestamp_ms`, `face_detected`, `landmarks_valid`, nullable `ear_left`,
  `ear_right`, and `ear_avg`, nullable `is_eye_closed`, cumulative
  `blink_count`, measured `blink_duration_ms`, trailing nullable `perclos`, and
  the runtime-only `blink_completed` event.
- `latency`: `FrameLatency` containing the measured `face_landmarks` and
  `eye_features` stages for runtime diagnostics.

Frame zero is timestamped `0.0` ms relative to the capture session. Frame
indices and timestamps must then increase strictly, although dropped-frame
index gaps are allowed. PERCLOS uses only the trailing configured window and
remains `None` during warm-up or while that window contains missing
observations.

When no face is detected, landmark-derived values remain `None`. When a face
is detected but the required eye points are incomplete or geometrically
invalid, `face_detected` remains true while `landmarks_valid` is false and the
same landmark-derived values remain `None`. Missing observations never become
zero EAR values or synthetic blinks. More than one detected face is rejected
to avoid silently changing driver identity.

The latency record is not a model feature and must not be added to M2 model
inputs without an explicitly reviewed contract change.

## Deterministic validation

The default tests do not require a camera and do not claim hardware
performance:

```powershell
python -m pytest tests/runtime/test_performance.py `
  tests/runtime/test_preprocessing.py `
  tests/runtime/test_benchmark_cli.py
python -m pytest
python -m ruff check .
```

Use a real webcam or an approved local video only for the separate manual
benchmark. Record the resulting measurements exactly; do not infer them from
unit tests or from another machine.
