# Vigil

Vigil is a learning-focused real-time driver-drowsiness detection prototype.
It connects webcam or video capture, face and eye landmarks, eye/blink
features, temporal model inference, and an OpenCV state overlay.

Vigil v0.1 is not a production safety system, a medical device, or a
substitute for attentive driving or certified vehicle safety equipment.

## Current capabilities

- OpenCV webcam and local-video capture with monotonic frame timing
- MediaPipe face and eye landmark detection
- Eye Aspect Ratio, blink, and trailing PERCLOS features
- A classical rule-based baseline for offline comparison
- PyTorch temporal dataset, GRU classifier, training, evaluation, and export
- Bundle-backed real-time temporal inference
- Live `ACTIVE`, `DROWSY`, or `SLEEPING` overlay with model confidence
- Deterministic tests separated from hardware-dependent demo and benchmarks

The repository does not include a trained model, dataset, recording, or
published accuracy/FPS result.

## Requirements

- Python 3.10 or newer
- A supported webcam or a local video file
- An exported Vigil `.pt` model bundle for live inference
- A desktop environment capable of displaying an OpenCV window

The model bundle must include its fitted normalizer, temporal-window
configuration, canonical class mapping, and preprocessing metadata. Keep
checkpoints outside Git unless maintainers explicitly approve one.

## Installation

From PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[desktop,dev,ml]"
```

If `py` is unavailable, use a working `python` or `python3` command instead.

## Run the live demo

Use preprocessing values that match the supplied model bundle's training
configuration. Do not choose values merely to obtain a preferred state.

Webcam:

```powershell
python -m vigil `
  --webcam 0 `
  --model <path-to-exported-bundle.pt> `
  --session-id <unique-session-id> `
  --closure-threshold-ear <approved-threshold> `
  --perclos-window-frames <approved-window-size> `
  --minimum-blink-duration-ms <approved-duration>
```

Local video:

```powershell
python -m vigil `
  --video <path-to-local-video> `
  --model <path-to-exported-bundle.pt> `
  --session-id <unique-session-id> `
  --closure-threshold-ear <approved-threshold> `
  --perclos-window-frames <approved-window-size> `
  --minimum-blink-duration-ms <approved-duration>
```

Use `--max-frames <count>` for a bounded session. Press `q`, `Q`, or Escape to
close the live window. The overlay reports `UNAVAILABLE` while the 60-frame
model window warms up or after a reset. Missing landmarks are retained as
missing observations instead of being converted to false eye closures.

The v0.1 alert is visual only: `ACTIVE` is green, `DROWSY` is orange, and
`SLEEPING` is red. No audio, notification service, or vehicle integration is
included.

## Run a measured vision benchmark

Benchmarking is a separate bounded command and does not require a model:

```powershell
python -m vigil.benchmark `
  --webcam 0 `
  --frames <positive-frame-count> `
  --closure-threshold-ear <approved-threshold> `
  --perclos-window-frames <approved-window-size>
```

See [the benchmark protocol](docs/m1-vision-benchmark.md) before recording or
publishing results. Test fixture timings are not hardware benchmarks.

## Development checks

```powershell
python -m pytest
python -m ruff check .
```

The default tests use deterministic fixtures and do not require a camera.

## Documentation

- [v0.1.0 changelog](CHANGELOG.md)
- [System architecture](docs/architecture.md)
- [Demo evidence and performance record](docs/demo-and-performance.md)
- [Release checklist](docs/release-checklist.md)
- [Vision benchmark protocol](docs/m1-vision-benchmark.md)
- [Data contract](docs/m2-data-contract.md)
- [Temporal windowing](docs/m2-temporal-windowing.md)
- [Temporal model](docs/m2-temporal-classifier.md)
- [Runtime inference and evaluation](docs/m2-runtime-and-evaluation.md)

## Data and artifact policy

Do not commit secrets, `.env` files, personally identifying recordings,
generated datasets, model checkpoints, or unreviewed benchmark dumps. Use only
data and video for which consent and licensing are documented.
