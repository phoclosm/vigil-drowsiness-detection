# OpenCV environments

The desktop path now uses OpenCV 5 and MediaPipe Tasks. The legacy comparison
path requires a separate virtual environment.
Never install multiple OpenCV wheel variants into one environment: they share
the `cv2` namespace. Core/ML installation does not require OpenCV or MediaPipe.

Supported desktop setup (Python 3.10 or 3.12):

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[desktop,ml,dev]"
.venv/Scripts/python -m pip check
.venv/Scripts/python -m pytest
```

The desktop extra pins MediaPipe 0.10.35 and opencv-contrib-python 5.0.0.93
with NumPy 2+. Use `--landmark-model PATH` with the separately downloaded
478-point task asset; see [modern landmarks](modern-landmarks.md). Feature
order, eye indices, normalized geometry and feature math are unchanged.
The `legacy` extra preserves MediaPipe 0.10.21, NumPy below 2 and OpenCV contrib
4.11.0.86 for comparison with `--landmark-backend legacy`.

On Ubuntu, Tasks' native library also needs GLES/EGL even with CPU inference:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libegl1 libgles2 libglib2.0-0 libportaudio2
```

The first expanded CI run caught missing `libGLESv2.so.2` during real backend
initialization after the deterministic suite passed. The dependency is now
explicit in CI; the camera-free cloud container does not install it.

The `opencv5` extra is now an alias for the modern desktop dependency set:

```powershell
python -m venv .venv-opencv5
.venv-opencv5/Scripts/python -m pip install -e ".[opencv5,ml,dev]"
.venv-opencv5/Scripts/python -m pip check
.venv-opencv5/Scripts/python -m pytest
```

Real Tasks initialization, blank-frame processing and a 120-frame webcam
benchmark succeeded locally on Python 3.12.0 with NumPy 2.5.4. The task asset
digest is documented in the migration guide. The legacy dependency conflict
below is historical; the Tasks adapter removes that blocker. No dependency
resolution checks were bypassed.

## Migration blocker observed on October 23 2026

A fresh Python 3.12 environment rejected `opencv-python==5.0.0.93` together
with `mediapipe==0.10.21` with `ResolutionImpossible`: OpenCV requires NumPy
at least 2, whereas this MediaPipe version requires NumPy below 2.
Newer MediaPipe versions without `solutions.face_mesh` cannot serve as a
drop-in replacement. Installing with `--no-deps` would hide these constraints
and is not a supported workaround.

The historical conflict is resolved by the Tasks adapter, rather than by changing
the feature schema. CI now checks modern desktop, legacy and vision-free cloud
environments on Python 3.10/3.12, plus a container smoke job. Real camera/display
checks remain manual. Numerical parity and model-quality evidence on recorded
faces are separate from the unchanged topology/feature contracts.

## Local verification on October 23

Windows, Python 3.12.0, separate freshly created virtual environments:

| Environment | Imported cv2 | NumPy | Full suite |
| --- | --- | --- | --- |
| desktop | 4.11.0 | 1.26.4 | 393 passed |
| opencv5 | 5.0.0 (wheel 5.0.0.93) | 2.5.4 | 393 passed |

Both environments passed `pip check`, Ruff 0.8.4, and `compileall -q src`.
The desktop backend initialized, processed a synthetic blank frame with no
detected faces, and closed successfully. This is a dependency/lifecycle check,
not evidence of face-detection accuracy or real-camera performance.
The OpenCV smoke checks exercise BGR-to-RGB conversion and text drawing without
a camera or display. CLI tests verify that version telemetry is emitted.

The first desktop installation attempt hit a certificate-chain error while
fetching setuptools; a subsequent ordinary retry succeeded with certificate
verification still enabled. New Ruff defaults exposed unrelated legacy lint
findings; development dependencies now pin the previously used Ruff 0.8.4.
Python 3.10 was unavailable locally. These October 23 results are historical.
The latest main at `18ce38f` has a verified successful
[CI run](https://github.com/phoclosm/vigil-drowsiness-detection/actions/runs/40273256252).
The final runtime/deployment branch has its own expanded CI gate; its results
must be checked on that branch before merge.
