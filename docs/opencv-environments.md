# OpenCV environments

The desktop and OpenCV 5 experiments require separate virtual environments.
Never install multiple OpenCV wheel variants into one environment: they share
the `cv2` namespace. Core/ML installation does not require OpenCV or MediaPipe.

Supported desktop setup (Python 3.10 or 3.12):

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[desktop,ml,dev]"
.venv/Scripts/python -m pip check
.venv/Scripts/python -m pytest
```

The desktop extra pins MediaPipe 0.10.21, which provides the existing
`solutions.face_mesh` API, NumPy below 2, and its required contrib OpenCV wheel
at 4.11.0.86. Feature order, landmark indices, model inputs and normalization
are unchanged.

OpenCV 5 experiment (use a different environment):

```powershell
python -m venv .venv-opencv5
.venv-opencv5/Scripts/python -m pip install -e ".[opencv5,ml,dev]"
.venv-opencv5/Scripts/python -m pip check
.venv-opencv5/Scripts/python -m pytest
```

This environment exercises OpenCV 5 operations and software contracts but does
not provide live MediaPipe landmarks. Do not use it as the desktop demo setup.
The full-suite fake-backend tests cannot establish real Face Mesh compatibility.

## Migration blocker observed on October 23 2026

A fresh Python 3.12 environment rejected `opencv-python==5.0.0.93` together
with `mediapipe==0.10.21` with `ResolutionImpossible`: OpenCV requires NumPy
at least 2, whereas this MediaPipe version requires NumPy below 2.
Newer MediaPipe versions without `solutions.face_mesh` cannot serve as a
drop-in replacement. Installing with `--no-deps` would hide these constraints
and is not a supported workaround.

The OpenCV 5 desktop acceptance gate remains blocked. Resolving it requires a
separately reviewed landmark-backend migration with evidence that the existing
feature contract is preserved. No cloud dependencies or services are added here.
CI tests both environments on Python 3.10 and 3.12; real camera/display checks
remain manual. Version telemetry reports the imported runtime, not a benchmark.

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
Python 3.10 was unavailable locally. GitHub Actions results remain unverified
until this branch is pushed and its workflow executes.
