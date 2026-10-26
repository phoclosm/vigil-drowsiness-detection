# Modern landmark adapter

The default runtime uses MediaPipe Tasks Face Landmarker in synchronous IMAGE
mode. One input frame produces one result; there is no asynchronous frame loss,
tracking timestamp substitution or temporal smoothing. Tracking-based performance
parity with the legacy backend has not been established. EAR/blink/PERCLOS math,
missing-observation behavior and the model feature order are unchanged.

The official 478-point task model extends the canonical 468-point face mesh with
iris landmarks. Both the installed Tasks connection tables and Google's
[published topology](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py)
place `(362, 385, 387, 263, 373, 380)` on the left eye and
`(33, 160, 158, 133, 153, 144)` on the right. Startup checks membership in
the respective contours. The adapter requires 478 finite points per detected
face and passes ordered normalized x/y/z coordinates without pixel conversion,
clamping or mirroring. Downstream components receive only Vigil `FaceLandmarks`.

Obtain Google's task model separately; never commit it:

```powershell
New-Item -ItemType Directory -Force artifacts/vision
Invoke-WebRequest https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -OutFile artifacts/vision/face_landmarker.task
Get-FileHash artifacts/vision/face_landmarker.task -Algorithm SHA256
python scripts/vision_smoke.py --model artifacts/vision/face_landmarker.task
```

Observed SHA-256 for version 1:
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
Use the asset according to the model terms linked from the
[official guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
No runtime download occurs. Configure `--landmark-model PATH` for both the demo
and benchmark. A missing/corrupt model, unsupported topology or malformed output
fails explicitly. No-face results are empty tuples, not fabricated coordinates.

For a comparison in a separate legacy environment, use `--landmark-backend legacy`.
The legacy adapter is retained unchanged; do not mix its dependency extra with
the Tasks desktop extra. Coordinate topology is verified, but numerical EAR
parity on real faces and threshold suitability require recorded evaluation and
owner review. A synthetic blank-frame smoke is not evidence of facial accuracy.
