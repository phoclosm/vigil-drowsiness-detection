# Vigil v0.1 Demo and Performance Record

## Current evidence status

The end-to-end demo command and deterministic integration tests are present.
No approved trained model bundle, consented demo recording or independent
real-world evaluation dataset is available for this runtime handoff.
A real bounded webcam vision benchmark was measured on October 26 below.
No accuracy or model-quality result is claimed.

Unit-test fixtures and the temporary untrained bundles used for contract tests
are not evidence of real-world detection quality or runtime performance.

## Demo asset manifest

| Asset | Status | Evidence boundary |
| --- | --- | --- |
| System architecture diagram | Available in `docs/architecture.md` | Describes implemented module and data flow |
| Live UI screenshot | Not captured | Requires an approved bundle and privacy-reviewed input |
| Demo video | Not captured | Requires consent, privacy review, and maintainer approval before publication |
| Trained model card | Not supplied | Must accompany any future model-quality claim |

The absence of screenshots and video is intentional. A synthetic state or
confidence must not be presented as though it came from a trained model on a
real driver.

## Reproducible demo capture procedure

1. Use a model bundle whose training provenance and preprocessing settings are
   documented. Keep the checkpoint outside Git.
2. Select a webcam session or video with documented consent and licensing.
   Remove or obscure unrelated people, screens, vehicle identifiers, and other
   personal information before publication.
3. Record the Git revision, operating system, Python/OpenCV/MediaPipe/PyTorch
   versions, hardware, exact command, bundle identifier, and input identifier.
4. Run the command from the README. Wait for a genuine model prediction; do
   not edit the displayed state or confidence.
5. Keep raw screenshots and recordings under the ignored `artifacts/`
   directory until a maintainer reviews them for privacy, size, relevance, and
   licensing.
6. If the demo fails or the state remains unavailable, preserve the failure
   details rather than substituting a successful-looking asset.

## Runtime performance table

A measured vision-only run is published below. No fatigue classifier was loaded.
Task initialization occurred before the benchmark timer; capture and processing
were measured together, including the first capture. No warm-up frames were
discarded. Input was a stationary webcam session; no raw frames were retained.

| Revision | Date | Hardware and OS | Input and resolution | Frames | Configuration | Measured FPS | Capture latency ms | Landmark latency ms | Eye-feature latency ms | Result |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `ef4ce53` | 2026-10-26 | i9-12900H, Windows 11 build 26300 | Webcam 0, 640×480 | 120 | Tasks IMAGE CPU; EAR 0.20; PERCLOS 30 | 22.838 | 11.962 mean | 31.607 mean | 0.074 mean | 120/120 frames with a detected face |

Exact benchmark revision: `ef4ce53b1757a79b5dbb76c1c06671c806caa2d7`.
Only deployment-related changes were uncommitted during this run; the benchmark
and vision implementation were committed. Python 3.12.0, NumPy 2.5.4,
opencv-contrib-python 5.0.0.93 (`cv2 5.0.0`), MediaPipe 0.10.35,
official float16 version-1 task asset with SHA-256 documented in
[modern landmarks](modern-landmarks.md). The machine also has Intel Iris Xe and
an NVIDIA RTX 3060 Laptop GPU; neither was used by the CPU XNNPACK landmark path.
OS identifier: `Windows-11-10.0.26300-SP0`. Power mode/thermal state were not
controlled. A local untrained cloud-service container was running concurrently.

```powershell
python -m vigil.benchmark --webcam 0 --frames 120 --landmark-model artifacts/vision/face_landmarker.task --closure-threshold-ear 0.20 --perclos-window-frames 30
```

The EAR/PERCLOS values above are explicit benchmark settings, not validated
fatigue thresholds. Elapsed time was 5.254421 seconds. Per-stage measurements:

| Stage | Samples | Minimum ms | Mean ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| Capture | 120 | 1.947 | 11.962 | 371.017 |
| Face landmarks | 120 | 24.695 | 31.607 | 61.668 |
| Eye features | 120 | 0.050 | 0.074 | 0.137 |

An earlier exploratory 30-frame run on `e59da29` is retained here as well:
2.115087 seconds, 14.184 FPS. Capture min/mean/max 1.498/38.682/944.724 ms;
landmarks 24.456/31.622/65.700 ms; eye features 0.049/0.064/0.102 ms (n=30 each).
That run did not record resolution or face coverage and overlapped a Docker
build, so its metadata is incomplete. Neither run is a stable performance
guarantee or model-quality evaluation. Local model latency and cloud network
round-trip latency remain unmeasured with an approved trained bundle.

For every numeric row, report the benchmark's measured sample count and
minimum, mean, and maximum latency rather than copying values from tests or
another machine. The model-inference latency reported by a bundle-backed demo
is a separate stage and must include the bundle identifier and inference
configuration.

## Model-quality table

No real-world evaluation has been completed. Do not populate this table from
synthetic fixtures or from data used for training or validation.

| Revision | Model bundle | Independent evaluation data | Samples or windows | Accuracy | Macro precision | Macro recall | Macro F1 | Limitations |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pending independent evaluation | Not supplied | Not supplied | - | - | - | - | - | Not measured |

Any future row must identify the split strategy, evaluation subjects or
sessions, class mapping, coverage or abstention count, and evidence that the
evaluation groups do not overlap model-development groups.

## v0.1 verification checklist

- [x] Webcam and local-video capture interfaces are implemented.
- [x] Face/eye landmarks and missing-observation handling are implemented.
- [x] EAR, blink, and PERCLOS feature extraction are implemented.
- [x] Temporal bundle loading and live inference integration are implemented.
- [x] Canonical state and confidence overlay is implemented.
- [x] The live CLI closes on `q`, `Q`, Escape, video end, or a frame limit.
- [x] Deterministic automated tests run without requiring a camera.
- [ ] A trained, provenance-complete bundle has been supplied for the demo.
- [ ] A privacy-reviewed live screenshot or demo video has been captured.
- [x] Hardware vision FPS and stage latency have been measured and recorded.
- [ ] Independent real-world model quality has been evaluated.

Unchecked items are release limitations, not implied successes.
