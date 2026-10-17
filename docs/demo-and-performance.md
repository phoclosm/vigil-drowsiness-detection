# Vigil v0.1 Demo and Performance Record

## Current evidence status

The end-to-end demo command and deterministic integration tests are present.
No approved trained model bundle, consented demo recording, independent
real-world evaluation dataset, or hardware benchmark session is available in
the repository. Consequently, this document publishes no accuracy, FPS,
latency, or model-quality number.

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

No measured run is published yet. Populate this table only from the bounded
benchmark command and retain the complete command output outside version
control unless a small report is explicitly approved.

| Revision | Date | Hardware and OS | Input and resolution | Frames | Configuration | Measured FPS | Capture latency ms | Landmark latency ms | Eye-feature latency ms | Result |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Pending measured run | - | - | - | - | - | - | - | - | - | Not measured |

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
- [ ] Hardware FPS and latency have been measured and recorded.
- [ ] Independent real-world model quality has been evaluated.

Unchecked items are release limitations, not implied successes.
