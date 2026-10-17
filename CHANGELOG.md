# Changelog

All notable changes to Vigil are documented in this file.

## 0.1.0 - Unreleased

### Added

- Webcam and local-video capture with explicit lifecycle and timing metadata.
- MediaPipe face/eye landmark extraction with missing-observation handling.
- EAR, blink-event, blink-duration, and trailing PERCLOS feature extraction.
- Rule-based fatigue baseline and deterministic evaluation utilities.
- Leakage-aware temporal datasets, GRU training, validation, and evaluation.
- Versioned PyTorch model-bundle export and validated runtime loading.
- Live temporal inference with canonical `ACTIVE`, `DROWSY`, and `SLEEPING`
  states.
- OpenCV state/confidence overlay and an end-to-end desktop demo command.
- Reproducible architecture, setup, benchmark, and evidence documentation.

### Known limitations

- Vigil v0.1 is a learning prototype, not a production driver-safety or
  medical system.
- No trained checkpoint, recorded driver dataset, or real-world model-quality
  result is distributed with the repository.
- No hardware FPS or latency result has been approved for publication.
- Alerts are visual only; no audio, remote notification, or vehicle control is
  implemented.
- A public release date, Git tag, and GitHub release have not been created.
