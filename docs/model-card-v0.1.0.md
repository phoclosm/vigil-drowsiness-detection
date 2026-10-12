# Vigil Drowsiness Detection Model Card — v0.1.0

## Summary

This model card describes the M2 temporal classifier and its software contract at
version `0.1.0`. It documents the implementation and the evidence available in
this repository; it is not a claim of validated driver-monitoring performance.

No real labeled driver recordings are present in the repository. The automated
results described here are contract and pipeline tests using controlled synthetic
fixtures. Real-world accuracy, precision, recall, F1, latency, robustness, and
safety performance have not been measured.

## Intended Use

The model is intended for:

- Research and educational work on temporal ocular-feature classification.
- Offline experimentation with the M2 `SampleRecord` schema.
- M1 integration through the documented, model-only live inference interface.

It is not intended for autonomous vehicle control, safety-critical alerts without
independent validation, medical diagnosis, driver fitness decisions, surveillance,
or deployment as a certified automotive safety system.

## Inputs and Preprocessing

The model consumes a sequence of exactly 60 records represented as eight ordered
channels:

1. `ear_avg`
2. `is_eye_closed`
3. `blink_duration_ms`
4. `perclos`
5. `ear_avg_valid`
6. `is_eye_closed_valid`
7. `blink_duration_valid`
8. `perclos_valid`

The three continuous features are standardized with statistics fitted only on
the training partition. Missing continuous values are imputed to zero on the
normalized scale and paired with validity masks. Binary indicators and masks are
not normalized. Windows use a locked stride of 15, do not cross session or
subject boundaries, are cut at temporal discontinuities, and exclude windows
containing ambiguous labels.

The live input type is `LiveInputFrame`. It intentionally has no ground-truth
label fields. The labeled `SampleRecord` type remains the training and offline
evaluation schema.

## Model Architecture

`GRUTemporalClassifier` is a compact, unidirectional GRU sequence classifier:

- Input shape: `(batch, 60, 8)`, `float32`.
- One GRU backbone by default, with configurable hidden width.
- Final-timestep representation.
- Linear head producing three logits.
- Cross-entropy-compatible logits during training.
- Softmax probabilities only through the prediction/inference helpers.

Exported bundles contain model weights, architecture configuration, the fitted
normalizer, canonical class mapping, locked window metadata, feature channels,
and validated model-development provenance.

## Output Contract

The inference response always includes:

- `predicted_class`
- `class_id`
- `probabilities`
- `confidence`
- `status`
- `latency_ms`

Statuses are `STATUS_OK`, `STATUS_INSUFFICIENT_HISTORY`,
`STATUS_INVALID_INPUT`, and `STATUS_ERROR`. Prediction-dependent values are
`None` when no valid prediction exists. The engine preserves warm-up, stride,
session-boundary, and temporal-discontinuity behavior.

## Classes

The canonical class order is:

- `0`: `ACTIVE`
- `1`: `DROWSY`
- `2`: `SLEEPING`

## Training and Evaluation Methodology

Training uses deterministic grouped splitting: subject-level grouping when
multiple subjects are available, with session-level fallback for a single
subject. Normalizer statistics are fitted on training records only. The best
checkpoint is selected by validation loss, so both training and validation
groups are recorded as model-development provenance.

Independent evaluation requires explicit, validated provenance. Evaluation
records must be disjoint from model-development groups according to the
strategy. The supplied independent set is evaluated directly without creating a
second train/validation split. The D3 rule baseline is evaluated on matching
window endpoints and all eligible supplied frames.

## Available Evidence

Available evidence is limited to automated tests that verify:

- Feature-channel ordering, masks, normalization behavior, finite tensors, and
  locked `(60, 8)` window dimensions.
- Temporal stride, segment boundaries, ambiguous-label exclusion, and grouped
  split invariants.
- Export/load round trips, strict bundle validation, provenance preservation,
  and live inference after warm-up.
- Metric and reporting mechanics on synthetic fixtures.

These tests establish software behavior and reproducibility of the pipeline.
They do not establish model quality or generalization.

## Limitations, Risks, and Safety

- There are no real labeled driver recordings or independently measured
  real-world metrics in this repository.
- Synthetic fixtures do not represent driver diversity, lighting, camera
  placement, occlusion, sensor failure, or fatigue prevalence.
- The model has not been calibrated for alert thresholds or safety decisions.
- Grouped validation is not a substitute for a multi-subject, independently
  labeled test set.
- Missing landmarks and feature dropouts can reduce coverage or produce invalid
  responses.
- M1 owns capture, landmark extraction, alerts, UI, and operational integration;
  those concerns are outside this model card.

Before any safety-related use, the project needs real labeled recordings,
subject-disjoint train/validation/test evaluation, calibration analysis,
subgroup and failure-mode analysis, measured runtime latency on target
hardware, and review of alert behavior by qualified stakeholders.

## Reproducibility

The implementation records the grouped provenance, normalizer statistics,
window configuration, class mapping, and model configuration in exported
bundles. Deterministic seeds are used by the training pipeline. The repository
does not contain a released trained-weight artifact or real recording dataset.
Results should therefore be reproduced only with the repository's controlled
fixtures unless an independently documented dataset is supplied.

