# M2 D7 Experiment and Evidence Summary

## Scope

D7 audits and regression-tests the existing D4-D6 preprocessing, export/load,
and inference contracts. It does not add an architecture, experiment tracker,
CI framework, or M1-owned runtime feature.

## What Is Verified

The ML test suite verifies the locked eight-channel feature order, training-only
normalization, missing-value masks, finite `(60, 8)` tensors, stride and
segment-boundary behavior, ambiguous-label exclusion, canonical class ordering,
and deterministic grouped splitting. Export/load tests verify model
configuration, fitted normalizer statistics, feature channels, window
configuration, class mapping, provenance, and a live-input warm-up smoke path.

These are automated software-contract and pipeline-integrity checks. They are
not empirical evidence that the classifier detects drowsiness accurately.

## Results Available in This Repository

The repository contains deterministic synthetic fixtures and executable tests,
but no real labeled driver recordings. Consequently, no defensible real-world
accuracy, precision, recall, macro F1, latency benchmark, robustness result, or
safety-performance result is available for v0.1.0.

Synthetic test outputs may demonstrate that a model can be trained, serialized,
loaded, and evaluated without violating the data and tensor contracts. They
must not be reported as a benchmark for real drivers.

## Evidence Still Needed

Meaningful model-quality evaluation requires:

1. Real recordings labeled under the M2 ground-truth protocol.
2. Subject-disjoint training, validation, and final test partitions.
3. Per-class support and accuracy, precision, recall, macro F1, and confusion
   matrices from an executed independent test run.
4. Calibration and abstention/coverage analysis under missing features.
5. Runtime latency measurements on the intended deployment hardware.
6. Analysis of demographic, environmental, camera, and failure-mode coverage.

Until those studies exist, the implementation should be treated as a prototype
pipeline with tested interfaces, not as a validated drowsiness detector.

