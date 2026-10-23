# M2 D1 Development Data and Experiment Harness

## Scope and evidence boundary

Day 1 adds mechanics for an approved-data manifest and a reproducible
development experiment record. The repository still contains no approved
participant recordings, and the tests use deterministic synthetic records only
to exercise validation and serialization. They are not model-quality evidence.

Raw recordings, credentials, trained bundles, and large generated artifacts
remain outside Git. A manifest is metadata, not proof by itself: the
authorization and consent references must resolve to evidence maintained by the
project owners before a run is treated as a real-data experiment.

## Approved-data manifest

`ApprovedDataManifest` in `src/vigil/ml/data/manifest.py` records:

- dataset name, version, and source URI;
- authorization and consent evidence references;
- an explicit mapping for `ACTIVE`, `DROWSY`, `SLEEPING`, and `AMBIGUOUS`;
- subject/session-aware recording references and `train`, `validation`, or
  `held_out` assignments.

Manifest validation rejects duplicate recordings, path traversal, inconsistent
session identities, and group assignments that cross splits. JSONL loading
preserves blank-line handling but fails with path and line context for malformed
records or manifest identity mismatches. Held-out recordings cannot enter the
development runner.

No public dataset is automatically approved by this workflow. D8 blockers
remain for UTA-RLDD, DROZY, and YawDD: access, terms, consent/privacy, and a
defensible frame-level canonical label mapping must be resolved first.

## Reproducible development runs

`run_development_experiment()` reuses the existing grouped trainer and records:
the Git SHA, manifest identity, seed, actual train/validation groups, GRU
configuration, Adam learning rate and weight decay, epoch count, best epoch,
validation metrics, optional checkpoint digest, and explicit synthetic-data
provenance. The trainer continues to fit normalization only on training
records and never receives manifest-held-out records.

The development harness does not run final held-out evaluation, model sweeps,
calibration, release export, or real-data benchmarking. Those are deferred to
later sprint days after approved data and a frozen final-test group exist.
