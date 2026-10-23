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

The canonical labels are fixed as `ACTIVE=0`, `DROWSY=1`, `SLEEPING=2`, and
`AMBIGUOUS=-1`. The manifest mapping is metadata only; `load_records()` does
not convert source labels. Manifest-referenced JSONL files must already contain
canonical, validated `SampleRecord` labels. Conversion from source annotations
belongs to a separate reviewed ingestion step. Unsupported or noncanonical
labels fail loudly and are never silently translated.

Manifest validation rejects duplicate recordings, path traversal, inconsistent
session identities, and group assignments that cross splits. JSONL loading
preserves blank-line handling but fails with path and line context for malformed
records or manifest identity mismatches. The runner accepts a data root and
loads only the manifest-listed train and validation paths; arbitrary caller
records are not accepted. Resolved paths must remain inside that root, so
traversal and symlink escapes are rejected. Held-out recordings are never
loaded into the development runner.

No public dataset is automatically approved by this workflow. D8 blockers
remain for UTA-RLDD, DROZY, and YawDD: access, terms, consent/privacy, and a
defensible frame-level canonical label mapping must be resolved first.

## Reproducible development runs

`run_development_experiment()` enforces the manifest partitions and delegates
to the existing partition-aware trainer without creating a second split. The
trainer fits the normalizer only on the loaded train records, extracts windows
with the existing rules, and computes model-selection metrics only on the
loaded validation records. The experiment record contains:
the Git SHA, manifest identity, seed, actual train/validation groups, GRU
configuration, Adam learning rate and weight decay, epoch count, best epoch,
validation metrics, the required SHA-256 digest of the checkpoint written to
the caller-provided artifact directory, and explicit synthetic-data provenance.
The artifact directory must already exist and should remain outside Git. This
proves which manifest partitions, software configuration, and checkpoint bytes
were consumed; it does not independently prove authorization or consent.

The development harness does not run final held-out evaluation, model sweeps,
calibration, release export, or real-data benchmarking. Those are deferred to
later sprint days after approved data and a frozen final-test group exist.
