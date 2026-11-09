# M2-to-M1 Handoff Status

**Status:** `BLOCKED — NO APPROVED TRAINED GRU BUNDLE`

## Evidence boundary

The repository currently has no configured `VIGIL_DATA_ROOT`, participant/session
records, approved real-data manifest, trained checkpoint, or private bundle
artifact. The existing M2 documentation states that non-empty manifest
authorization/consent fields are metadata only and are not proof of approval.
No synthetic fixture is a model release or model-quality evidence.

## Delivery state

| Area | State | Evidence |
|---|---|---|
| Bundle exporter/loader | Implemented and locally tested | GRU-only versioned bundle path preserves model, normalizer, and metadata. |
| Normalizer provenance round-trip | Implemented and locally tested | New bundles preserve provenance; legacy bundles remain loadable and unverified. |
| M1 cloud contract | Implemented and locally tested | `raw_with_masks_v1`, 60 rows, locked 8-channel order, stride 15, and single server-side normalization. |
| M2-to-M1 integration smoke | Locally tested | Uses an explicitly untrained synthetic infrastructure fixture only. |
| CI | Passed | PR #28 head `56a7cbd509aa70671f418c9eae39f3a385e11a8c` had 14/14 passing CI checks. Hosted CI is not equivalent to real-data or deployment approval. |
| Trained on approved real data | Not done | No approved records are available. |
| Independently evaluated | Not done | Held-out evaluation is prohibited until all data/governance gates pass. |
| Approved by owner | Not done | Requires external data and model-governance approval. |
| Deployed by M1 | Not done | No approved trained bundle exists for deployment. |

## Feature-setting handoff

M1 receives the feature settings from the runtime construction sites:

- EAR closure threshold: supplied by `--closure-threshold-ear`.
- Minimum blink duration: supplied by `--minimum-blink-duration-ms` (optional,
  default `0 ms`).
- PERCLOS window: required `--perclos-window-frames`.

EAR threshold and PERCLOS window are required operator configuration. The
minimum blink duration is optional and defaults to `0 ms`. Example/test values
commonly use EAR `0.20`, blink duration `75 ms`, and PERCLOS `60` frames; these
are not empirically approved training settings. The exact values used to create
training features must be recorded in the eventual private bundle metadata and
must match M1's live configuration. No such approved training configuration
exists yet.

## M1 cloud contract

The existing M1 `BundlePredictor` accepts 60 raw rows in
`raw_with_masks_v1`, in the locked channel order:

1. `ear_avg`
2. `is_eye_closed`
3. `blink_duration_ms`
4. `perclos`
5. `ear_avg_valid`
6. `is_eye_closed_valid`
7. `blink_duration_valid`
8. `perclos_valid`

The service applies the loaded bundle normalizer exactly once and returns the
canonical three-class response. The M2 regression is an infrastructure check,
not a trained-model demonstration.

## Required external unblockers

1. Approved and accessible real participant/session records.
2. Verifiable authorization, licensing/permitted-use, consent, and privacy evidence.
3. Reviewed frame-level mapping to `ACTIVE=0`, `DROWSY=1`, `SLEEPING=2`, with
   `AMBIGUOUS=-1` excluded from supervised evaluation.
4. Frozen leakage-free subject/session train, validation, and held-out groups.
5. Real GRU training, development-only checkpoint selection, calibration, and
   selection-freeze provenance.
6. Independent held-out evaluation authorization and M1 deployment approval.

Until these are supplied, M1 must not receive the synthetic fixture as a final
demo model.
