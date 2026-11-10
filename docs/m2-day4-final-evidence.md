# M2 Day 4 Final Evidence

## Release status

**BLOCKED — NO APPROVED TRAINED GRU BUNDLE**

| Evidence state | Status | Evidence boundary |
|---|---|---|
| Implemented | PASS | GRU bundle export/load, provenance validation, and M1 contract integration exist. |
| Locally tested | PASS | Focused and repository test suites pass in the development environment. |
| CI tested | PASS | Implementation commit `83234fad467d1dc04a95a32b88d8478a18cbda97` was validated by push workflow `42658161401` and pull-request workflow `42658174732`; each completed 7/7 jobs successfully. |
| Trained on approved real data | FAIL | No configured data root, approved manifest, or accessible approved records. |
| Independently evaluated | BLOCKED | Cannot begin without approved data, frozen groups, and authorization. |
| Calibrated | BLOCKED | No trained candidate exists to calibrate. |
| Owner approved | BLOCKED | Human governance and release approval are outstanding. |
| Deployed by M1 | BLOCKED | M1 has no approved trained bundle to deploy. |

## M1 integration evidence

The local M1 contract smoke uses a controlled, explicitly untrained
infrastructure fixture. It verifies the 60-row, eight-channel
`raw_with_masks_v1` request, bundle digest, one-time normalization, finite GRU
inference, and response schema. It is not a trained-model test, hosted
deployment test, accuracy result, or model-quality evidence.

Verified normalizer and partition provenance is software lineage evidence only.
It does not prove consent, licensing, privacy approval, human approval, or
release authorization. No approved real participant records or reviewed
canonical frame-level labels are available. No real model training, calibration,
candidate ranking, or independent held-out evaluation has occurred.

## External unblockers

The smallest acceptable path is:

1. Provide an approved manifest and accessible participant/session records.
2. Provide externally verifiable authorization, licensing, consent, and privacy
   evidence plus reviewed frame-level canonical labels.
3. Freeze subject/session-disjoint train, validation, and held-out groups.
4. Train and select a GRU checkpoint using development data only, preserving
   complete provenance and exact feature settings.
5. Obtain calibration and independent held-out evaluation authorization, then
   complete evaluation without tuning on held-out results.
6. Obtain owner approval and give M1 the private bundle location and digest.

Non-empty manifest strings and synthetic fixtures do not satisfy these gates.
