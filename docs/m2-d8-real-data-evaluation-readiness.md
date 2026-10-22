# M2 D8 Real-Data Evaluation Readiness

**Assessment date:** 2026-10-22  
**Scope:** Prepare a reproducible path to evaluate the existing M2 temporal
classifier on approved, labeled driver recordings. This document does not
report model-quality results.

## Executive finding

No real dataset is currently verified as ready for evaluation in this
repository. The UTA Real-Life Drowsiness Dataset (UTA-RLDD) is the strongest
candidate found, because its official page describes 60 subjects, 180 RGB
videos, three drowsiness conditions, subject folds, and a public research
contact. However, the page does not state a standard open-source license or
provide a verified direct download workflow, and its labels are
participant-provided predominant video labels rather than frame-level
annotations. Access, terms of use, consent/privacy conditions, and an
annotation conversion protocol must therefore be confirmed with the
maintainers before any data is acquired or evaluated.

The repository contains no real recordings. Synthetic fixtures must not be
used as evidence of real-world performance.

## 1. Existing M2 contract and integrity rules

The source data must be converted to the existing `SampleRecord` contract; the
model interface must not be changed.

- Canonical labels are `0: ACTIVE`, `1: DROWSY`, and `2: SLEEPING`.
  `AMBIGUOUS` is `label_id = -1` and is excluded from supervised windows.
- Each record needs an anonymized `subject_id`, `session_id`, recording/frame
  identity, strictly increasing `frame_index` and `timestamp_ms`, facial
  validity flags, the source ocular features, and a validated label pair.
- The model uses the locked channels, in order:
  `ear_avg`, `is_eye_closed`, `blink_duration_ms`, `perclos`,
  `ear_avg_valid`, `is_eye_closed_valid`, `blink_duration_valid`,
  `perclos_valid`.
- Windows are exactly 60 samples with stride 15. Windows do not cross a
  session or subject boundary, a frame gap greater than one, or a timestamp
  gap greater than 200 ms.
- Continuous normalization is fitted on development-training records only.
  The independent test set is transformed with that frozen normalizer.
- Evaluation requires validated model-development provenance. Under
  `strategy="subject"`, test subjects and sessions must be disjoint from all
  training and validation groups. Under `strategy="session"`, test sessions
  must be disjoint; the same subject is allowed only when the sessions are
  genuinely independent.
- `compare_models()` evaluates supplied independent records directly. It does
  not create a second train/validation split. It reports the GRU and D3
  baseline metrics, including per-class support, coverage/abstentions, and a
  canonical 3x3 confusion matrix.

Sources: [M2 data contract](m2-data-contract.md),
[temporal windowing specification](m2-temporal-windowing.md),
[evaluation protocol](m2-evaluation-protocol.md), and
[runtime/evaluation documentation](m2-runtime-and-evaluation.md).

## 2. Candidate dataset comparison

The access date for every source check below is **2026-10-22**. “Verified”
means verified from the linked source on that date, not inferred from a
third-party implementation or a paper citation.

| Candidate | Authoritative source and verified facts | Access, license, consent/privacy | Labels and identifiers | Fit with this contract | Assessment |
|---|---|---|---|---|---|
| **UTA Real-Life Drowsiness Dataset (UTA-RLDD)** | [Official UTA-RLDD page](https://sites.google.com/view/utarldd/home) states approximately 30 hours of RGB video from 60 adult volunteers: 180 videos, one each for alertness, low vigilance, and drowsiness. It describes five subject folds and self-recorded phone/webcam videos. The [CVPRW paper](https://arxiv.org/abs/1904.07312) is the cited publication. | The official page publishes a [Google Drive folder link](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl). A published link proves that a distribution route is advertised; it does **not** prove that this project can successfully download the files or use/redistribute them. The page describes voluntary/extra-credit participation and says that 36 participants permitted publication of faces, while identities are not provided. It does not state a standard dataset license, commercial-use permission, or complete consent/IRB documentation. | The official page says labels are participant-provided predominant video labels: alert, low vigilant, and drowsy. They are video-level, not frame-level, labels. It identifies subjects for five folds; delivered-file recording/session identifiers still need confirmation. | RGB video is not the eight-channel feature stream. M1-owned face/landmark extraction is required. Alert maps plausibly to `ACTIVE`; low vigilant plausibly to `DROWSY`; “drowsy” is not defensibly `SLEEPING` without an annotation rule. Ten-minute videos can support windows, but frame timestamps and label boundaries must be generated and audited. | **Strongest candidate, not yet approved or usable.** The published Drive route is promising, but successful access, terms, privacy approval, raw-file inventory, and conservative label mapping remain unresolved. |
| **DROZY (ULg Multimodality Drowsiness Database)** | The [current official DROZY site](https://www.drozy.ulg.ac.be/) was identified but could not be retrieved in this environment because of a DNS/TLS transport failure. The [ULiège ORBi archive](https://orbi.uliege.be/handle/2268/191620) provides the dataset paper, and the [DROZY license agreement](https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf) states that the database is downloadable from ORBi. The paper reports 14 healthy subjects (3 male, 11 female), about 500,000 frames, and approximately 283 GB. Modalities are synchronized KSS, PVT reaction-time data, PSG (5 EEG, 2 EOG, ECG, EMG), Kinect v2 NIR intensity/range images, and 68 facial landmarks per frame in 2D and 3D. | The agreement permits **research and evaluation only** and prohibits commercial product development. It prohibits redistribution in whole or part without prior written approval, database modification, and revealing subject identities. Images may only be displayed when marked publishable; required citation and Belgian-law terms also apply. Permission to extract and retain derived ocular features should be checked with ULg-INTELSIG under the agreement before processing; the agreement does not by itself establish this project's derivative-feature workflow. | KSS is a self-reported level-of-drowsiness score taken just before each PVT, not a frame-level `ACTIVE`/`DROWSY`/`SLEEPING` annotation. The paper describes three PVTs per subject and synchronized signals. It reports 720 manually landmark-annotated frames (about 51 per subject) plus automatic 68-landmark tracking for all frames. Stable subject/session IDs and the exact downloadable file layout still require an access-controlled inventory. | The landmarks can support ocular feature derivation without changing the M2 model interface, but M1-owned extraction or an approved equivalent is needed. KSS/PVT/PSG are useful supporting evidence, not an automatic three-class mapping. The 14-subject, sleep-deprivation protocol supports grouped sessions, but KSS granularity, sparse manual labels, identity restrictions, and no direct sleeping-class annotation prevent an unqualified three-class benchmark. | **Technically promising but legally and label-wise gated.** It is not directly suitable for a three-class benchmark without a defensible annotation protocol, explicit derived-feature permission, and an identity-safe manifest. |
| **YawDD: Yawning Detection Dataset** | The current [IEEE DataPort record](https://doi.org/10.21227/e1qm-hb90) describes in-car-camera videos in natural/varying illumination. Set 1 has 322 videos covering normal driving, talking/singing, and yawning, with 3–4 videos per subject; set 2 has 29 subject videos containing all three activities. The record states that videos are unlabeled because yawning sequences are visually apparent. | The IEEE record is marked **Open Access**, but the page requires an IEEE DataPort login for access. Its metadata exposes a [CC BY 4.0 license link](https://creativecommons.org/licenses/by/4.0/); the exact record-level terms and any restrictions beyond that metadata should be confirmed after login. The participant-information file controls whether screenshots may be publicly shared for each video. No claim is made here about permission to redistribute raw videos or derive a new labeled database without checking the record terms. | The record describes activity/yawning content rather than drowsiness ground truth. Videos are explicitly unlabeled; subjects and video counts are described, but a project-ready frame/session manifest and annotation timestamps are not supplied by the page. | In-car video is a plausible M1 input source, but yawning is not equivalent to `DROWSY` or `SLEEPING`; talking/singing and silent driving are not canonical fatigue labels. Ocular channels would require M1-owned landmark extraction, and labels would require new protocol-based annotation with uncertain intervals excluded. | **Accessible in principle, not suitable as-is.** Login and exact terms must be verified, and substantial new annotation would be required. It cannot support a valid three-class evaluation merely because it contains yawning videos. |

### Recommendation

Pursue UTA-RLDD first, but treat it as a gated candidate rather than an
approved dataset. The official page's Google Drive folder is evidence of a
published distribution route, not evidence of successful access or permission
for this project. Request from the UTA-RLDD maintainers:

1. An approved download procedure and the governing license/terms, including
   research, commercial, redistribution, and derivative-feature permissions.
2. Consent/privacy documentation and confirmation that local feature extraction
   is allowed without committing video or identifiable frames.
3. The exact subject, video, and annotation identifiers and whether the
   advertised five folds may be used as an untouched final-test split.
4. Annotation granularity and an agreed mapping from alert/low-vigilant/drowsy
   to the M2 classes. Do not infer `SLEEPING` from the word “drowsy.”
5. Any required citation, access agreement, or institutional review.

Also contact ULiège/ULg-INTELSIG before using DROZY-derived features, and use
the IEEE DataPort login and record terms to verify YawDD access and any
participant-information restrictions. Across all three candidates, no source
currently provides a verified, approved, frame-level three-class
`ACTIVE`/`DROWSY`/`SLEEPING` evaluation set for this project. Until access,
terms, consent/privacy review, feature permissions, and a label audit are
complete, there is no legally and practically verified real-data evaluation
set.

## 3. Required ingestion and manifest

Raw videos and identifiable recordings must remain outside this repository.
The ingestion process should create an access-controlled manifest and derived
JSONL records only after permission and privacy review.

### 3.1 Manifest fields

Each recording manifest row should contain:

```text
dataset_name
dataset_version_or_release
source_url
access_date
license_or_terms_reference
consent_privacy_reference
subject_id                 # stable pseudonym; never a name or direct identifier
session_id                 # one continuous source recording/session
recording_id               # source-provided ID or a documented local hash/slug
source_file_digest         # optional, access-controlled; never the video itself
source_frame_rate
source_duration_ms
annotation_source
annotation_version
annotation_granularity     # frame, interval, video, or unknown
source_label
mapped_label
mapped_label_id            # 0, 1, 2, or -1 for AMBIGUOUS
mapping_rule_version
exclusion_reason           # empty only when eligible
review_status               # pending, accepted, excluded, or needs_review
```

The manifest must retain excluded recordings and the reason for exclusion.
Examples include missing consent terms, unavailable subject identity,
ambiguous labels, missing eye visibility, broken timestamps, duplicate
recordings, insufficient continuous frames, or overlap with a development
group. A guessed label must never be represented as an accepted mapping.

### 3.2 Derived `SampleRecord` requirements

The feature-extraction stage must emit one validated record per usable frame:

- Use the manifest's pseudonymous `subject_id`, `session_id`, and
  `recording_id` in a deterministic `sample_id`.
- Preserve source frame order and generate elapsed `timestamp_ms`; record the
  source frame rate and timing assumptions in session metadata.
- Derive `ear_avg`, `is_eye_closed`, `blink_duration_ms`, and `perclos` only
  through the existing M1 feature pipeline. Missing landmarks remain missing
  and are represented by the existing validity masks.
- Carry only labels supported by the annotation protocol. Intervals that cross
  a state transition, lack frame-level support, or cannot distinguish
  `DROWSY` from `SLEEPING` must be `AMBIGUOUS` or excluded.
- Validate strict monotonicity, one subject per session, duplicate identities,
  finite values, and the existing session metadata before window extraction.

M1 owns video capture, face/landmark extraction, and instantaneous feature
calculation. M2 owns the manifest, mapping review, schema validation,
normalization, windowing, training provenance, and evaluation. This D8 work
does not implement or alter M1 components.

## 4. Development and final-test methodology

The final evaluation must be independent of every group used to fit the
normalizer, train the model, or select its checkpoint.

1. Freeze an approved dataset release and manifest version.
2. Reserve the final test subjects before model development. Do not inspect
   final-test metrics during preprocessing or checkpoint selection.
3. With multiple subjects, use `strategy="subject"` and assign every session
   of a subject to exactly one development partition or final test. A
   subject must not appear in both development and final test.
4. Within development, perform the existing deterministic subject-level
   train/validation split. Use session-level fallback only for a genuinely
   single-subject development set and state that cross-subject generalization
   is not measured.
5. Fit the normalizer on development-training records only. Record the
   training and validation subjects and sessions in model provenance.
6. Validate final-test group separation against all training and validation
   provenance before extracting windows.
7. Extract eligible windows from the final-test records directly. Do not split
   the final-test records again, and do not allow windows across sessions,
   subjects, gaps, or ambiguous intervals.
8. Require non-zero support for all three canonical classes before calling the
   result a three-class benchmark. Otherwise report the missing support and
   stop the performance claim.

For a dataset that cannot support a defensible `SLEEPING` class, the result
may be useful for a scoped exploratory analysis, but it is not a valid
three-class M2 benchmark.

## 5. Reproducible evaluation command and artifacts

After a verified bundle and approved JSONL test directory exist, the intended
entry point is:

```python
from vigil.ml.evaluation.comparison import evaluate_recording_directory

report = evaluate_recording_directory(
    directory_path="data/approved/<dataset>/<release>/final_test",
    bundle_path="artifacts/models/<bundle>.pt",
)
if report is None:
    raise RuntimeError("No approved evaluation recordings were found")

with open("artifacts/evaluation/<run_id>.json", "w", encoding="utf-8") as f:
    import json
    json.dump(report.to_dict(), f, indent=2, sort_keys=True)

with open("artifacts/evaluation/<run_id>.md", "w", encoding="utf-8") as f:
    f.write(report.to_markdown())
```

The run record must additionally preserve:

- Dataset/release, manifest hash, mapping-rule version, and access/terms
  reference.
- Model bundle digest and training provenance.
- Exact source commit, Python/PyTorch versions, and evaluation timestamp.
- Number of raw records, sessions, subjects, and dropped/excluded records.
- Number of eligible 60-frame windows and the window-endpoint policy.
- Per-class support for `ACTIVE`, `DROWSY`, and `SLEEPING`.
- Accuracy, per-class and macro precision, per-class and macro recall, macro F1.
- Canonical 3x3 confusion matrix with rows as truth and columns as prediction.
- Missing-feature coverage: per-channel valid/missing counts or rates,
  window-level usable/invalid counts, and baseline abstentions.
- The synthetic-fixture flag must be false only for a verified real-data run.

The existing `BaselineEvaluationReport` and `ModelComparisonReport` already
provide the core metrics, canonical confusion matrices, support, coverage,
abstention counts, and independent evaluation metadata. The missing-feature
coverage artifact needs to be produced by the approved ingestion/evaluation
wrapper or a future narrowly scoped extension once real data is available; it
must not be estimated from synthetic fixtures.

## 6. Blockers and prerequisites

### Current blockers

- No real labeled recordings are present in the repository or working tree.
- UTA-RLDD's published Drive folder has not been successfully accessed by this
  project, and its license/terms remain unresolved.
- UTA-RLDD's participant-provided video-level labels do not directly establish
  frame-level `ACTIVE`, `DROWSY`, and `SLEEPING` ground truth.
- DROZY access through the current official domain could not be verified in this
  environment, and its license requires research/evaluation-only use,
  non-redistribution, no database modification, and no identity disclosure.
- DROZY KSS/PVT/PSG evidence and landmarks do not automatically create the
  required three frame-level canonical labels; derived-feature permission must
  be checked.
- YawDD requires IEEE DataPort login, is explicitly yawning-focused, and is
  unlabeled. Exact record-level terms and permission for a new fatigue
  annotation/derived-feature release remain to be verified.
- A source-to-M2 feature extraction run has not been approved or executed.
- No final-test manifest, label audit, or development provenance exists for a
  real dataset.
- Target-hardware latency, calibration, robustness, subgroup coverage, and
  safety performance remain unmeasured.

### Concrete prerequisites

1. Obtain written dataset access, license, consent/privacy, and IRB/ethics
   guidance where applicable.
2. Freeze a versioned, access-controlled raw-data release and manifest.
3. Agree on annotation mapping and exclusion rules with an annotator or
   dataset maintainer; preserve uncertain intervals as `AMBIGUOUS`.
4. Run M1-owned feature extraction without committing videos or sensitive
   artifacts.
5. Audit subject/session/recording identity uniqueness and label support.
6. Create and sign off the held-out final-test partition before training.
7. Train/export a bundle whose provenance records both development partitions.
8. Run the command above and archive the JSON/Markdown report plus manifest
   and bundle digests.

No benchmark is valid until these prerequisites are complete.
