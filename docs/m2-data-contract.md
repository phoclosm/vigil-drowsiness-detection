# Vigil M2 Data Contract

## 1. Overview and Scope

This document specifies the canonical data contract for **Milestone 2 (M2: Machine Learning, Deep Learning, and Evaluation)** of the Vigil Driver Drowsiness Detection system. It defines:

1. The canonical fatigue state labels and ground-truth semantics.
2. The structured training record schema implemented by the D2 data recorder.
3. The upstream feature contract consumed from M1 (Vision and Runtime).
4. The downstream inference contract published to M1 for real-time integration.
5. Invariants and architectural boundaries that must remain stable across M2 Day 2 through Day 6.

### 1.1 Subsystem Ownership Boundaries
* **M2 Ownership**: Data schema, training data recorder, rule-based baseline, PyTorch dataset and loader, model architecture, training, validation, evaluation metrics, exported model artifacts, model loading contract, inference input/output specification, preprocessing specification, and class map.
* **M1 Ownership**: Video capture, facial/eye landmark extraction, instantaneous eye/blink feature calculation, live runtime integration, consuming the model inference result, UI/alert behavior, and runtime robustness.
* **Non-Deliverable Notice**: Live runtime integration into the webcam loop and alert dispatch are strictly owned by M1. M2 delivers the model artifact, preprocessing pipeline, and inference interface specification.
* **Operational Notice**: Vigil is a prototype research and engineering system developed for evaluation and educational purposes. It is **not** a certified automotive driver-safety system or medical diagnostic device.

---

## 2. Canonical Fatigue Labels

### 2.1 Class Definitions and Ordering
M2 defines exactly three mutually exclusive fatigue classes. The canonical ordering and zero-indexed integer encodings are permanent and must not drift across any dataset, baseline, model architecture, or evaluation metric:

| Index | Canonical Class | Semantic Definition | Observable Protocol Indicators |
|:---:|:---|:---|:---|
| `0` | `ACTIVE` | The driver is alert, visually attentive to the driving environment, and exhibiting normal, unimpaired wakefulness under the observation protocol. | Eyes open and responsive; natural spontaneous blinking; active gaze; alert head posture. |
| `1` | `DROWSY` | The driver exhibits discernible behavioral or physiological fatigue under the observation protocol. | Sluggish eyelid kinetics; slow or incomplete eyelid reopening; heavy or drooping eyelids; repeated yawning or head-nodding; observable struggle to maintain alertness. |
| `2` | `SLEEPING` | The driver is observed to be in an acute state of sleep or microsleep under the observation protocol. | Sustained complete or near-complete eye closure; loss of active visual engagement with driving environment; head slumped or motionless in sleep posture. |

### 2.2 Ground Truth vs. Heuristic Baseline Thresholds
* **Ground Truth Defined by Protocol**: Ground truth represents the observed human state established strictly by the recording/annotation protocol (e.g., controlled experiment instructions, observer event logs, or verified post-hoc video review).
* **Independence from Numerical Heuristics**: Numerical Eye Aspect Ratio (EAR) values, blink duration thresholds, blink rate ranges, and eye closure duration limits do **not** define ground truth. Such numbers represent heuristic engineering proxies, not physiological ground truth.
* **Distinction from D3 Rule Baseline**:
  * *Ground Truth Labels*: Objective human state determined by protocol annotation.
  * *D3 Rule Baseline*: A heuristic prediction model that applies configurable empirical thresholds to feature streams. These thresholds will be calibrated in D3 against actual recorded data and must never be conflated with the definition of ground truth.

### 2.3 Handling Ambiguity and Edge Cases
* **State Exclusivity**: A subject cannot simultaneously occupy two classes. Transitional frames must be assigned based on the dominant observable state specified by the annotation protocol.
* **Uncertain / Degraded States**: When a label cannot be assigned with confidence (e.g., driver rubbing eyes, squinting due to sudden glare, face occluded, or annotator uncertainty), the sample must be marked as `AMBIGUOUS` (`label_id = -1`) or omitted from recording.
* **Supervised Training Filter**: The downstream PyTorch training pipeline (D4/D5) must strictly reject or filter out any sample where `label_id == -1` or `label == "AMBIGUOUS"`. Unlabeled or ambiguous data must never be used for supervised loss calculation or benchmark evaluation.

---

## 3. Training Record Schema

The D2 data recorder must emit structured, tabular records. The schema is designed for JSON Lines serialization (`samples.jsonl`) accompanied by session-level metadata (`session_meta.json`).

### 3.1 Field Specifications

| Field Name | Type | Unit | Requirement | Valid / Expected Range | Meaning | Missing / Invalid Semantics |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| `sample_id` | `string` | — | Required | Unique string (`{session_id}_f{frame_index}`) | Globally unique sample identifier. | Cannot be null or empty. |
| `session_id` | `string` | — | Required | Unique session slug (e.g., `s_20261006_1530_sub01`) | Identifies the continuous recording session. Critical for grouped splitting. | Cannot be null or empty. |
| `subject_id` | `string` | — | Required | Anonymized identifier (e.g., `subject_001`) | Stable anonymized identifier for the recorded subject. Free of personally identifying information. | Must not be empty. |
| `frame_index` | `int` | frames | Required | $[0, \infty)$ | Zero-indexed frame counter within the session representing the source capture frame. Monotonically increasing; may contain gaps when frames are dropped or skipped by the capture pipeline. | Cannot be negative; cannot decrement or duplicate ($i_k > i_{k-1}$). First recorded frame must be 0. |
| `timestamp_ms` | `float` | ms | Required | $[0.0, \infty)$ | Monotonic elapsed capture time from the beginning of the recording session ($t_0 = 0.0$). | Must be strictly increasing ($t_i > t_{i-1}$). Epoch time is stored in session metadata. |
| `frame_delta_ms` | `float` | ms | Required for frame > 0 | $(0.0, \infty)$ | Time elapsed since previous captured frame ($t_i - t_{i-1}$). Positive measured deltas are preserved without clamping; large values represent stalls/gaps. | Must be `null` for frame 0 (no previous frame exists). |
| `face_detected` | `bool` | — | Required | `True`, `False` | Flag indicating whether a human face was localized in the frame. | If `False`, all downstream facial features are invalid. |
| `landmarks_valid` | `bool` | — | Required | `True`, `False` | Flag indicating whether eye landmark coordinates passed confidence and geometric validity checks. | If `False`, eye features are invalid. |
| `ear_left` | `float` | ratio | Optional | $[0.0, 0.60]$ | Eye Aspect Ratio for the left eye calculated from canonical landmarks. | `null` / `NaN` if `landmarks_valid == False`. Must not be coerced to `0.0`. |
| `ear_right` | `float` | ratio | Optional | $[0.0, 0.60]$ | Eye Aspect Ratio for the right eye calculated from canonical landmarks. | `null` / `NaN` if `landmarks_valid == False`. Must not be coerced to `0.0`. |
| `ear_avg` | `float` | ratio | Optional | $[0.0, 0.60]$ | Mean Eye Aspect Ratio: $(ear_{left} + ear_{right}) / 2.0$. Primary instantaneous metric. | `null` / `NaN` if `landmarks_valid == False`. |
| `is_eye_closed` | `bool` | — | Optional | `True`, `False` | Derived instantaneous indicator comparing `ear_avg` against the configured closure threshold. | `null` if `landmarks_valid == False`. Threshold version recorded in session metadata. |
| `blink_count` | `int` | count | Required | $[0, \infty)$ | Cumulative completed blinks recorded in the current session up to this frame. | Starts at 0 on frame 0; monotonically non-decreasing. |
| `blink_duration_ms` | `float` | ms | Required | $[0.0, \infty)$ | Duration of the currently active or most recently completed eye closure event. | `0.0` if eyes are continuously open with no active closure. |
| `perclos` | `float` | ratio | Optional | $[0.0, 1.0]$ | Percentage of Eye Closure over trailing temporal window. Trailing (causal) only. | `null` during initial warm-up buffer; $[0.0, 1.0]$ when valid. Window config recorded in metadata. |
| `fps` | `float` | fps | Optional | $(0.0, 120.0]$ | Instantaneous or smoothed capture frame rate. | Informative metric; `null` if unmeasured. |
| `label` | `string` | — | Required | `"ACTIVE"`, `"DROWSY"`, `"SLEEPING"`, `"AMBIGUOUS"` | Ground-truth observed driver fatigue class name from protocol annotation. | Discarded from supervised training set if `"AMBIGUOUS"`. |
| `label_id` | `int` | — | Required | `0, 1, 2, -1` | Integer encoding matching canonical class ordering. `-1` denotes invalid/ambiguous. | Must strictly equal mapping of `label`. |

### 3.2 Null and Missing Value Semantics
1. **Valid Zero vs. Missing Data**:
   * A closed eye produces an EAR approaching `0.0`. This is a **valid numerical measurement**.
   * A frame where the face is not detected or landmark confidence fails must record `landmarks_valid = False` and set `ear_left`, `ear_right`, `ear_avg`, and `is_eye_closed` to `null` (or IEEE `NaN`).
   * **Invariant**: Missing landmark measurements must **never** be silently imputed as `0.0` during data collection, because `0.0` indicates complete eye closure.
2. **Session Storage and Metadata Convention**:
   Each recorded session produces a self-contained directory under `data/recordings/{session_id}/`:
   ```text
   data/recordings/{session_id}/
       samples.jsonl
       session_meta.json
   ```
   * `samples.jsonl`: Tabular records stored as JSON Lines adhering strictly to the schema above.
   * `session_meta.json`: Top-level metadata recording:
     * `session_id` and stable anonymized `subject_id`.
     * Wall-clock capture start (ISO 8601 UTC timestamp).
     * Camera hardware and capture resolution/target FPS.
     * `is_eye_closed` configuration version and EAR threshold (calibrated in D3).
     * `perclos` configuration (trailing window duration in seconds or frame count, closure criterion).
     * Protocol notes and ambient environment description.

---

## 4. M1 $\rightarrow$ M2 Feature Contract

This section defines the observable semantic interface that M1 (Vision Runtime) provides to M2 (Machine Learning). M1 can implement its feature extraction components independently using this specification.

### 4.1 Feature Family Scope
The M1-to-M2 feature contract is strictly restricted to:
1. **Ocular Geometry**: Left, right, and average Eye Aspect Ratio (EAR).
2. **Eyelid Kinetics and Temporal Blink Features**: Instantaneous derived closure flag, cumulative blink counter, current/recent blink duration, and trailing PERCLOS.
3. **Capture Timing & Quality Indicators**: Frame timestamps, frame deltas, face detection flag, landmark validity flag.

*Note: Unrelated feature families—such as mouth aspect ratio (yawn detection), head pose angles (pitch/yaw/roll), EEG signals, or vehicle telemetry—are outside the current feature contract and must not be introduced without formal cross-milestone revisions.*

### 4.2 Observable Interface Contract

```
M1 Frame Output (per captured frame)
├── Metadata & Validity
│   ├── timestamp_ms: float (monotonic elapsed ms from session start)
│   ├── frame_delta_ms: float or null (null for frame 0, > 0.0 thereafter)
│   ├── face_detected: bool
│   └── landmarks_valid: bool
└── Extracted Ocular Features (valid only when landmarks_valid == True)
    ├── ear_left: float [0.0, 0.60] or null
    ├── ear_right: float [0.0, 0.60] or null
    ├── ear_avg: float [0.0, 0.60] or null
    ├── is_eye_closed: bool or null (derived from configured threshold)
    ├── blink_count: int [0, inf)
    ├── blink_duration_ms: float [0.0, inf)
    └── perclos: float [0.0, 1.0] or null (trailing window only)
```

### 4.3 Signal Degradation and Observable Behavior
* When face tracking is lost or landmark confidence fails:
  * M1 must set `landmarks_valid = False` (and `face_detected = False` if face is lost).
  * Instantaneous landmark-derived features (`ear_left`, `ear_right`, `ear_avg`, `is_eye_closed`) are invalid and must be emitted as `null` / `NaN`.
  * **No Synthetic Generation**: No synthetic blinks or synthetic feature values may be generated from missing observations.
  * **Internal Tracking Ownership**: M1 owns its internal tracking strategy, filtering methods, and state machines. M2 dictates only the observable interface contract above.

### 4.4 PERCLOS and Derived Closure Determinism
* **Trailing Window Requirement**: PERCLOS must be computed strictly over a trailing (causal) temporal history. Centered or future-looking windows are strictly prohibited to prevent lookahead leakage in online processing.
* **Dataset Release Consistency**: The exact trailing window length and closure criteria for PERCLOS may be configured, but the configuration must remain invariant across all sessions in a given dataset release and must be explicitly recorded in `session_meta.json`.
* **Derived Eye Closure Flag**: `is_eye_closed` is a derived convenience flag generated by comparing `ear_avg` to the configured threshold. It is not ground truth. The threshold will be calibrated empirically in D3 from real recorded data.

---

## 5. M2 $\rightarrow$ M1 Inference Contract

This section defines the semantic output payload contract that M2's trained model / inference adapter will publish to M1.

### 5.1 Inference Payload Specification

When M1 supplies a feature sequence to the M2 inference adapter, M2 returns a structured payload:

```json
{
  "predicted_class": "ACTIVE",
  "class_id": 0,
  "probabilities": {
    "ACTIVE": 0.85,
    "DROWSY": 0.10,
    "SLEEPING": 0.05
  },
  "confidence": 0.85,
  "status": "STATUS_OK",
  "latency_ms": "<measured_latency_ms>"
}
```

*Note: Numerical probability and latency values in the example above are illustrative schema placeholders. No benchmark performance or latency claims are made in this D1 document.*

### 5.2 Payload Field Definitions

| Field | Type | Description |
|:---|:---:|:---|
| `predicted_class` | `string` | Top-1 predicted class name: `"ACTIVE"`, `"DROWSY"`, or `"SLEEPING"`. |
| `class_id` | `int` | Canonical class index (`0`, `1`, or `2`). Must strictly match `predicted_class`. |
| `probabilities` | `map[string, float]` | Softmax probability distribution over canonical classes. Sum of probabilities equals $1.0 \pm 10^{-5}$. |
| `confidence` | `float` | Top-1 confidence score: $\max(probabilities)$, bounded in $[0.0, 1.0]$. |
| `status` | `string` | Execution status indicator describing M2's result (see status table below). |
| `latency_ms` | `float` | Wall-clock execution time for feature preprocessing and neural inference in milliseconds, populated at runtime. |

### 5.3 M2 Inference Status Semantics

The `status` field communicates the validity of M2's output:

| Status Code | Meaning (M2 Scope) |
|:---|:---|
| `STATUS_OK` | Inference succeeded; M2 produced a valid prediction and probability distribution. |
| `STATUS_INSUFFICIENT_HISTORY` | Temporal feature buffer is filling (warm-up phase); fewer frames than required window length $T$. |
| `STATUS_INVALID_INPUT` | Input features are invalid or missing across the window (e.g., face occluded or landmarks invalid). |
| `STATUS_ERROR` | Internal exception or invalid tensor conversion; M2 could not produce a valid result. |

*M1 Runtime Policy Separation: Downstream runtime behavior—including UI notifications, alert thresholds, auditory warnings, and fallback strategies—is strictly designed and owned by M1. M2 reports only the inference outcome.*

### 5.4 Deferral of Tensor Shapes and Windowing to D4
* **D1 Non-Locking Rule**: D1 does **not** lock the tensor dimension, sliding window length $T$, stride $S$, or feature normalization parameters.
* **D4 Responsibility**: M2 Day 4 (`feat/pytorch-dataset`) owns and will explicitly lock:
  1. Window length $T$ (number of timesteps).
  2. Input tensor shape (e.g., `(batch_size, sequence_length, feature_dim)`: `(B, T, D)`).
  3. Preprocessing transformation (feature scaling, mean/std normalization vectors, and imputation rules for missing frames).

---

## 6. Contract Invariants (D2 through D6)

The following invariants are non-negotiable architectural constraints across all M2 deliverables:

1. **Class Order Invariance**: Canonical class names and zero-indexed ordering (`0: ACTIVE`, `1: DROWSY`, `2: SLEEPING`) must never change, be reordered, or be partially omitted.
2. **Session-Relative Monotonicity**: Timestamps (`timestamp_ms`) must measure elapsed milliseconds from session start ($t_0 = 0.0$) and be strictly monotonically increasing.
3. **First-Frame Semantics**: Frame 0 must have `frame_index = 0`, `timestamp_ms = 0.0`, `frame_delta_ms = null`, and `blink_count = 0`. For all subsequent frames, `frame_delta_ms` must be strictly $> 0.0$.
4. **Frame Index Monotonicity and Gap Semantics**: `frame_index` must start at 0 for the first frame and strictly increase ($i_k > i_{k-1}$) for subsequent frames. Gaps (e.g., $0 \rightarrow 5$) are permitted to faithfully preserve dropped or skipped frames in upstream capture, but duplicate and backward frame indices are strictly prohibited.
5. **Subject Identity and Generalization Semantics**:
   * Anonymized stable identifiers (`subject_001`, `subject_002`) must persist uncorrupted into dataset records.
   * Subject-level split = evaluation of cross-subject generalization.
   * Session-level split on a single subject = evaluation of session generalization only.
   * A single-subject dataset must never be described as demonstrating cross-subject generalization.
6. **Ground Truth Independence**: Dataset labels represent verified observed human states from the recording protocol. Heuristic threshold rules and model predictions must never define or overwrite ground truth.
7. **Zero Distinction**: A valid zero (eyes closed) must remain distinguishable from a missing signal (`landmarks_valid = False`). Missing values must never be stored as numeric zero.
8. **Causal PERCLOS**: PERCLOS must remain a trailing (causal) measurement. Its definition must not silently vary across sessions within a dataset release.
9. **Deterministic Preprocessing**: Preprocessing parameters (e.g., mean, variance, imputation constants) must be fit strictly on training partitions and frozen before evaluation.

---

## 7. Requirements, Decisions, Assumptions, and Deferred Items

### Requirements
* Canonical class list: `ACTIVE` (0), `DROWSY` (1), `SLEEPING` (2).
* Ground truth established strictly by protocol annotation, independent of numerical heuristic thresholds.
* Recording schema requires session-relative `timestamp_ms` and `null` for frame 0 `frame_delta_ms`.
* Anonymized stable subject identifiers (`subject_001`) required without PII.
* M1 feature set restricted to ocular geometry, blink kinetics, and timing indicators.
* Missing landmarks represented as `null`/`NaN`, not `0.0`.
* PERCLOS is strictly trailing; derived `is_eye_closed` threshold version logged in metadata.

### Design Decisions
* Tabular flat format (CSV/JSONL) selected for D2 recording with companion `session_meta.json`.
* Multi-class classification formulation with explicit probability distributions.
* M2 inference status decoupled from M1 runtime/alert policy.

### Assumptions
* M1 captures video at a target rate of 25–30 FPS with reasonable lighting and frontal face orientation.
* A single driver face is dominant in the field of view.

### Deferred Decisions
* Exact window length $T$, stride $S$, and input tensor shape `(B, T, D)` $\rightarrow$ **Deferred to M2 Day 4**.
* Specific landmark detector selection and coordinate topology $\rightarrow$ **Owned by M1**.
* Feature normalization scaling constants $\rightarrow$ **Deferred to M2 Day 4**.
* Empirical EAR and PERCLOS baseline thresholds $\rightarrow$ **Calibrated in M2 Day 3**.
* Live runtime integration into capture loop $\rightarrow$ **Owned by M1**.
