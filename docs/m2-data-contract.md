# Vigil M2 Data Contract

## 1. Overview and Scope

This document specifies the canonical data contract for **Milestone 2 (M2: Machine Learning, Deep Learning, and Evaluation)** of the Vigil Driver Drowsiness Detection system. It defines:

1. The canonical fatigue state labels and ground-truth semantics.
2. The structured training record schema implemented by the D2 data recorder.
3. The upstream feature contract consumed from M1 (Vision and Runtime).
4. The downstream inference contract published to M1 for real-time integration.
5. Invariants and architectural boundaries that must remain stable across M2 Day 2 through Day 6.

### System Boundaries and Responsibility
* **M1 Ownership**: Video capture, facial/eye landmark extraction, instantaneous eye/blink feature calculation, runtime loop, and alert UI.
* **M2 Ownership**: Data schema, training data recorder, rule-based baseline, PyTorch dataset/loader, model training, evaluation metrics, and exported model artifacts.
* **Operational Notice**: Vigil is a prototype research and engineering system developed for evaluation and educational purposes. It is **not** a certified automotive driver-safety system or medical diagnostic device.

---

## 2. Canonical Fatigue Labels

### 2.1 Class Definitions and Ordering
M2 defines exactly three mutually exclusive fatigue classes. The canonical ordering and zero-indexed integer encodings are permanent and must not drift across any dataset, baseline, model architecture, or evaluation metric:

| Index | Canonical Class | Semantic Definition | Observable Physiological Indicators |
|:---:|:---|:---|:---|
| `0` | `ACTIVE` | The driver is alert, visually attentive, and exhibiting normal physiological ocular behavior. | Eyes fully open; regular spontaneous blink rate (typically 10–25 blinks/min); rapid eyelid closure and reopening duration (< 250–300 ms); stable facial pose; no drooping eyelids. |
| `1` | `DROWSY` | The driver exhibits noticeable signs of physiological fatigue, sluggish eyelid kinetics, or reduced alertness. | Prolonged blinks (eyelid closure > 300–500 ms); slow or incomplete eyelid reopening; reduced average Eye Aspect Ratio (EAR) relative to alert baseline; elevated cumulative eye closure (PERCLOS); sluggish gaze response. |
| `2` | `SLEEPING` | The driver is in an acute state of severe inattention, microsleep, or sleep. | Sustained complete or near-complete eye closure for $\ge 1.5$–$2.0$ consecutive seconds; loss of active visual engagement with driving environment; head droop or fixed closed-eye state. |

### 2.2 Ground Truth vs. Model Predictions
* **Observed State Only**: The label recorded in training data represents the *verified ground-truth state* of the subject at capture time (determined by experimental protocol or post-recording human annotation).
* **No Synthetic Heuristic Labels**: A model's output or a heuristic rule's output must never be recorded as ground truth in supervised training corpora.

### 2.3 Handling Ambiguity and Edge Cases
* **State Exclusivity**: A subject cannot simultaneously occupy two states. In transitional phases (e.g., transition from `ACTIVE` to `DROWSY`), labels must reflect the dominant observable behavior over the immediate evaluation interval.
* **Uncertain / Degraded States**: When a label cannot be assigned with confidence (e.g., driver rubbing eyes, squinting due to sudden glare, yawning with eyes squeezed, partial occlusion, or annotator uncertainty), the data recorder or annotator must mark the sample as `AMBIGUOUS` (encoded as `label_id = -1`) or discard the interval entirely.
* **Supervised Training Filter**: The downstream PyTorch training pipeline (D4/D5) must strictly reject or filter out any sample where `label_id == -1` or `label == "AMBIGUOUS"`. Unlabeled or ambiguous data must never be used for supervised loss calculation or benchmark evaluation.

---

## 3. Training Record Schema

The D2 data recorder must emit structured, tabular records. The schema is designed for flat serialization (CSV or JSON Lines) accompanied by session-level metadata (`session_meta.json`).

### 3.1 Field Specifications

| Field Name | Type | Unit | Requirement | Valid / Expected Range | Meaning | Missing / Invalid Semantics |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| `sample_id` | `string` | — | Required | Unique string (`{session_id}_f{frame_index}`) | Globally unique sample identifier. | Cannot be null or empty. |
| `session_id` | `string` | — | Required | Unique session slug (e.g., `s_20261006_1530_sub01`) | Identifies the continuous recording session. Critical for grouped splitting. | Cannot be null or empty. |
| `subject_id` | `string` | — | Required | Alphanumeric identifier or `"unknown"` | Identifies the recorded individual. Critical for preventing identity leakage across splits. | Defaults to `"unknown"` if anonymized/unspecified. |
| `frame_index` | `int` | frames | Required | $[0, \infty)$ | Zero-indexed sequential frame counter within the session. Monotonically increasing. | Cannot be negative; cannot decrement. |
| `timestamp_ms` | `float` | ms | Required | $[0.0, \infty)$ | Monotonic timestamp elapsed since recording start (or epoch ms). | Must be strictly increasing ($t_i > t_{i-1}$). |
| `frame_delta_ms` | `float` | ms | Required | $(0.0, 1000.0]$ | Time elapsed since the previous captured frame ($t_i - t_{i-1}$). | For frame 0, equals `0.0`. If $> 1000.0$ ms, indicates recording stall/gap. |
| `face_detected` | `bool` | — | Required | `True`, `False` | Flag indicating whether a human face was localized in the frame. | If `False`, all downstream facial features are invalid. |
| `landmarks_valid` | `bool` | — | Required | `True`, `False` | Flag indicating whether eye landmark coordinates passed confidence and geometric checks. | If `False`, eye features are invalid. |
| `ear_left` | `float` | ratio | Optional | $[0.0, 0.60]$ | Eye Aspect Ratio for the left eye calculated from canonical landmarks. | `null` / `NaN` if `landmarks_valid == False`. Must not be coerced to `0.0`. |
| `ear_right` | `float` | ratio | Optional | $[0.0, 0.60]$ | Eye Aspect Ratio for the right eye calculated from canonical landmarks. | `null` / `NaN` if `landmarks_valid == False`. Must not be coerced to `0.0`. |
| `ear_avg` | `float` | ratio | Optional | $[0.0, 0.60]$ | Mean Eye Aspect Ratio: $(ear_{left} + ear_{right}) / 2.0$. Primary instantaneous metric. | `null` / `NaN` if `landmarks_valid == False`. |
| `is_eye_closed` | `bool` | — | Optional | `True`, `False` | Instantaneous binary indicator whether `ear_avg` is below eye closure threshold. | `null` if `landmarks_valid == False`. |
| `blink_count` | `int` | count | Required | $[0, \infty)$ | Cumulative completed blinks recorded in the current session up to this frame. | Starts at 0; monotonically non-decreasing. |
| `blink_duration_ms` | `float` | ms | Required | $[0.0, 10000.0]$ | Duration of the currently active or most recently completed eye closure event. | `0.0` if eyes are continuously open. |
| `perclos` | `float` | ratio | Optional | $[0.0, 1.0]$ | Percentage of Eye Closure over rolling temporal window (e.g., past 60s or past $N$ frames). | `null` during initial warm-up buffer; $[0.0, 1.0]$ when valid. |
| `fps` | `float` | fps | Optional | $(0.0, 120.0]$ | Instantaneous or smoothed capture frame rate. | Informative metric; `null` if unmeasured. |
| `label` | `string` | — | Required | `"ACTIVE"`, `"DROWSY"`, `"SLEEPING"`, `"AMBIGUOUS"` | Ground-truth observed driver fatigue class name. | Discarded from supervised set if `"AMBIGUOUS"`. |
| `label_id` | `int` | — | Required | `0, 1, 2, -1` | Integer encoding matching canonical class ordering. `-1` denotes invalid/ambiguous. | Must strictly equal mapping of `label`. |

### 3.2 Null and Missing Value Semantics
1. **Valid Zero vs. Missing Data**:
   * A closed eye has an EAR near `0.0` (or below ~`0.15`). This is a **valid numerical measurement**.
   * A frame where the face is not detected or occluded must record `landmarks_valid = False` and set `ear_left`, `ear_right`, and `ear_avg` to `null` (or IEEE `NaN` in floating-point representations).
   * **Invariant**: Missing landmark measurements must **never** be silently imputed as `0.0` during data collection, because `0.0` indicates complete eye closure (extreme fatigue/sleep).
2. **Session Storage Convention**:
   Each recorded session $k$ produces a self-contained directory under `data/recordings/{session_id}/`:
   * `samples.csv` (or `samples.jsonl`): Tabular rows satisfying the schema above.
   * `session_meta.json`: Top-level metadata including recording date, operator, subject ID, camera hardware info, baseline alert EAR, and environmental notes.

---

## 4. M1 $\rightarrow$ M2 Feature Contract

This section defines the semantic interface that M1 (Vision Runtime) provides to M2 (Machine Learning). M1 can implement its feature extraction components independently using this specification.

### 4.1 Feature Family Scope
The M1-to-M2 feature contract is strictly restricted to:
1. **Ocular Geometry**: Left, right, and average Eye Aspect Ratio (EAR).
2. **Eyelid Kinetics and Temporal Blink Features**: Instantaneous closure flag, cumulative blink counter, current/recent blink duration, and rolling PERCLOS.
3. **Capture Timing & Quality Indicators**: Frame timestamps, frame deltas, face detection flag, landmark validity flag.

*Note: Unrelated feature families—such as mouth aspect ratio (yawn detection), head pose angles (pitch/yaw/roll), EEG signals, or steering telemetry—are outside the current feature contract and must not be injected without cross-milestone contract revisions.*

### 4.2 Feature Delivery Contract

```
M1 Frame Output (per captured frame)
├── Metadata & Validity
│   ├── timestamp_ms: float (strictly monotonic)
│   ├── frame_delta_ms: float (> 0.0)
│   ├── face_detected: bool
│   └── landmarks_valid: bool
└── Extracted Ocular Features (valid only when landmarks_valid == True)
    ├── ear_left: float [0.0, 0.60]
    ├── ear_right: float [0.0, 0.60]
    ├── ear_avg: float [0.0, 0.60]
    ├── is_eye_closed: bool
    ├── blink_count: int [0, inf)
    ├── blink_duration_ms: float [0.0, inf)
    └── perclos: float [0.0, 1.0]
```

### 4.3 Signal Degradation Behavior
* When face tracking is lost or landmark confidence fails:
  * M1 must set `face_detected = False` and `landmarks_valid = False`.
  * M1 must emit `null` / `NaN` for instantaneous EAR features.
  * M1 must retain the last known `blink_count` and freeze closure timers rather than emitting spurious blinks.

---

## 5. M2 $\rightarrow$ M1 Inference Contract

This section defines the output payload contract that M2's trained model / inference adapter will publish to M1 for runtime UI overlay and alert triggering.

### 5.1 Inference Payload Specification

When M1 supplies a feature sequence to the M2 inference adapter, M2 returns a structured payload:

```json
{
  "predicted_class": "ACTIVE",
  "class_id": 0,
  "probabilities": {
    "ACTIVE": 0.942,
    "DROWSY": 0.048,
    "SLEEPING": 0.010
  },
  "confidence": 0.942,
  "status": "STATUS_OK",
  "latency_ms": 1.45
}
```

### 5.2 Payload Field Definitions

| Field | Type | Description |
|:---|:---:|:---|
| `predicted_class` | `string` | Top-1 predicted class name: `"ACTIVE"`, `"DROWSY"`, or `"SLEEPING"`. |
| `class_id` | `int` | Canonical class index (`0`, `1`, or `2`). Must strictly match `predicted_class`. |
| `probabilities` | `map[string, float]` | Softmax probability distribution over canonical classes. Sum of probabilities must equal $1.0 \pm 10^{-5}$. |
| `confidence` | `float` | Top-1 confidence score $\max(P)$, bounded in $[0.0, 1.0]$. |
| `status` | `string` | Execution status indicator (see status table below). |
| `latency_ms` | `float` | Wall-clock execution time for feature preprocessing and neural inference in milliseconds. |

### 5.3 Status Code Semantics

| Status Code | Meaning | M1 Runtime Action |
|:---|:---|:---|
| `STATUS_OK` | Inference succeeded; prediction is valid and actionable. | Update UI; feed alert controller with prediction. |
| `STATUS_INSUFFICIENT_HISTORY` | Temporal window buffer is filling (warm-up phase); insufficient frames to produce inference. | Display "Buffering / Initializing"; suppress alerts. |
| `STATUS_INVALID_INPUT` | Face missing or landmark validity failed across majority of the temporal window. | Display "Face Not Detected"; suspend fatigue classification. |
| `STATUS_ERROR` | Internal exception or invalid tensor conversion. | Log error; gracefully fallback to rule-based baseline or safe standby. |

### 5.4 Deferral of Tensor Shapes and Windowing to D4
* **D1 Non-Locking Rule**: D1 does **not** lock the tensor dimension, sliding window length $T$, stride $S$, or feature normalization parameters.
* **D4 Responsibility**: M2 Day 4 (`feat/pytorch-dataset`) owns and will explicitly lock:
  1. Window length $T$ (e.g., $T = 30$ frames at 30 FPS = 1.0s or $T = 60$ frames = 2.0s).
  2. Input tensor shape (e.g., `(batch_size, sequence_length, feature_dim)`: `(B, T, D)`).
  3. Preprocessing transformation (feature scaling, mean/std normalization vectors, and imputation rules for missing frames).

---

## 6. Contract Invariants (D2 through D6)

The following invariants are non-negotiable architectural constraints across all M2 deliverables:

1. **Class Order Invariance**: The canonical class names and zero-indexed ordering (`0: ACTIVE`, `1: DROWSY`, `2: SLEEPING`) must never change, be reordered, or be partially omitted.
2. **Strict Monotonicity**: Timestamps (`timestamp_ms`) and frame indices (`frame_index`) within a session must be strictly monotonically increasing.
3. **Identity Preservation**: Session identifiers (`session_id`) and subject identifiers (`subject_id`) must persist uncorrupted from recording into dataset files to ensure leakage-safe evaluation.
4. **Ground Truth Purity**: Dataset labels represent verified observed states. Model predictions and rule baseline outputs must never overwrite or masquerade as ground truth.
5. **Zero Distinction**: A valid zero (e.g., eyes closed) must remain distinguishable from a missing signal (landmark failure). Missing values must never be stored as numeric zero.
6. **Deterministic Preprocessing**: Preprocessing parameters (e.g., mean, variance, imputation constants) must be fit strictly on training partitions and frozen before evaluation.

---

## 7. Requirements, Decisions, Assumptions, and Deferred Items

### Requirements
* Canonical class list: `ACTIVE` (0), `DROWSY` (1), `SLEEPING` (2).
* Recording schema must support grouped leakage-free splitting using session and subject IDs.
* M1 feature set must be restricted to ocular geometry, blink kinetics, and timing indicators.
* Missing landmarks must be represented as `null`/`NaN`, not `0.0`.

### Design Decisions
* Tabular flat format (CSV/JSONL) selected for D2 recording to maximize inspectability and eliminate heavy database dependencies.
* Multi-class classification formulation with explicit probability distributions rather than a single binary alert flag.
* Structured status codes (`STATUS_OK`, `STATUS_INSUFFICIENT_HISTORY`, etc.) defined to cleanly decouple vision capture state from classifier state.

### Assumptions
* M1 captures video at a target rate of 25–30 FPS with reasonable lighting and frontal face orientation.
* A single driver face is dominant in the field of view.

### Deferred Decisions
* Exact window length $T$, stride $S$, and input tensor shape `(B, T, D)` $\rightarrow$ **Deferred to M2 Day 4**.
* Specific landmark detector selection and coordinate topology $\rightarrow$ **Owned by M1**.
* Feature normalization scaling constants $\rightarrow$ **Deferred to M2 Day 4**.
* Real-time inference integration adapter implementation $\rightarrow$ **Deferred to M2 Day 6**.
