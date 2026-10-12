# Vigil M2 Model Export, Runtime Inference, and Evaluation Comparison

## 1. Overview and Purpose

This document specifies the artifact export bundle format, stateful streaming runtime inference engine, and head-to-head baseline evaluation protocol implemented for **Milestone 2 Day 6 (M2 D6)** of the Vigil Driver Drowsiness Detection system.

M2 D6 bridges offline temporal model training with production runtime readiness by delivering:
1. **Self-Contained Model Bundles**: Versioned `.pt` checkpoints encapsulating the trained GRU weights, architecture configuration, fitted `FeatureNormalizer` state, window configuration ($W=60, S=15$), and canonical class mappings.
2. **Stateful Streaming Inference (`TemporalInferenceEngine`)**: A real-time rolling-window inference component consuming M1 `SampleRecord` or `LiveInputFrame` objects, respecting warm-up and stride intervals, enforcing temporal discontinuity resets, and emitting contract-conforming `InferenceResult` payloads on every step.
3. **Reproducible Baseline Comparison (`compare_models`)**: A leakage-safe evaluation pipeline benchmarking the learned GRU sequence classifier against the D3 heuristic rule baseline across matched window endpoints and full frame sequences.

---

## 2. Model Export and Loading Specification

The export and loading contracts are implemented in `src/vigil/ml/models/export.py`.

### 2.1 Bundle Schema (`BUNDLE_SCHEMA_VERSION = "1.0.0"`)

An exported model bundle is a self-contained PyTorch serialization dictionary containing all components required for runtime execution without retraining:

```python
{
    "bundle_version": "1.0.0",
    "model_type": "GRUTemporalClassifier",
    "created_at_utc": "2026-10-10T16:00:00.000000+00:00",
    "model_config": {
        "input_dim": 8,
        "hidden_dim": 64,
        "num_layers": 1,
        "num_classes": 3,
        "dropout": 0.0,
        "bidirectional": False,
    },
    "state_dict": OrderedDict(...),  # Tensor weights cloned onto CPU
    "normalizer": {
        "means": {"ear_avg": 0.285, "blink_duration_ms": 142.0, "perclos": 0.110},
        "stds": {"ear_avg": 0.048, "blink_duration_ms": 48.5, "perclos": 0.075},
        "observed_counts": {"ear_avg": 1250, "blink_duration_ms": 1250, "perclos": 1250},
        "is_fitted": True,
    },
    "window_config": {
        "window_length": 60,
        "stride": 15,
        "max_frame_gap": 1,
        "max_time_gap_ms": 200.0,
    },
    "feature_channels": [
        "ear_avg",
        "is_eye_closed",
        "blink_duration_ms",
        "perclos",
        "ear_avg_valid",
        "is_eye_closed_valid",
        "blink_duration_valid",
        "perclos_valid",
    ],
    "class_labels": {
        0: "ACTIVE",
        1: "DROWSY",
        2: "SLEEPING",
    },
    "metadata": {
        "git_commit": "ce02bd7",
        "author": "M2",
        "notes": "Trained on synthetic fixtures for D6 contract validation",
    },
}
```

### 2.2 APIs

#### Exporting: `export_model_bundle`

```python
from vigil.ml.models.export import export_model_bundle

bundle_path = export_model_bundle(
    model=trained_gru,
    normalizer=fitted_normalizer,
    filepath="artifacts/models/gru_bundle.pt",
    metadata={"experiment_id": "exp_001"},
)
```

- Enforces that `normalizer.is_fitted == True`. Unfitted normalizers raise `ValueError`.
- Creates parent directories automatically.
- Detaches and clones model weights onto the CPU to guarantee cross-device loading portability.

#### Loading: `load_model_bundle` / `LoadedModelBundle.from_bundle`

```python
from vigil.ml.models.export import load_model_bundle, LoadedModelBundle

# Via top-level loader
bundle = load_model_bundle("artifacts/models/gru_bundle.pt", map_location="cpu")

# Or via classmethod
bundle = LoadedModelBundle.from_bundle("artifacts/models/gru_bundle.pt")

model = bundle.model              # GRUTemporalClassifier in eval mode
normalizer = bundle.normalizer    # Restored FeatureNormalizer with fitted parameters
window_config = bundle.window_config  # WindowConfig(window_length=60, stride=15, ...)
```

#### Validation & Compatibility Guards

During `load_model_bundle`, the following validations occur:
1. **File Existence**: Raises `FileNotFoundError` if the file does not exist.
2. **Bundle Schema**: Raises `TypeError` if the deserialized payload is not a dictionary.
3. **Required Sections**: Verifies presence of `bundle_version`, `model_config`, `state_dict`, `normalizer`, `window_config`, `feature_channels`, and `class_labels`. Missing sections raise `ValueError`.
4. **Version Compatibility**: Matches major version against `BUNDLE_SCHEMA_VERSION.split(".")[0]`. Incompatible versions raise `ValueError`.
5. **Feature Channel Alignment**: Confirms that channel names and order match `LOCKED_FEATURE_CHANNELS` exactly.
6. **Normalizer Integrity**: Requires `normalizer["is_fitted"] == True` with non-empty means and stds.
7. **Model Reconstruction**: Instantiates `GRUTemporalClassifier` with restored `model_config`, loads `state_dict`, and sets `model.eval()`.

---

## 3. Stateful Streaming Runtime Inference

The streaming engine is implemented in `src/vigil/ml/inference/engine.py`.

`TemporalInferenceEngine` accepts frame-by-frame M1 input in live capture order and emits structured `InferenceResult` predictions conforming to the published M2→M1 data contract.

It accepts two input types:
- **`SampleRecord`** — labelled training/test records (offline evaluation, `compare_models`).
- **`LiveInputFrame`** — unlabelled live camera frames (M1 production inference).

Missing ground-truth annotation on a `LiveInputFrame` is **never** treated as invalid sensor data.

### 3.1 Streaming Contracts and Mechanics

```mermaid
stateDiagram-v2
    [*] --> Idle: Initialize / reset()
    Idle --> WarmingUp: sample.validate() OK
    WarmingUp --> WarmingUp: buf_len < 60
    WarmingUp --> Predicted: buf_len == 60 (frame 59)
    Predicted --> StrideSkip: 1 <= steps_since_pred < 15
    StrideSkip --> StrideSkip: 1 <= steps_since_pred < 15
    StrideSkip --> Predicted: steps_since_pred == 15
    WarmingUp --> Resetting: Frame gap > 1 / Time stall > 200ms / New session_id
    StrideSkip --> Resetting: Frame gap > 1 / Time stall > 200ms / New session_id
    Predicted --> Resetting: Frame gap > 1 / Time stall > 200ms / New session_id
    Resetting --> WarmingUp: Buffer cleared (size=1)
```

1. **Schema Validation**: Every incoming `SampleRecord` or `LiveInputFrame` is validated against its strict schema. Structurally invalid inputs return `STATUS_INVALID_INPUT` with no prediction.
2. **Buffer Bounding**: An internal `deque(maxlen=60)` retains at most 60 timesteps.
3. **Warm-Up Phase**:
   - Frames 0 to 58 (first 59 samples): No prediction is emitted (`has_prediction=False`, `reason="buffer_warming_up"`).
   - Frame 59 (60th consecutive valid sample): First prediction is emitted (`has_prediction=True`, `reason="predicted"`).
4. **Stride Cadence ($S=15$)**:
   - Following any prediction, incoming samples roll into the buffer.
   - For steps 1 through 14 after a prediction: `has_prediction=False`, `reason="stride_skip"`.
   - On the 15th step: Next inference is executed on the updated 60-sample window (`has_prediction=True`, `reason="predicted"`).
5. **Session Boundary**:
   - Encountering a different `sample.session_id` automatically invokes `reset()` and initializes the buffer with the new record (`reason="session_reset"`).
6. **Temporal Discontinuity Protections**:
   - **Frame Drops** ($\Delta_{\text{frame}} > 1$): The rolling buffer is cleared, and warm-up restarts from the current frame (`reason="discontinuity_frame_gap"`).
   - **Capture Stalls** ($\Delta t > 200.0\text{ ms}$): The rolling buffer is cleared, and warm-up restarts (`reason="discontinuity_timing_gap"`).
   - A window is **never** silently constructed across a discontinuity.
7. **Monotonicity Enforcement**:
   - Duplicate frame indices ($\text{frame} \le \text{prev\_frame}$) or backward timestamps ($\text{timestamp} \le \text{prev\_timestamp}$) within the same session raise a `ValueError`.
8. **Missing Features & Landmark Imputation**:
   - Frames with missing landmarks (`landmarks_valid=False`) or missing continuous ocular metrics are imputed deterministically using the fitted `FeatureNormalizer` (`imputed_value = 0.0`), with validity masks set to `0.0`. The engine continues streaming without crashing.

### 3.2 Unlabeled Live-Inference Input: `LiveInputFrame`

For unannotated live camera frames (no ground-truth label), M1 must supply a `LiveInputFrame` instead of a `SampleRecord`.

```python
from vigil.ml.data.schema import LiveInputFrame

frame = LiveInputFrame(
    session_id="live_session_001",
    frame_index=42,
    timestamp_ms=1400.0,
    frame_delta_ms=33.3,
    face_detected=True,
    landmarks_valid=True,
    ear_left=0.28,
    ear_right=0.30,
    ear_avg=0.29,           # must equal (ear_left + ear_right) / 2 ± 1e-3
    is_eye_closed=False,
    blink_count=2,
    blink_duration_ms=0.0,
    perclos=0.05,
    fps=30.0,
)
```

**Ownership boundary:** `LiveInputFrame` is owned by M2. M1 is the exclusive caller; it must never fabricate `label`, `label_id`, or `subject_id` on live frames.

**Validation rules** mirror `SampleRecord` for every shared field:
- EAR fields: `[0.0, 0.60]`, consistency check `|ear_avg - (ear_left + ear_right) / 2| ≤ 1e-3`
- `frame_index=0` requires `blink_count=0`, `timestamp_ms=0.0`, `frame_delta_ms=None`
- FPS: `(0.0, 120.0]`
- Missing landmarks require all ocular fields to be `None`

### 3.3 Output Schema: `InferenceResult`

Each call to `engine.step(frame)` (or `engine.process_sample(frame)`) returns an immutable `InferenceResult`. Every response — including warm-up, stride-skip, and discontinuity responses — contains **all six required contract fields**.

#### 3.3.1 Required Contract Fields

| Field | Type | Description |
|---|---|---|
| `predicted_class` | `str \| None` | Top-1 predicted class (`'ACTIVE'`, `'DROWSY'`, `'SLEEPING'`) when `status == STATUS_OK`; `None` otherwise. |
| `class_id` | `int \| None` | Canonical class index (`0`, `1`, `2`) when `status == STATUS_OK`; `None` otherwise. |
| `probabilities` | `dict[str, float] \| None` | Softmax probability distribution across all 3 canonical classes when `status == STATUS_OK`; `None` otherwise. |
| `confidence` | `float \| None` | Max predicted class probability (`max(probabilities)`) when `status == STATUS_OK`; `None` otherwise. |
| `status` | `str` | One of the published status codes (see §3.3.2). Always present. |
| `latency_ms` | `float` | Wall-clock time for feature preprocessing and neural inference, in milliseconds. Measured from just before feature extraction to just after `torch.softmax` returns. Zero for non-prediction steps. Always finite and ≥ 0. |

#### 3.3.2 Published Status Codes

| Constant | Value | When emitted |
|---|---|---|
| `STATUS_OK` | `'STATUS_OK'` | Inference succeeded; valid prediction and probabilities produced. |
| `STATUS_INSUFFICIENT_HISTORY` | `'STATUS_INSUFFICIENT_HISTORY'` | Buffer is filling, between stride steps, or rebuilding after a session change or temporal discontinuity. |
| `STATUS_INVALID_INPUT` | `'STATUS_INVALID_INPUT'` | Frame is structurally invalid or the feature window is unusable. |
| `STATUS_ERROR` | `'STATUS_ERROR'` | Internal inference failure; no prediction produced. |

```python
from vigil.ml.inference import STATUS_OK, STATUS_INSUFFICIENT_HISTORY, STATUS_INVALID_INPUT, STATUS_ERROR
```

#### 3.3.3 Diagnostic Fields (supplemental)

| Field | Type | Description |
|---|---|---|
| `reason` | `str` | Internal status code: `'predicted'`, `'buffer_warming_up'`, `'stride_skip'`, `'session_reset'`, `'discontinuity_frame_gap'`, `'discontinuity_timing_gap'`, `'invalid_input'`, `'inference_error'`. |
| `session_id` | `str \| None` | Active session identifier. |
| `buffer_size` | `int` | Samples currently in the rolling buffer ($1 \le N \le 60$). |
| `start_frame_index` | `int \| None` | Frame index of the oldest sample in the active window. |
| `end_frame_index` | `int \| None` | Frame index of the newest sample in the active window. |
| `start_timestamp_ms` | `float \| None` | Timestamp of the oldest frame in the active window. |
| `end_timestamp_ms` | `float \| None` | Timestamp of the newest frame in the active window. |

#### 3.3.4 Backward-Compatibility Aliases

Sites that previously consumed `RuntimeInferenceResult` can use the following aliases without breaking:

| Old field | Maps to |
|---|---|
| `has_prediction` | `status == STATUS_OK` |
| `label` | `predicted_class` |
| `label_id` | `class_id` |

`RuntimeInferenceResult` is retained as an import alias for `InferenceResult`.

---

## 4. Evaluation and Baseline Comparison

The comparison pipeline is implemented in `src/vigil/ml/evaluation/comparison.py`.

### 4.1 Comparison Protocol (`compare_models`)

The evaluation compares the learned GRU sequence classifier against the D3 heuristic rule baseline under strictly leakage-free conditions:

```python
from vigil.ml.evaluation.comparison import compare_models

report = compare_models(
    records=dataset_records,
    model=bundle.model,
    normalizer=bundle.normalizer,
    baseline_classifier=RuleBasedBaseline(),
    train_ratio=0.75,
    seed=42,
    is_synthetic=False,
)

print(report.to_markdown())
```

### 4.2 Data Leakage Prevention Guarantees

1. **Pre-Window Grouped Splitting**: Records are partitioned into train and validation sets by `subject_id` (with fallback to `session_id`) **before** any window extraction occurs.
2. **Train-Only Normalization**: The `FeatureNormalizer` is fitted strictly on the training partition.
3. **Dual Baseline Benchmarking**:
   - **Matched Window Endpoints**: Evaluates the D3 rule baseline on the exact final-timestep frame ($t = 59$) of each valid 60-frame evaluation window. This yields a direct 1-to-1 comparison on identical decision events.
   - **Full Partition Evaluation**: Evaluates the D3 rule baseline frame-by-frame across all eligible validation records.
4. **Metrics**:
   - Overall Accuracy (on evaluated/covered samples)
   - Macro-averaged Precision, Recall, and F1 score
   - Prediction Coverage and Skipped/Abstained prediction tracking
   - $3 \times 3$ Confusion Matrix in canonical class order (`0: ACTIVE`, `1: DROWSY`, `2: SLEEPING`)

### 4.3 Production Directory Evaluation: `evaluate_recording_directory`

```python
from vigil.ml.evaluation.comparison import evaluate_recording_directory

report = evaluate_recording_directory(
    directory_path="data/recordings",
    bundle_path="artifacts/models/gru_bundle.pt",
)
```

- Discovers session `.jsonl` files recursively (`**/samples.jsonl` or `*.jsonl`).
- Returns `None` if the directory does not exist or contains no recordings.
- **Never fabricates synthetic numbers as real performance.**

---

## 5. Current Dataset Status & Limitations

> [!IMPORTANT]
> **Repository Audit Finding**: The repository currently contains no real recorded session datasets (directories `/data` and `/data/recordings` are intentionally empty and excluded from Git).
>
> All automated tests in `tests/ml` use controlled synthetic fixtures to verify tensor dimensions, window slicing, mathematical normalizer properties, and pipeline contracts.
>
> **No claim of real-world drowsiness detection accuracy is made at this milestone.** Real-world benchmark numbers can only be published once physical driver sessions have been recorded and labeled in accordance with M2 data-collection protocols.

---

## 6. Verification and Reproduction Commands

To reproduce the verification results:

```bash
# 1. Run all tests in the repository
pytest -q

# 2. Run all ML-specific unit and integration tests
pytest -q tests/ml

# 3. Verify linting and formatting compliance
ruff check src/vigil/ml tests/ml docs

# 4. Verify no whitespace or merge artifacts
git diff --check

# 5. Verify ordinary package imports function without PyTorch installed
python -c "import sys; sys.modules['torch'] = None; import vigil; import vigil.ml; print('Import succeeded without torch')"
```
