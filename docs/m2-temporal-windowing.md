# Vigil M2 Temporal Windowing & Dataset Contract

## 1. Overview and Purpose

This document defines the temporal feature representation, sliding window extraction contract, train-only normalization protocol, and PyTorch dataset API implemented for **Milestone 2 Day 4 (M2 D4)** of the Vigil Driver Drowsiness Detection system.

The temporal windowing pipeline transforms sequentially recorded `SampleRecord` data streams into fixed-shape multidimensional tensors suitable for sequence-based fatigue classification models in M2 Day 5 and Day 6.

---

## 2. Locked Window Settings & Tensor Contract

### 2.1 Window Dimensions

* **Window Length ($W$)**: Exactly **60 recorded samples**.
* **Stride ($S$)**: Exactly **15 recorded samples**.
* **Tensor Shape**: `(60, 8)` with `torch.float32`.
* **Target Label**: Canonical integer scalar (`0: ACTIVE`, `1: DROWSY`, `2: SLEEPING`) with `torch.long`.
* **Target Alignment**: The ground-truth class of the **final sample ($t = 59$)** in the window, representing the driver fatigue state at the culmination of the temporal observation window.

> [!NOTE]
> **Physical Duration Disclaimer**:
> In a continuous 30 FPS video feed, 60 samples span approximately 2.0 seconds ($60 \times 33.3\text{ ms} = 2000\text{ ms}$) and a stride of 15 samples advances by 0.5 seconds ($500\text{ ms}$). However, because camera frame rates fluctuate in real-world environments, the contract is locked strictly to **sample count ($W = 60$)** rather than an assumed fixed physical duration.

---

## 3. Feature Channel Order & Mask Semantics

Every temporal window contains an exact 8-channel feature vector per timestep:

| Index | Feature Name | Dtype | Range | Description |
|---|---|---|---|---|
| `0` | `ear_avg` | `float32` | $\mathbb{R}$ (standardized) | Continuous eye aspect ratio; standardized on training statistics. |
| `1` | `is_eye_closed` | `float32` | $\{0.0, 1.0\}$ | Binary instantaneous eye-closure indicator supplied directly by M1. |
| `2` | `blink_duration_ms`| `float32` | $\mathbb{R}$ (standardized) | Continuous single blink duration; standardized on training statistics. |
| `3` | `perclos` | `float32` | $\mathbb{R}$ (standardized) | Trailing PERCLOS ratio; standardized on training statistics. |
| `4` | `ear_avg_valid` | `float32` | $\{0.0, 1.0\}$ | Binary mask: `1.0` if `landmarks_valid` and `ear_avg` finite; else `0.0`. |
| `5` | `is_eye_closed_valid` | `float32` | $\{0.0, 1.0\}$ | Binary mask: `1.0` if `landmarks_valid` and `is_eye_closed` observed; else `0.0`. |
| `6` | `blink_duration_valid` | `float32` | $\{0.0, 1.0\}$ | Binary mask: `1.0` if `blink_duration_ms` observed; else `0.0`. |
| `7` | `perclos_valid` | `float32` | $\{0.0, 1.0\}$ | Binary mask: `1.0` if `perclos` observed; else `0.0`. |

### 3.1 Validity Mask and Imputation Policy

* **No NaN or Infinity**: Tensors are strictly validated to prevent `NaN` or infinite values.
* **Distinguishing Valid Zero from Missing Observation**:
  * An observed `perclos = 0.0` is valid data: channel 3 is normalized, channel 7 (`perclos_valid`) is `1.0`.
  * An unobserved `perclos = None` is missing data: channel 3 is imputed as `0.0` (representing the training mean), channel 7 (`perclos_valid`) is `0.0`.
* The inclusion of explicit validity masks allows neural sequence models (D5/D6) to distinguish between an attentive driver with zero eye closure and a sensor dropout with missing landmark tracking.

---

## 4. Window and Session Integrity

1. **Strict Session & Subject Boundaries**:
   * Sliding windows **must never cross session or subject boundaries**.
   * Window generation operates strictly on continuous records within a single `session_id`.
   * **One Subject Per Session Invariant**: All records within a `session_id` must have the same `subject_id`. Mixed-subject sessions are rejected prior to splitting or windowing to prevent cross-partition session contamination.
2. **Temporal Sequence Validation**:
   * Records within each session are ordered strictly by `frame_index`.
   * **Duplicate Frame Detection**: Any duplicate `frame_index` within a session triggers an immediate `ValueError`.
   * **Monotonic Timestamps**: Timestamps must strictly increase with frame index. Non-monotonic or reversed timestamps are rejected.
   * **Preservation of Valid Gaps**: Valid frame drops ($\Delta_{\text{frame}} > 1$) and capture pauses ($\Delta_t > 200.0\text{ ms}$) trigger clean segment cuts without error.
3. **Segmentation on Discontinuities**:
   * Continuous segments are split whenever:
     * Frame index gap $\Delta_{\text{frame}} > 1$ (dropped frames), OR
     * Timestamp gap $\Delta_t > 200.0\text{ ms}$ (capture stalls or pauses).
   * Observations are **never invented or interpolated** across gaps.
   * If a segment has fewer than 60 samples, zero windows are extracted from that segment.
4. **Ambiguous Label Exclusion**:
   * Any sliding window that overlaps an `AMBIGUOUS` annotation (`label_id = -1`) anywhere within its 60-frame span is **immediately discarded** from supervised training and evaluation sets.
5. **Contract Enforcement**:
   * `WindowConfig` enforces locked $W = 60$ and $S = 15$.
   * `WindowSample` and `FatigueWindowDataset` validate exactly 60 timesteps, 8 feature channels, finite values, and canonical targets ($0, 1, 2$).

---

## 5. Train-Only Normalization Protocol

Feature normalization follows rigorous data hygiene to prevent forward data leakage:

* **Training Set Fitting Only**: Normalizer parameters (mean $\mu$ and standard deviation $\sigma$) are fitted **exclusively on valid continuous observations in the Training partition**.
* **Pre-Window Fitting**: Statistics are computed across raw training sample records before windowing, avoiding artificial weighting from overlapping sliding windows.
* **Frozen Transformation on Validation**: The validation partition is transformed using the frozen training statistics. Validation statistics are never computed or used.
* **Non-Continuous Invariant**: Binary indicators (`is_eye_closed`) and validity masks are never normalized.
* **Deterministic Fallbacks**: Constant features ($\sigma = 0$) default to $\sigma = 1.0$. All-missing features default to $\mu = 0.0, \sigma = 1.0$.

---

## 6. Leakage-Safe Grouped Splitting

Data splitting is executed before window extraction via `split_records_by_group`:

1. **Split Ratio Invariant**:
   * `train_ratio` must be a finite float strictly in $(0.0, 1.0)$. Out-of-bounds, infinite, or non-numeric ratios are rejected.
2. **Primary: Subject-Level Grouping (`subject_id`)**:
   * When multiple subjects exist, all sessions for a given subject are assigned exclusively to either Training or Validation.
   * Prevents models from memorizing subject-specific facial features or eye morphology.
3. **Fallback: Session-Level Grouping (`session_id`)**:
   * When only a single subject exists across the dataset, the system falls back to session-level grouping.
   * Cross-subject generalization cannot be claimed in single-subject datasets; this limitation is explicitly reported.
4. **Forbidden Practices**:
   * Random row-level or window-level splitting is strictly prohibited.
5. **Insufficient Groups**:
   * If only a single session exists, manufacturing a validation partition is rejected. The system reports that an independent evaluation partition is unavailable.

---

## 7. Python & PyTorch API

```python
from vigil.ml.dataset import (
    FatigueWindowDataset,
    FeatureNormalizer,
    WindowConfig,
    split_records_by_group,
)

# 1. Split raw records without leakage
train_recs, val_recs, split_info = split_records_by_group(all_records, train_ratio=0.75)

# 2. Fit normalizer on training records only
normalizer = FeatureNormalizer().fit(train_recs)

# 3. Create deterministic PyTorch datasets
train_dataset = FatigueWindowDataset.from_records(train_recs, normalizer=normalizer)
val_dataset = FatigueWindowDataset.from_records(val_recs, normalizer=normalizer)

# 4. Standard PyTorch index access
features, target = train_dataset[0]
# features.shape -> torch.Size([60, 8]), dtype torch.float32
# target -> tensor(0, dtype=torch.int64)
```
