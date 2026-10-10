# Vigil M2 Temporal Sequence Classifier & Training Specification

## 1. Overview and Purpose

This document specifies the recurrent neural sequence classifier architecture, training pipeline, reproducibility controls, and evaluation protocol implemented for **Milestone 2 Day 5 (M2 D5)** of the Vigil Driver Drowsiness Detection system.

The temporal classifier consumes fixed $(B, 60, 8)$ feature tensors produced by the M2 D4 dataset pipeline and models multi-frame temporal dynamics (such as prolonged closure progressions and cumulative eyelid closure velocity) to categorize driver states into canonical classes:
* `ACTIVE = 0`
* `DROWSY = 1`
* `SLEEPING = 2`

---

## 2. Model Architecture: Compact GRU Sequence Classifier

The model is implemented in `src/vigil/ml/models/gru.py` as `GRUTemporalClassifier`.

### 2.1 Layer Specifications

1. **Input Tensor**:
   * Shape: `(batch_size, 60, 8)`
   * Dtype: `torch.float32`
   * Channel order: strictly preserves the 8 locked D4 channels (`ear_avg`, `is_eye_closed`, `blink_duration_ms`, `perclos`, and their respective validity masks).
2. **Recurrent Backbone**:
   * Type: Gated Recurrent Unit (`nn.GRU`)
   * Input dimension: `8`
   * Hidden dimension: configurable (default: `64`)
   * Number of layers: `1`
   * Directionality: strictly unidirectional (`bidirectional = False`), ensuring causal temporal inference that does not look into future frames.
   * Batch formatting: `batch_first = True`.
3. **Classification Head**:
   * Pooling / Temporal Selection: Representation of the final timestep ($t = 59$), aligned strictly with the ground-truth annotation at the end of the 60-frame window.
   * Projection: Fully connected layer `nn.Linear(hidden_dim, 3)`.
4. **Output Logits**:
   * Shape: `(batch_size, 3)`
   * Raw unnormalized logits suitable directly for `torch.nn.CrossEntropyLoss`. Softmax is **not** applied in `forward()`.
   * Normalized probabilities for inference are computed via `model.predict_proba(x)`.
   * Discrete class predictions ($0, 1, 2$) are obtained via `model.predict(x)`.

---

## 3. Reproducible Training Pipeline

The training API is exposed as `train_temporal_classifier` under `src/vigil/ml/training/`.

### 3.1 Pipeline Invariants & Data Integrity

1. **Pre-Window Grouped Splitting**:
   * Records are partitioned using `split_records_by_group` prior to window generation.
   * **Subject-First Priority**: When multiple subjects exist, all sessions for any subject remain strictly in a single partition.
   * **Single-Subject Fallback**: If only one subject exists, grouping falls back to `session_id`. Cross-subject generalization cannot be evaluated in this scenario.
2. **Train-Only Normalization**:
   * `FeatureNormalizer` is fitted strictly on valid continuous features in the **training partition records**.
   * Frozen training parameters ($\mu, \sigma$) standardize validation records.
   * Binary flags (`is_eye_closed`) and validity masks are never normalized.
3. **Independent Window Extraction**:
   * Training and validation windows are extracted separately after split assignment, eliminating cross-partition window overlap or frame contamination.
4. **Deterministic Seeding**:
   * `seed_everything(seed)` configures Python `random`, NumPy, PyTorch CPU/GPU, and DataLoader generators.
   * Training DataLoader shuffles minibatches deterministically via a seeded `torch.Generator`.
   * Validation DataLoader evaluates without shuffling.
5. **Loss & Optimization Defaults**:
   * Loss: Multi-class Cross-Entropy Loss (`torch.nn.CrossEntropyLoss()`).
   * Optimizer: Adam (`torch.optim.Adam`) with default learning rate $\eta = 10^{-3}$ and weight decay $10^{-4}$.
   * Minibatch size: `32`.
   * Epoch limit: `30`.
6. **Best Checkpoint Selection**:
   * Epoch-level validation loss is recorded at every epoch.
   * The model state dictionary with the minimum validation loss is preserved in memory (and saved to disk only when `output_dir` is explicitly designated).
   * Generated weight files are excluded from Git commits.
7. **Strict Partition Failure**:
   * If either partition yields 0 usable windows (due to short segments or ambiguous annotations), training aborts with an informative `ValueError`. Artificial partitions are never manufactured.

---

## 4. Evaluation & Metric Integration

The training pipeline integrates directly with the canonical M2 evaluation helper (`compute_metrics` from `vigil.ml.baseline.evaluation`):

* **Evaluated Metrics**:
  * Overall Accuracy on validation windows.
  * Macro Precision, Macro Recall, and Macro F1 Score.
  * Canonical $3 \times 3$ confusion matrix:
    $$\begin{pmatrix} C_{00} & C_{01} & C_{02} \\ C_{10} & C_{11} & C_{12} \\ C_{20} & C_{21} & C_{22} \end{pmatrix}$$
* **Support & Abstention Tracking**:
  * Classes with zero support in the validation set are explicitly surfaced in `zero_support_classes`.
  * Macro F1 is noted as unvalidated across all 3 classes if any canonical class lacks validation support.
* **Separation of Concerns**:
  * Model predictions are strictly separated from ground-truth labels.
  * Ambiguous ground-truth samples (`label_id = -1`) are discarded before window creation.

---

## 5. Dataset Status & Experimental Limitations

* **Local Dataset Inspection**: As verified in milestone audits, no physical driver recordings exist under `data/recordings/` in this repository.
* **Synthetic Test Fixtures**: Model architecture, training mechanics, gradient descent, loss reduction, and evaluation metrics are validated using synthetic fixtures in `tests/ml/`.
* **Important Disclaimer**: Synthetic fixtures prove code correctness, tensor contracts, and deterministic execution; they **do not establish real-world drowsiness detection performance**. Calibration on real-world multi-subject sensor data remains required before operational deployment.
