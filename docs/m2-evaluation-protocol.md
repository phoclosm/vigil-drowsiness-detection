# Vigil M2 Evaluation Protocol

## 1. Overview and Purpose

This document establishes the evaluation methodology, metric standards, data splitting protocols, and leakage prevention rules for **Milestone 2 (M2: Machine Learning, Deep Learning, and Evaluation)** of the Vigil Driver Drowsiness Detection system.

The objective of this protocol is to ensure that all reported performance figures reflect genuine generalization to unseen driver sessions rather than memorization or temporal leakage.

---

## 2. Required Evaluation Metrics

Every model evaluation in M2—including the rule-based baseline (D3) and the learned neural classifiers (D5/D6)—must report the following five core metrics computed on the held-out validation partition:

1. **Overall Accuracy**
2. **Precision** (per-class and macro-averaged)
3. **Recall / Sensitivity** (per-class and macro-averaged)
4. **Macro F1 Score**
5. **Confusion Matrix** ($3 \times 3$, canonically ordered)

### 2.1 Metric Formulations

Let $C \in \{0: \text{ACTIVE}, 1: \text{DROWSY}, 2: \text{SLEEPING}\}$ represent the canonical classes. Let $TP_c$, $FP_c$, $FN_c$, and $TN_c$ denote true positives, false positives, false negatives, and true negatives for class $c$:

* **Overall Accuracy**:
  $$\text{Accuracy} = \frac{\sum_{c=0}^2 TP_c}{N_{\text{total}}}$$

* **Per-Class Precision and Recall**:
  $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
  *(Zero-division rule: if $TP_c + FP_c = 0$ or $TP_c + FN_c = 0$, the metric evaluates to $0.0$.)*

* **Per-Class F1 Score**:
  $$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

* **Macro F1 Score**:
  $$\text{Macro } F1 = \frac{1}{3} \sum_{c=0}^2 F1_c = \frac{F1_{\text{ACTIVE}} + F1_{\text{DROWSY}} + F1_{\text{SLEEPING}}}{3}$$

* **Confusion Matrix Layout**:
  Rows represent ground truth; columns represent predictions. The matrix must strictly adhere to canonical class order:
  $$\begin{pmatrix}
  N_{\text{ACTIVE} \rightarrow \text{ACTIVE}} & N_{\text{ACTIVE} \rightarrow \text{DROWSY}} & N_{\text{ACTIVE} \rightarrow \text{SLEEPING}} \\
  N_{\text{DROWSY} \rightarrow \text{ACTIVE}} & N_{\text{DROWSY} \rightarrow \text{DROWSY}} & N_{\text{DROWSY} \rightarrow \text{SLEEPING}} \\
  N_{\text{SLEEPING} \rightarrow \text{ACTIVE}} & N_{\text{SLEEPING} \rightarrow \text{DROWSY}} & N_{\text{SLEEPING} \rightarrow \text{SLEEPING}}
  \end{pmatrix}$$

### 2.2 Why Macro F1 is Mandatory for Fatigue Detection

Driver drowsiness detection exhibits severe class imbalance by nature:
1. **Majority Class Dominance**: In typical naturalistic and experimental driving sessions, a driver spends the vast majority of time in the `ACTIVE` state. `DROWSY` episodes and acute `SLEEPING` events represent rare, short-lived transitions.
2. **The Accuracy Paradox**: A trivial majority-class classifier that blindly predicts `ACTIVE` on 100% of frames would achieve deceptively high accuracy. In a driver safety context, such a system is completely ineffective because it fails to detect hazardous states.
3. **Flaw of Micro and Weighted F1**:
   * *Micro F1* equals overall accuracy in multi-class single-label classification and is dominated by `ACTIVE`.
   * *Weighted F1* weights each class score by its sample support, meaning poor recall on `SLEEPING` (small support) has negligible impact on the aggregate score.
4. **Macro F1 as Safety Guardrail**: Macro F1 computes an unweighted arithmetic mean across all three classes, giving equal statistical weight to `ACTIVE`, `DROWSY`, and `SLEEPING`. A model cannot achieve an acceptable Macro F1 without demonstrating high precision and recall on the critical fatigue classes.

### 2.3 Evaluation Validity and Class Support Rules
* **Mandatory Class Support Check**: A validation result must **not** be presented as a meaningful three-class benchmark if one or more canonical classes have zero validation support ($N_{\text{support}} = 0$).
* **Reporting Missing Support**: When a validation partition lacks samples for a class (e.g., no `SLEEPING` frames recorded in early sessions), the evaluator must explicitly report missing class support and note the limitation, rather than making the model appear valid through a zero-filled or artificially inflated metric.
* **Canonical Order Guarantee**: All evaluation summaries and confusion matrices must strictly maintain the canonical class order: `[0: ACTIVE, 1: DROWSY, 2: SLEEPING]`.

---

## 3. Split Strategy: Grouped, Leakage-Safe Partitioning

### 3.1 The Golden Rule
> **CRITICAL RULE**: Under no circumstances may a naive random row-level or sample-level split be performed on temporal frames or sliding windows.

### 3.2 Partitioning Hierarchy
Data must be partitioned using **Grouped Splitting**:

1. **Primary: Subject-Level Grouping (`subject_id`)**:
   * When multiple identifiable subjects exist, all recording sessions belonging to a specific `subject_id` must be assigned exclusively to either the **Training partition** or the **Validation partition**.
   * A subject-level split evaluates genuine **cross-subject generalization**, preventing the model from memorizing subject-specific facial geometry, resting eye morphology, or individual blink kinetics.
2. **Fallback: Session-Level Grouping (`session_id`)**:
   * When true subject-level generalization cannot be performed (e.g., single-subject dataset or unidentifiable subject metadata), grouping must be performed strictly at the continuous `session_id` level.
   * Every frame and every sliding window originating from recording session $S_k$ must reside in the same partition.
   * **Generalization Scope Limitation**: A session-level split on a single-subject dataset evaluates **session generalization only**. A single-subject dataset must **never** be described as demonstrating cross-subject generalization.

### 3.3 Partition Ratios for Small Early Datasets
* **Target vs. Guarantee**:
  * Grouped splitting is a **strict requirement** to prevent leakage.
  * In contrast, the partition ratio (e.g., ~70% to 80% Training / ~20% to 30% Validation) is a **target rather than a guarantee** when the total number of recording sessions is small (as in Week 1).
  * Integer session allocation takes precedence over achieving exact fractional split targets.
* **Deterministic Allocation**:
  * The split assignment must be completely deterministic (e.g., sorting session/subject keys and applying a fixed random seed).
* **Scope Boundary (Week 1)**:
  * Week 1 requires **Training** and **Validation** partitions for model selection and baseline comparison.
  * A separate held-out test set is deferred to later milestones when multi-subject corpora expand.

---

## 4. Temporal Leakage Prevention

### 4.1 Root Causes of Temporal Leakage in Fatigue Time-Series

Temporal data exhibits unique vulnerability to data leakage:

1. **High Frame Autocorrelation**:
   * Video capture operates at 25–30 FPS (one sample every 33–40 ms). Human physiological states change over hundreds of milliseconds to seconds.
   * Frame $t$ and frame $t+1$ within the same session have nearly identical EAR values, head orientation, lighting conditions, and facial expressions.
   * If frame $t$ is in the training set and frame $t+1$ is in the validation set, the model evaluates its ability to interpolate contiguous frames rather than generalize to new fatigue events.
2. **Temporal Window Overlap (Sliding Windows)**:
   * In M2 Day 4, continuous features will be sliced into temporal sliding windows of length $W$ with stride $S < W$.
   * Adjacent windows share $W - S$ identical frames.
   * A naive random split of window indices causes overlapping windows from the same session to appear simultaneously in training and validation, resulting in severe data leakage and artificially inflated metrics.
3. **Subject Identity Memorization**:
   * Individuals have distinct facial features and baseline eye openness.
   * A model evaluated on the same subject it trained on easily learns a trivial per-subject bias rather than the dynamic drop in ocular alertness that signals fatigue.

### 4.2 Concrete Enforcement Rule
* **Session Integrity Invariant**: All sliding windows whose temporal span $[t_{\text{start}}, t_{\text{end}}]$ falls within session $S_k$ must belong to the partition to which session $S_k$ is assigned.
* **No Cross-Partition Sequences**: A temporal sequence or rolling buffer must never bridge across two distinct sessions or across train/validation boundaries.

---

## 5. Reproducibility and Determinism

To ensure experimental defensibility, the evaluation pipeline enforces practical reproducibility standards:

### 5.1 Random Seed Control
* All stochastic operations must accept an explicit `seed` parameter (default: `42`).
* The seed must be initialized across:
  * Python standard library: `random.seed(seed)`
  * NumPy: `np.random.seed(seed)`
  * PyTorch: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`

### 5.2 Deterministic Pipeline Execution
* **Split Assignment**: The assignment of sessions/subjects to partitions must be computed deterministically (e.g., hashing session IDs or sorting keys before fixed-seed shuffling).
* **Preprocessing Isolation**: Normalization parameters (mean, standard deviation, min, max) must be computed **strictly on the Training partition** and applied as a frozen transformation to the Validation partition. Global dataset normalization is strictly prohibited.
* **DataLoader Ordering**: Validation DataLoaders must operate with `shuffle=False` to ensure identical sample evaluation order across runs.
* **Deterministic Metric Calculations**: Metric calculations must use standardized routines with consistent zero-division handling and canonical class label lists (`labels=[0, 1, 2]`).

### 5.3 Honest Practical Reproducibility
* *Disclaimer*: Bitwise numerical invariance across differing hardware architectures, GPU driver versions, and non-deterministic CUDA/cuDNN convolution algorithms cannot be guaranteed across diverse environments.
* *Practical Standard*: Practical reproducibility requires identical dataset partition splits, identical random seeds, version-controlled code, and comprehensive experiment configuration logging.

---

## 6. Baseline Comparison Protocol

M2 evaluates two distinct fatigue detection paradigms:
1. **Rule-Based Baseline (M2 Day 3)**: A classical heuristic classifier operating on instantaneous EAR thresholds, blink duration limits, and trailing PERCLOS.
2. **Learned Neural Classifier (M2 Day 5 / Day 6)**: A deep learning model trained on temporal feature sequences using PyTorch.

### 6.1 Fair Comparison Requirements
To make the comparison scientifically defensible, both approaches must satisfy the following constraints:
* **Identical Validation Data**: Both the baseline and the neural classifier must be evaluated on the **exact same held-out validation session partition**.
* **Identical Target Classes**: Both systems must output predictions mapped to the canonical classes: `ACTIVE` (0), `DROWSY` (1), `SLEEPING` (2).
* **Identical Evaluation Code**: Both models must be evaluated using the exact same evaluation script and metric calculation functions.
* **No Numerical Fabrication**: This document does not include placeholder or fabricated benchmark numbers. Numerical results must be recorded only when real experiments are executed on D3 and D6.

---

## 7. Reporting Rules and Experimental Integrity

All experimental documentation in M2 must adhere to strict integrity guidelines:

1. **Grounding in Actual Execution**: All reported metrics (accuracy, precision, recall, macro F1, confusion matrices) must be derived from actual executed benchmark runs. Inventing or rounding up metrics is strictly forbidden.
2. **Preservation of Failures**: Underperforming results, false positive surges, and training instabilities must be preserved and analyzed in experiment logs rather than hidden.
3. **Mandatory Configuration Logging**: Every reported evaluation result must document:
   * Execution timestamp and Git commit SHA.
   * Random seed used.
   * Specific `session_id` values in Training and Validation partitions.
   * Model architecture and hyperparameter configuration.
   * Temporal window length $W$ and stride $S$.
   * Upstream feature subset used.
4. **Explicit Class Ordering**: All printed confusion matrices, classification reports, and tabular summaries must explicitly display the canonical class ordering: `[0: ACTIVE, 1: DROWSY, 2: SLEEPING]`.

---

## 8. Requirements, Decisions, Assumptions, and Deferred Items

### Requirements
* All evaluations must report Accuracy, Macro Precision, Macro Recall, Macro F1, and the $3 \times 3$ Confusion Matrix.
* Macro F1 is the primary model selection metric due to class imbalance.
* Zero-support classes must be explicitly reported rather than hidden under zero-filled aggregates.
* Data partitioning must use grouped splitting at the `subject_id` or `session_id` level.
* Grouped splitting is a strict requirement; train/val ratio is a target for small datasets.
* Random row-level splitting of time-series samples or sliding windows is strictly prohibited.
* Evaluated models must be compared on the exact same validation partition using the same target classes.

### Design Decisions
* Week 1 evaluates a two-way partition: Training and Validation.
* Normalization parameters are fit exclusively on the Training set.
* Class order is fixed across all evaluation tables: `ACTIVE` (0), `DROWSY` (1), `SLEEPING` (2).

### Assumptions
* Sessions contain contiguous frames with valid monotonic elapsed timestamps.
* Recording sessions contain labeled episodes representative of alert, drowsy, or sleeping states.

### Deferred Decisions
* Test set held-out partition $\rightarrow$ **Deferred to multi-subject expansion**.
* Specific hyperparameter tuning benchmarks $\rightarrow$ **Deferred to M2 Day 5/Day 6**.
* Hardware-accelerated inference latency benchmarks $\rightarrow$ **Owned jointly with M1 in Day 6/Release**.
