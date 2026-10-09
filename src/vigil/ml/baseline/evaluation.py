"""Abstention-aware evaluation helper and leakage-safe splitters for Vigil M2.

This module implements the evaluation metrics and splitting rules specified in
docs/m2-evaluation-protocol.md and the M2 D3 baseline contract:
    1. Overall Accuracy (on evaluated/covered predictions)
    2. Precision (per-class and macro-averaged)
    3. Recall / Sensitivity (per-class and macro-averaged)
    4. Macro F1 Score
    5. Confusion Matrix (3x3 canonically ordered [0: ACTIVE, 1: DROWSY, 2: SLEEPING])
    6. Prediction Coverage and Abstention Tracking (skipped predictions)

Integrity Rules:
    - Never score missing-feature abstentions as correct ACTIVE predictions.
    - Report prediction coverage and count skipped predictions.
    - Ambiguous ground truth (label_id=-1) is excluded from supervised benchmarks.
    - Subject-first grouped splitting: all sessions for a subject remain in one partition.
    - Single-subject fallback to session-level grouping; never random row/window splits.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from vigil.ml.baseline.classifier import RuleBasedBaseline
from vigil.ml.data.schema import CANONICAL_CLASSES, FatigueLabel, SampleRecord


@dataclass(frozen=True)
class BaselineEvaluationReport:
    """Standard evaluation report for Vigil M2 models and baselines.

    Attributes:
        accuracy: Overall classification accuracy on valid evaluated targets.
        macro_precision: Unweighted arithmetic mean of per-class precisions.
        macro_recall: Unweighted arithmetic mean of per-class recalls.
        macro_f1: Unweighted arithmetic mean of per-class F1 scores.
        coverage: Proportion of eligible non-ambiguous samples that received a prediction.
        per_class_precision: Mapping from class ID to precision score.
        per_class_recall: Mapping from class ID to recall score.
        per_class_f1: Mapping from class ID to F1 score.
        per_class_support: Number of ground-truth samples per class among evaluated samples.
        confusion_matrix: 3x3 matrix in canonical order [[0->0, 0->1, 0->2], ...].
        total_samples: Total samples processed.
        eligible_samples: Total samples excluding ambiguous annotations.
        evaluated_samples: Samples with both a valid canonical ground-truth and prediction.
        skipped_predictions: Samples where classifier abstained due to missing features.
        ambiguous_samples: Ambiguous ground-truth samples excluded from evaluation.
        zero_support_classes: List of canonical class IDs with 0 evaluation support.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    coverage: float
    per_class_precision: dict[int, float]
    per_class_recall: dict[int, float]
    per_class_f1: dict[int, float]
    per_class_support: dict[int, int]
    confusion_matrix: list[list[int]]
    total_samples: int
    eligible_samples: int
    evaluated_samples: int
    skipped_predictions: int
    ambiguous_samples: int
    zero_support_classes: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "coverage": self.coverage,
            "per_class_precision": {
                FatigueLabel(c).name: self.per_class_precision[c]
                for c in CANONICAL_CLASSES
            },
            "per_class_recall": {
                FatigueLabel(c).name: self.per_class_recall[c]
                for c in CANONICAL_CLASSES
            },
            "per_class_f1": {
                FatigueLabel(c).name: self.per_class_f1[c]
                for c in CANONICAL_CLASSES
            },
            "per_class_support": {
                FatigueLabel(c).name: self.per_class_support[c]
                for c in CANONICAL_CLASSES
            },
            "confusion_matrix": self.confusion_matrix,
            "total_samples": self.total_samples,
            "eligible_samples": self.eligible_samples,
            "evaluated_samples": self.evaluated_samples,
            "skipped_predictions": self.skipped_predictions,
            "ambiguous_samples": self.ambiguous_samples,
            "zero_support_classes": [
                FatigueLabel(c).name for c in self.zero_support_classes
            ],
        }

    def to_markdown(self) -> str:
        """Render a formatted markdown summary conforming to M2 evaluation standards."""
        lines = [
            "### Evaluation Report",
            "",
            f"- **Overall Accuracy**: {self.accuracy:.4f}",
            f"- **Macro Precision**: {self.macro_precision:.4f}",
            f"- **Macro Recall**: {self.macro_recall:.4f}",
            f"- **Macro F1 Score**: {self.macro_f1:.4f}",
            f"- **Prediction Coverage**: {self.coverage:.4f} ({self.evaluated_samples}/{self.eligible_samples})",
            f"- **Skipped Predictions (Abstentions)**: {self.skipped_predictions}",
            f"- **Excluded Ambiguous Samples**: {self.ambiguous_samples}",
            f"- **Total Samples**: {self.total_samples}",
            "",
            "#### Per-Class Metrics",
            "| Class | ID | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|---|",
        ]
        for c in CANONICAL_CLASSES:
            name = FatigueLabel(c).name
            p = self.per_class_precision.get(c, 0.0)
            r = self.per_class_recall.get(c, 0.0)
            f = self.per_class_f1.get(c, 0.0)
            supp = self.per_class_support.get(c, 0)
            lines.append(f"| {name} | {c} | {p:.4f} | {r:.4f} | {f:.4f} | {supp} |")

        lines.extend([
            "",
            "#### Confusion Matrix (Canonical Order: 0: ACTIVE, 1: DROWSY, 2: SLEEPING)",
            "| True \\ Pred | ACTIVE (0) | DROWSY (1) | SLEEPING (2) |",
            "|---|---|---|---|",
            f"| **ACTIVE (0)** | {self.confusion_matrix[0][0]} | {self.confusion_matrix[0][1]} | {self.confusion_matrix[0][2]} |",
            f"| **DROWSY (1)** | {self.confusion_matrix[1][0]} | {self.confusion_matrix[1][1]} | {self.confusion_matrix[1][2]} |",
            f"| **SLEEPING (2)** | {self.confusion_matrix[2][0]} | {self.confusion_matrix[2][1]} | {self.confusion_matrix[2][2]} |",
        ])

        if self.zero_support_classes:
            zero_names = [FatigueLabel(c).name for c in self.zero_support_classes]
            warning_msg = (
                f"> Classes with zero evaluation support: {zero_names}. "
                "Per M2 evaluation protocol, Macro F1 cannot be claimed as a validated three-class benchmark."
            )
            lines.extend([
                "",
                "> [!WARNING]",
                warning_msg,
            ])

        return "\n".join(lines)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int | None],
    total_samples: int | None = None,
    ambiguous_count: int = 0,
) -> BaselineEvaluationReport:
    """Compute abstention-aware M2 multi-class evaluation metrics.

    Args:
        y_true: List of ground-truth integer labels (must be in CANONICAL_CLASSES).
        y_pred: List of predicted integer labels (or None if classifier abstained).
        total_samples: Total number of raw samples (including ambiguous).
        ambiguous_count: Count of ambiguous ground-truth samples excluded.

    Returns:
        BaselineEvaluationReport with the 5 core M2 metrics, coverage, and abstention counts.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have equal length."
        )

    eligible_samples = len(y_true)
    if total_samples is None:
        total_samples = eligible_samples + ambiguous_count

    # Filter out abstentions (y_pred is None) so missing features do not become ACTIVE predictions
    valid_true: list[int] = []
    valid_pred: list[int] = []
    skipped_predictions = 0

    for t, p in zip(y_true, y_pred):
        if t not in CANONICAL_CLASSES:
            raise ValueError(
                f"Ground-truth targets must be in canonical classes (0, 1, 2), got {t}."
            )
        if p is None:
            skipped_predictions += 1
            continue
        if p not in CANONICAL_CLASSES:
            raise ValueError(
                f"Predictions must be in canonical classes (0, 1, 2) or None, got {p}."
            )
        valid_true.append(t)
        valid_pred.append(p)

    evaluated_samples = len(valid_true)
    coverage = (evaluated_samples / eligible_samples) if eligible_samples > 0 else 0.0

    # Build 3x3 confusion matrix strictly in canonical order [0, 1, 2]
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t, p in zip(valid_true, valid_pred):
        cm[t][p] += 1

    per_class_precision: dict[int, float] = {}
    per_class_recall: dict[int, float] = {}
    per_class_f1: dict[int, float] = {}
    per_class_support: dict[int, int] = {}
    zero_support_classes: list[int] = []

    total_correct = 0

    for c in CANONICAL_CLASSES:
        tp = cm[c][c]
        total_correct += tp

        support = sum(cm[c])
        per_class_support[c] = support
        if support == 0:
            zero_support_classes.append(c)

        predicted_c = sum(cm[row][c] for row in range(3))

        # Precision = TP / (TP + FP) with zero-division handling
        p_val = (tp / predicted_c) if predicted_c > 0 else 0.0
        per_class_precision[c] = p_val

        # Recall = TP / (TP + FN) with zero-division handling
        r_val = (tp / support) if support > 0 else 0.0
        per_class_recall[c] = r_val

        # F1 Score
        f1_val = (2.0 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0.0 else 0.0
        per_class_f1[c] = f1_val

    accuracy = (total_correct / evaluated_samples) if evaluated_samples > 0 else 0.0
    macro_precision = sum(per_class_precision.values()) / 3.0
    macro_recall = sum(per_class_recall.values()) / 3.0
    macro_f1 = sum(per_class_f1.values()) / 3.0

    return BaselineEvaluationReport(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        coverage=coverage,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_matrix=cm,
        total_samples=total_samples,
        eligible_samples=eligible_samples,
        evaluated_samples=evaluated_samples,
        skipped_predictions=skipped_predictions,
        ambiguous_samples=ambiguous_count,
        zero_support_classes=zero_support_classes,
    )


def evaluate_baseline(
    samples: list[SampleRecord],
    classifier: RuleBasedBaseline | None = None,
) -> BaselineEvaluationReport:
    """Evaluate a RuleBasedBaseline classifier on a sequence of SampleRecords.

    IMPORTANT INTEGRITY RULES:
        - Predictions are generated without reading sample.label / sample.label_id.
        - Missing features cause the classifier to abstain (label=None), which is counted
          in skipped_predictions and coverage, NEVER scored as correct ACTIVE predictions.
        - Ambiguous samples (label_id == -1) are excluded from supervised evaluation.

    Args:
        samples: List of SampleRecords from a validation session or partition.
        classifier: RuleBasedBaseline classifier instance. If None, default
            provisional baseline is used.

    Returns:
        BaselineEvaluationReport with 5 core M2 metrics, coverage, and abstention counts.
    """
    if classifier is None:
        classifier = RuleBasedBaseline()

    classifier.reset()

    y_true: list[int] = []
    y_pred: list[int | None] = []
    ambiguous_count = 0

    for sample in samples:
        # Step 1: Predict strictly using observational features (never reading label)
        prediction = classifier.classify(sample)

        # Step 2: Extract ground-truth label for evaluation only
        gt_id = sample.label_id
        if gt_id == FatigueLabel.AMBIGUOUS:
            ambiguous_count += 1
            continue

        y_true.append(gt_id)
        y_pred.append(prediction.label_id)

    return compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        total_samples=len(samples),
        ambiguous_count=ambiguous_count,
    )


@dataclass(frozen=True)
class GroupSplitResult:
    """Result of subject-first or session-fallback grouped splitting.

    Attributes:
        train_sessions: Sorted list of session IDs assigned to Training.
        val_sessions: Sorted list of session IDs assigned to Validation.
        strategy: 'subject' (cross-subject), 'session' (single-subject fallback), or 'none'.
        subject_count: Number of unique subjects present.
        session_count: Number of unique sessions present.
        limitation_note: Explanatory note when split strategy is constrained or impossible.
    """

    train_sessions: list[str]
    val_sessions: list[str]
    strategy: str
    subject_count: int
    session_count: int
    limitation_note: str | None


def group_split_sessions(
    session_ids: list[str],
    subject_map: dict[str, str] | None = None,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Convenience wrapper returning (train_sessions, val_sessions).

    Maintains backward compatibility with callers expecting a tuple.
    """
    res = group_split_by_subject_or_session(
        session_ids=session_ids,
        subject_map=subject_map,
        train_ratio=train_ratio,
        seed=seed,
    )
    return (res.train_sessions, res.val_sessions)


def group_split_by_subject_or_session(
    session_ids: list[str],
    subject_map: dict[str, str] | None = None,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> GroupSplitResult:
    """Perform subject-first grouped splitting with session fallback.

    Enforces docs/m2-evaluation-protocol.md Section 3:
        1. When multiple subjects exist, all sessions belonging to a subject remain
           exclusively in a single partition (cross-subject generalization).
        2. When only one subject exists, fall back to session-level grouping
           (session generalization only; limitation reported).
        3. Never split individual frames or windows randomly.
        4. If fewer than 2 usable groups exist, report that held-out evaluation is impossible.

    Args:
        session_ids: List of session IDs to partition.
        subject_map: Optional dictionary mapping session_id -> subject_id. If None,
            each session is assumed to have an independent subject if subject IDs are unknown.
        train_ratio: Target proportion assigned to training (default 0.75).
        seed: Deterministic random seed (default 42).

    Returns:
        GroupSplitResult containing partition assignments, strategy, and limitations.
    """
    if not session_ids:
        return GroupSplitResult(
            train_sessions=[],
            val_sessions=[],
            strategy="none",
            subject_count=0,
            session_count=0,
            limitation_note="No sessions provided for partitioning.",
        )

    unique_sessions = sorted(set(session_ids))

    # Map session to subject
    if subject_map is None:
        # If no subject mapping supplied, assume distinct session-level entities
        session_to_subj = {s: s for s in unique_sessions}
    else:
        session_to_subj = {s: subject_map.get(s, s) for s in unique_sessions}

    subj_to_sessions: dict[str, list[str]] = defaultdict(list)
    for s in unique_sessions:
        subj = session_to_subj[s]
        subj_to_sessions[subj].append(s)

    unique_subjects = sorted(subj_to_sessions.keys())
    subject_count = len(unique_subjects)
    session_count = len(unique_sessions)

    rng = random.Random(seed)

    # Strategy 1: Multi-subject -> Subject-level grouping
    if subject_count > 1:
        shuffled_subjs = list(unique_subjects)
        rng.shuffle(shuffled_subjs)

        n_train_subjs = max(1, min(subject_count - 1, round(subject_count * train_ratio)))
        train_subjs = set(shuffled_subjs[:n_train_subjs])
        val_subjs = set(shuffled_subjs[n_train_subjs:])

        train_sessions = sorted(
            s for subj in train_subjs for s in subj_to_sessions[subj]
        )
        val_sessions = sorted(
            s for subj in val_subjs for s in subj_to_sessions[subj]
        )

        return GroupSplitResult(
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            strategy="subject",
            subject_count=subject_count,
            session_count=session_count,
            limitation_note=None,
        )

    # Strategy 2: Single-subject -> Fall back to session-level grouping
    if session_count > 1:
        shuffled_sessions = list(unique_sessions)
        rng.shuffle(shuffled_sessions)

        n_train = max(1, min(session_count - 1, round(session_count * train_ratio)))
        train_sessions = sorted(shuffled_sessions[:n_train])
        val_sessions = sorted(shuffled_sessions[n_train:])

        return GroupSplitResult(
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            strategy="session",
            subject_count=1,
            session_count=session_count,
            limitation_note=(
                "Single subject detected across all sessions. Evaluation is restricted to "
                "session-level generalization and does not evaluate cross-subject generalization."
            ),
        )

    # Strategy 3: Single session and single subject -> Partition impossible
    return GroupSplitResult(
        train_sessions=unique_sessions,
        val_sessions=[],
        strategy="none",
        subject_count=1,
        session_count=1,
        limitation_note=(
            "Only one session is available. A valid held-out validation partition is impossible "
            "without violating temporal session integrity."
        ),
    )
