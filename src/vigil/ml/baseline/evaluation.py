"""Leakage-safe evaluation helper and metric calculators for Vigil M2 baseline.

This module implements the 5 mandatory evaluation metrics specified in
docs/m2-evaluation-protocol.md:
    1. Overall Accuracy
    2. Precision (per-class and macro-averaged)
    3. Recall / Sensitivity (per-class and macro-averaged)
    4. Macro F1 Score
    5. Confusion Matrix (3x3 canonically ordered [0: ACTIVE, 1: DROWSY, 2: SLEEPING])

Integrity Rules:
    - Never fabricate performance metrics.
    - Zero-support classes are explicitly tracked and reported.
    - Ambiguous ground truth (label_id=-1) is excluded from validation metrics.
    - Canonical class order [0, 1, 2] is strictly maintained.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from vigil.ml.baseline.classifier import RuleBasedBaseline
from vigil.ml.data.schema import CANONICAL_CLASSES, FatigueLabel, SampleRecord


@dataclass(frozen=True)
class BaselineEvaluationReport:
    """Standard evaluation report for Vigil M2 models and baselines.

    Attributes:
        accuracy: Overall classification accuracy on valid targets.
        macro_precision: Unweighted arithmetic mean of per-class precisions.
        macro_recall: Unweighted arithmetic mean of per-class recalls.
        macro_f1: Unweighted arithmetic mean of per-class F1 scores.
        per_class_precision: Mapping from class ID to precision score.
        per_class_recall: Mapping from class ID to recall score.
        per_class_f1: Mapping from class ID to F1 score.
        per_class_support: Number of ground-truth samples per class.
        confusion_matrix: 3x3 matrix in canonical order [[0->0, 0->1, 0->2], ...].
        total_samples: Total samples processed.
        evaluated_samples: Count of samples evaluated (excluding ambiguous).
        ambiguous_samples: Count of ambiguous samples excluded.
        zero_support_classes: List of canonical class IDs with 0 validation samples.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: dict[int, float]
    per_class_recall: dict[int, float]
    per_class_f1: dict[int, float]
    per_class_support: dict[int, int]
    confusion_matrix: list[list[int]]
    total_samples: int
    evaluated_samples: int
    ambiguous_samples: int
    zero_support_classes: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
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
            "evaluated_samples": self.evaluated_samples,
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
            f"- **Evaluated Samples**: {self.evaluated_samples}",
            f"- **Excluded Ambiguous Samples**: {self.ambiguous_samples}",
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
    y_pred: list[int],
    total_samples: int | None = None,
    ambiguous_count: int = 0,
) -> BaselineEvaluationReport:
    """Compute standard M2 multi-class evaluation metrics.

    Args:
        y_true: List of ground-truth integer labels (must be in CANONICAL_CLASSES).
        y_pred: List of predicted integer labels (must be in CANONICAL_CLASSES).
        total_samples: Total number of samples before ambiguous exclusions.
        ambiguous_count: Count of ambiguous samples excluded from evaluation.

    Returns:
        BaselineEvaluationReport with the 5 core M2 metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have equal length."
        )

    evaluated_samples = len(y_true)
    if total_samples is None:
        total_samples = evaluated_samples + ambiguous_count

    # Build 3x3 confusion matrix strictly in canonical order [0, 1, 2]
    # cm[true_class][pred_class]
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t, p in zip(y_true, y_pred):
        if t not in CANONICAL_CLASSES or p not in CANONICAL_CLASSES:
            raise ValueError(
                f"Targets and predictions must be canonical classes (0, 1, 2), got true={t}, pred={p}."
            )
        cm[t][p] += 1

    # Per-class support, precision, recall, F1
    per_class_precision: dict[int, float] = {}
    per_class_recall: dict[int, float] = {}
    per_class_f1: dict[int, float] = {}
    per_class_support: dict[int, int] = {}
    zero_support_classes: list[int] = []

    total_correct = 0

    for c in CANONICAL_CLASSES:
        tp = cm[c][c]
        total_correct += tp

        # row sum is ground truth support
        support = sum(cm[c])
        per_class_support[c] = support
        if support == 0:
            zero_support_classes.append(c)

        # column sum is total predicted as c
        predicted_c = sum(cm[row][c] for row in range(3))

        # Precision = TP / (TP + FP) with zero-division handling
        if predicted_c > 0:
            p = tp / predicted_c
        else:
            p = 0.0
        per_class_precision[c] = p

        # Recall = TP / (TP + FN) with zero-division handling
        if support > 0:
            r = tp / support
        else:
            r = 0.0
        per_class_recall[c] = r

        # F1 = 2 * P * R / (P + R)
        if (p + r) > 0.0:
            f1 = 2.0 * (p * r) / (p + r)
        else:
            f1 = 0.0
        per_class_f1[c] = f1

    # Overall accuracy
    if evaluated_samples > 0:
        accuracy = total_correct / evaluated_samples
    else:
        accuracy = 0.0

    # Macro averages (unweighted arithmetic mean across the 3 canonical classes)
    macro_precision = sum(per_class_precision.values()) / 3.0
    macro_recall = sum(per_class_recall.values()) / 3.0
    macro_f1 = sum(per_class_f1.values()) / 3.0

    return BaselineEvaluationReport(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_matrix=cm,
        total_samples=total_samples,
        evaluated_samples=evaluated_samples,
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
        - Ambiguous samples (label_id == -1) are excluded from metric calculation.

    Args:
        samples: List of SampleRecords from a validation session or partition.
        classifier: RuleBasedBaseline classifier instance. If None, default
            provisional baseline is used.

    Returns:
        BaselineEvaluationReport with the 5 core M2 metrics.
    """
    if classifier is None:
        classifier = RuleBasedBaseline()

    classifier.reset()

    y_true: list[int] = []
    y_pred: list[int] = []
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


def group_split_sessions(
    session_ids: list[str],
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministically partition sessions into train and validation sets.

    Conforms to the Grouped Splitting requirement in docs/m2-evaluation-protocol.md
    to prevent cross-frame temporal data leakage.

    Args:
        session_ids: List of unique session IDs to partition.
        train_ratio: Target proportion of sessions assigned to training (default 0.75).
        seed: Deterministic random seed (default 42).

    Returns:
        A tuple of (train_session_ids, val_session_ids).
    """
    if not session_ids:
        return ([], [])

    unique_sessions = sorted(set(session_ids))
    if len(unique_sessions) == 1:
        # Single session cannot be partitioned across subjects; fallback to session warning
        return (unique_sessions, [])

    rng = random.Random(seed)
    shuffled = list(unique_sessions)
    rng.shuffle(shuffled)

    # Allocate integer sessions
    n_train = max(1, min(len(shuffled) - 1, round(len(shuffled) * train_ratio)))
    train_sessions = sorted(shuffled[:n_train])
    val_sessions = sorted(shuffled[n_train:])

    return (train_sessions, val_sessions)
