"""Unit tests for Vigil M2 baseline evaluation helper and metric calculators."""

from __future__ import annotations

import pytest

from vigil.ml.baseline.classifier import RuleBasedBaseline
from vigil.ml.baseline.evaluation import (
    BaselineEvaluationReport,
    compute_metrics,
    evaluate_baseline,
    group_split_sessions,
)
from vigil.ml.data.schema import SampleRecord


def make_eval_sample(
    frame_index: int,
    timestamp_ms: float,
    ear_avg: float,
    label: str,
    label_id: int,
    is_eye_closed: bool = False,
    session_id: str = "session_001",
) -> SampleRecord:
    """Helper to create a SampleRecord with specified features and label."""
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id="subject_001",
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=True,
        landmarks_valid=True,
        ear_left=ear_avg,
        ear_right=ear_avg,
        ear_avg=ear_avg,
        is_eye_closed=is_eye_closed,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=0.05,
        fps=30.0,
        label=label,
        label_id=label_id,
    )


class TestMetricCalculations:
    """Tests for compute_metrics mathematical formulations and rules."""

    def test_perfect_predictions_across_all_classes(self) -> None:
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]

        report = compute_metrics(y_true, y_pred)
        assert isinstance(report, BaselineEvaluationReport)
        assert report.accuracy == 1.0
        assert report.macro_precision == 1.0
        assert report.macro_recall == 1.0
        assert report.macro_f1 == 1.0
        assert report.per_class_precision == {0: 1.0, 1: 1.0, 2: 1.0}
        assert report.per_class_recall == {0: 1.0, 1: 1.0, 2: 1.0}
        assert report.per_class_f1 == {0: 1.0, 1: 1.0, 2: 1.0}
        assert report.per_class_support == {0: 2, 1: 2, 2: 2}
        assert report.confusion_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert report.zero_support_classes == []

    def test_known_confusion_matrix_and_metrics(self) -> None:
        # Ground truth:
        # Class 0: 3 samples -> 2 predicted as 0, 1 predicted as 1
        # Class 1: 2 samples -> 1 predicted as 1, 1 predicted as 2
        # Class 2: 1 sample  -> 1 predicted as 2
        # Total: 6 samples
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 1, 1, 2, 2]

        report = compute_metrics(y_true, y_pred)

        # Expected Confusion Matrix:
        # [ [2, 1, 0],
        #   [0, 1, 1],
        #   [0, 0, 1] ]
        assert report.confusion_matrix == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]

        # Accuracy = (2 + 1 + 1) / 6 = 4 / 6 = 0.6666...
        assert pytest.approx(report.accuracy, rel=1e-4) == 4.0 / 6.0

        # Class 0: TP=2, FP=0, FN=1 -> Prec=2/2=1.0, Rec=2/3=0.6667, F1 = 2*(1*2/3)/(1 + 2/3) = 0.80
        assert pytest.approx(report.per_class_precision[0], rel=1e-4) == 1.0
        assert pytest.approx(report.per_class_recall[0], rel=1e-4) == 2.0 / 3.0
        assert pytest.approx(report.per_class_f1[0], rel=1e-4) == 0.80

        # Class 1: TP=1, FP=1, FN=1 -> Prec=1/2=0.5, Rec=1/2=0.5, F1 = 0.5
        assert pytest.approx(report.per_class_precision[1], rel=1e-4) == 0.5
        assert pytest.approx(report.per_class_recall[1], rel=1e-4) == 0.5
        assert pytest.approx(report.per_class_f1[1], rel=1e-4) == 0.5

        # Class 2: TP=1, FP=1, FN=0 -> Prec=1/2=0.5, Rec=1/1=1.0, F1 = 2*(0.5*1)/(0.5 + 1) = 2/3
        assert pytest.approx(report.per_class_precision[2], rel=1e-4) == 0.5
        assert pytest.approx(report.per_class_recall[2], rel=1e-4) == 1.0
        assert pytest.approx(report.per_class_f1[2], rel=1e-4) == 2.0 / 3.0

        # Macro F1 = (0.80 + 0.50 + 0.6667) / 3 = 1.9667 / 3 = 0.6556
        expected_macro_f1 = (0.80 + 0.50 + (2.0 / 3.0)) / 3.0
        assert pytest.approx(report.macro_f1, rel=1e-4) == expected_macro_f1

    def test_zero_division_rule_and_missing_support_reporting(self) -> None:
        # All samples are class 0 (ACTIVE)
        # Class 1 and Class 2 have zero support in ground truth
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]

        report = compute_metrics(y_true, y_pred)

        assert report.accuracy == 1.0
        assert report.per_class_support == {0: 3, 1: 0, 2: 0}
        assert report.zero_support_classes == [1, 2]

        # Zero division rule: if TP + FP == 0 or TP + FN == 0, evaluates to 0.0
        assert report.per_class_precision[1] == 0.0
        assert report.per_class_recall[1] == 0.0
        assert report.per_class_f1[1] == 0.0

        assert report.per_class_precision[2] == 0.0
        assert report.per_class_recall[2] == 0.0
        assert report.per_class_f1[2] == 0.0

        # Macro F1 is (1.0 + 0.0 + 0.0) / 3 = 1/3 = 0.3333
        assert pytest.approx(report.macro_f1, rel=1e-4) == 1.0 / 3.0

    def test_length_mismatch_and_invalid_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            compute_metrics([0, 1], [0])

        with pytest.raises(ValueError, match="canonical classes"):
            compute_metrics([0, 5], [0, 1])

        with pytest.raises(ValueError, match="canonical classes"):
            compute_metrics([0, 1], [0, 99])

    def test_to_dict_and_to_markdown_formatting(self) -> None:
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        report = compute_metrics(y_true, y_pred, total_samples=4, ambiguous_count=1)

        d = report.to_dict()
        assert d["accuracy"] == 1.0
        assert d["ambiguous_samples"] == 1
        assert "ACTIVE" in d["per_class_precision"]

        md = report.to_markdown()
        assert "### Evaluation Report" in md
        assert "**Excluded Ambiguous Samples**: 1" in md


class TestEvaluateBaselineEndToEnd:
    """Tests for evaluate_baseline function integrating RuleBasedBaseline."""

    def test_evaluate_baseline_filters_ambiguous_and_computes_metrics(self) -> None:
        clf = RuleBasedBaseline()
        records = [
            # Active sample
            make_eval_sample(0, 0.0, 0.35, "ACTIVE", 0),
            # Ambiguous sample (must be excluded from target metrics!)
            make_eval_sample(1, 33.3, 0.35, "AMBIGUOUS", -1),
            # Another active sample
            make_eval_sample(2, 66.6, 0.35, "ACTIVE", 0),
        ]

        report = evaluate_baseline(records, classifier=clf)

        assert report.total_samples == 3
        assert report.evaluated_samples == 2
        assert report.ambiguous_samples == 1
        assert report.accuracy == 1.0
        assert report.per_class_support[0] == 2
        assert report.zero_support_classes == [1, 2]


class TestGroupSplitSessions:
    """Tests for deterministic grouped session splitting."""

    def test_group_split_is_deterministic_and_disjoint(self) -> None:
        sessions = [f"session_{i:03d}" for i in range(10)]

        train_1, val_1 = group_split_sessions(sessions, train_ratio=0.70, seed=42)
        train_2, val_2 = group_split_sessions(sessions, train_ratio=0.70, seed=42)

        # Deterministic with seed
        assert train_1 == train_2
        assert val_1 == val_2

        # Disjoint sets: no session overlap
        assert set(train_1).isdisjoint(set(val_1))
        assert len(train_1) + len(val_1) == 10
        assert len(train_1) == 7
        assert len(val_1) == 3

    def test_single_session_fallback(self) -> None:
        train, val = group_split_sessions(["session_single"], train_ratio=0.70)
        assert train == ["session_single"]
        assert val == []
