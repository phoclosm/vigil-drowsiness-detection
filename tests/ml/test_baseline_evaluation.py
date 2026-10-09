"""Unit tests for Vigil M2 abstention-aware evaluation helper and splitters."""

from __future__ import annotations

from vigil.ml.baseline.classifier import RuleBasedBaseline
from vigil.ml.baseline.evaluation import (
    BaselineEvaluationReport,
    GroupSplitResult,
    compute_metrics,
    evaluate_baseline,
    group_split_by_subject_or_session,
    group_split_sessions,
)
from vigil.ml.data.schema import SampleRecord


def make_eval_sample(
    frame_index: int,
    timestamp_ms: float,
    label: str,
    label_id: int,
    is_eye_closed: bool | None = False,
    face_detected: bool = True,
    landmarks_valid: bool = True,
    session_id: str = "session_001",
    subject_id: str = "subject_001",
) -> SampleRecord:
    """Helper to create a SampleRecord with specified features and label."""
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=face_detected,
        landmarks_valid=landmarks_valid,
        ear_left=0.30 if landmarks_valid else None,
        ear_right=0.30 if landmarks_valid else None,
        ear_avg=0.30 if landmarks_valid else None,
        is_eye_closed=is_eye_closed,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=0.05,
        fps=30.0,
        label=label,
        label_id=label_id,
    )


class TestAbstentionAwareMetrics:
    """Tests for compute_metrics with coverage, abstentions, and canonical metrics."""

    def test_coverage_and_abstention_tracking(self) -> None:
        # 4 eligible samples: 3 evaluated, 1 abstained (None)
        y_true = [0, 1, 2, 0]
        y_pred = [0, 1, 2, None]

        report = compute_metrics(y_true, y_pred, total_samples=5, ambiguous_count=1)

        assert report.total_samples == 5
        assert report.eligible_samples == 4
        assert report.evaluated_samples == 3
        assert report.skipped_predictions == 1
        assert report.ambiguous_samples == 1
        assert report.coverage == 0.75  # 3 evaluated / 4 eligible

        # On the 3 evaluated samples, accuracy is 100%
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0

    def test_missing_features_never_scored_as_active(self) -> None:
        """Regression test: Abstentions must not be treated as ACTIVE (0)."""
        # Ground truth: 2 ACTIVE samples
        # Model: 1 predicts ACTIVE (0), 1 abstains (None)
        y_true = [0, 0]
        y_pred = [0, None]

        report = compute_metrics(y_true, y_pred)

        assert report.eligible_samples == 2
        assert report.evaluated_samples == 1
        assert report.skipped_predictions == 1
        assert report.coverage == 0.50
        # If the None had been falsely scored as ACTIVE, evaluated_samples would be 2 and accuracy 1.0
        # Instead, only the 1 covered prediction is evaluated
        assert report.per_class_support[0] == 1
        assert report.accuracy == 1.0

    def test_all_abstained_evaluates_zero_coverage_cleanly(self) -> None:
        y_true = [0, 1]
        y_pred = [None, None]

        report = compute_metrics(y_true, y_pred)
        assert report.coverage == 0.0
        assert report.evaluated_samples == 0
        assert report.skipped_predictions == 2
        assert report.accuracy == 0.0
        assert report.macro_f1 == 0.0

    def test_perfect_predictions_metrics(self) -> None:
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]

        report = compute_metrics(y_true, y_pred)
        assert isinstance(report, BaselineEvaluationReport)
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0
        assert report.coverage == 1.0
        assert report.skipped_predictions == 0
        assert report.confusion_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_markdown_and_dict_output_contain_coverage(self) -> None:
        y_true = [0, 1]
        y_pred = [0, None]
        report = compute_metrics(y_true, y_pred, total_samples=3, ambiguous_count=1)

        d = report.to_dict()
        assert d["coverage"] == 0.5
        assert d["skipped_predictions"] == 1

        md = report.to_markdown()
        assert "Prediction Coverage" in md
        assert "Skipped Predictions (Abstentions)" in md


class TestEvaluateBaselineEndToEnd:
    """Tests for evaluate_baseline function integrating RuleBasedBaseline."""

    def test_evaluate_baseline_abstains_on_missing_features(self) -> None:
        clf = RuleBasedBaseline()
        records = [
            # Frame 0: Valid active sample
            make_eval_sample(0, 0.0, "ACTIVE", 0, is_eye_closed=False),
            # Frame 1: Missing face/landmarks -> Abstains!
            make_eval_sample(1, 33.3, "ACTIVE", 0, face_detected=False, landmarks_valid=False, is_eye_closed=None),
            # Frame 2: Ambiguous annotation -> Excluded!
            make_eval_sample(2, 66.6, "AMBIGUOUS", -1, is_eye_closed=False),
        ]

        report = evaluate_baseline(records, classifier=clf)

        assert report.total_samples == 3
        assert report.ambiguous_samples == 1
        assert report.eligible_samples == 2
        assert report.skipped_predictions == 1
        assert report.evaluated_samples == 1
        assert report.coverage == 0.50
        assert report.accuracy == 1.0


class TestSubjectFirstGroupedSplitting:
    """Tests for subject-first grouped splitting and session fallback."""

    def test_multi_subject_split_keeps_subject_sessions_disjoint(self) -> None:
        # 4 subjects, each having 2 sessions
        subject_map = {
            "session_1a": "subject_1",
            "session_1b": "subject_1",
            "session_2a": "subject_2",
            "session_2b": "subject_2",
            "session_3a": "subject_3",
            "session_3b": "subject_3",
            "session_4a": "subject_4",
            "session_4b": "subject_4",
        }
        all_sessions = list(subject_map.keys())

        result = group_split_by_subject_or_session(
            session_ids=all_sessions,
            subject_map=subject_map,
            train_ratio=0.75,
            seed=42,
        )

        assert isinstance(result, GroupSplitResult)
        assert result.strategy == "subject"
        assert result.subject_count == 4
        assert result.session_count == 8
        assert result.limitation_note is None

        # Verify cross-subject leakage prevention:
        # Find subjects in train vs subjects in val
        train_subjects = {subject_map[s] for s in result.train_sessions}
        val_subjects = {subject_map[s] for s in result.val_sessions}

        assert train_subjects.isdisjoint(val_subjects)
        assert len(train_subjects) + len(val_subjects) == 4

        # Also test tuple wrapper
        train_s, val_s = group_split_sessions(
            session_ids=all_sessions,
            subject_map=subject_map,
            train_ratio=0.75,
            seed=42,
        )
        assert train_s == result.train_sessions
        assert val_s == result.val_sessions

    def test_single_subject_falls_back_to_session_grouping_with_limitation(self) -> None:
        # 1 subject with 4 sessions
        subject_map = {
            "session_01": "subject_1",
            "session_02": "subject_1",
            "session_03": "subject_1",
            "session_04": "subject_1",
        }
        all_sessions = list(subject_map.keys())

        result = group_split_by_subject_or_session(
            session_ids=all_sessions,
            subject_map=subject_map,
            train_ratio=0.75,
            seed=42,
        )

        assert result.strategy == "session"
        assert result.subject_count == 1
        assert result.session_count == 4
        assert result.limitation_note is not None
        assert "Single subject detected" in result.limitation_note
        assert "session-level generalization" in result.limitation_note

        # Verify sessions are disjoint
        assert set(result.train_sessions).isdisjoint(set(result.val_sessions))
        assert len(result.train_sessions) + len(result.val_sessions) == 4

    def test_single_session_reports_partition_impossible(self) -> None:
        subject_map = {"session_single": "subject_1"}
        result = group_split_by_subject_or_session(
            session_ids=["session_single"],
            subject_map=subject_map,
        )

        assert result.strategy == "none"
        assert result.val_sessions == []
        assert result.limitation_note is not None
        assert "A valid held-out validation partition is impossible" in result.limitation_note
