"""Rule-based fatigue baseline and evaluation tools for Vigil M2."""

from vigil.ml.baseline.classifier import BaselinePrediction, RuleBasedBaseline
from vigil.ml.baseline.config import BaselineConfig
from vigil.ml.baseline.evaluation import (
    BaselineEvaluationReport,
    GroupSplitResult,
    compute_metrics,
    evaluate_baseline,
    group_split_by_subject_or_session,
    group_split_sessions,
)

__all__ = [
    "BaselineConfig",
    "BaselineEvaluationReport",
    "BaselinePrediction",
    "GroupSplitResult",
    "RuleBasedBaseline",
    "compute_metrics",
    "evaluate_baseline",
    "group_split_by_subject_or_session",
    "group_split_sessions",
]
