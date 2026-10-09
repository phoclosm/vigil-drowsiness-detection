"""Rule-based fatigue baseline and evaluation tools for Vigil M2."""

from vigil.ml.baseline.classifier import BaselinePrediction, RuleBasedBaseline
from vigil.ml.baseline.config import BaselineConfig
from vigil.ml.baseline.evaluation import (
    BaselineEvaluationReport,
    compute_metrics,
    evaluate_baseline,
    group_split_sessions,
)

__all__ = [
    "BaselineConfig",
    "BaselineEvaluationReport",
    "BaselinePrediction",
    "RuleBasedBaseline",
    "compute_metrics",
    "evaluate_baseline",
    "group_split_sessions",
]
