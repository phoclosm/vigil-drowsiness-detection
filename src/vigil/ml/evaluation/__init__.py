"""Evaluation and baseline comparison utilities for Vigil M2."""

from vigil.ml.evaluation.comparison import (
    ModelComparisonReport,
    compare_models,
    evaluate_recording_directory,
)
from vigil.ml.evaluation.input_audit import EvaluationInputAudit, audit_evaluation_input

__all__ = [
    "EvaluationInputAudit",
    "ModelComparisonReport",
    "audit_evaluation_input",
    "compare_models",
    "evaluate_recording_directory",
]
