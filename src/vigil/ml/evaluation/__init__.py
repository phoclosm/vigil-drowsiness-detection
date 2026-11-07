"""Evaluation and baseline comparison utilities for Vigil M2."""

from vigil.ml.evaluation.comparison import (
    CandidateComparisonReport,
    CandidateEvaluationReport,
    ModelComparisonReport,
    compare_candidates,
    compare_models,
    evaluate_recording_directory,
)
from vigil.ml.evaluation.input_audit import EvaluationInputAudit, audit_evaluation_input

__all__ = [
    "EvaluationInputAudit",
    "CandidateComparisonReport",
    "CandidateEvaluationReport",
    "ModelComparisonReport",
    "audit_evaluation_input",
    "compare_candidates",
    "compare_models",
    "evaluate_recording_directory",
]
