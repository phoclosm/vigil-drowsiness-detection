"""Stateful streaming inference components for Vigil M2."""

from vigil.ml.inference.engine import (
    CANONICAL_CLASS_LABELS,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_INVALID_INPUT,
    STATUS_OK,
    InferenceResult,
    InputFrame,
    RuntimeInferenceResult,
    TemporalInferenceEngine,
)

__all__ = [
    "CANONICAL_CLASS_LABELS",
    "STATUS_ERROR",
    "STATUS_INSUFFICIENT_HISTORY",
    "STATUS_INVALID_INPUT",
    "STATUS_OK",
    "InferenceResult",
    "InputFrame",
    "RuntimeInferenceResult",
    "TemporalInferenceEngine",
]
