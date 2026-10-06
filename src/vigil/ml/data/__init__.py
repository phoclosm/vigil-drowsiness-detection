"""Data recording and schema representations for Vigil M2."""

from vigil.ml.data.recorder import SessionRecorder
from vigil.ml.data.schema import (
    CANONICAL_CLASSES,
    CameraConfig,
    FatigueLabel,
    FeatureConfig,
    PerclosConfig,
    ProtocolConfig,
    SampleRecord,
    SessionMetadata,
    validate_label_pair,
)

__all__ = [
    "CANONICAL_CLASSES",
    "CameraConfig",
    "FatigueLabel",
    "FeatureConfig",
    "PerclosConfig",
    "ProtocolConfig",
    "SampleRecord",
    "SessionMetadata",
    "SessionRecorder",
    "validate_label_pair",
]
