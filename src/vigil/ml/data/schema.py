"""Canonical data schema and fatigue label representations for Vigil M2.

This module implements the structured sample schema, label definitions,
and session metadata models specified in docs/m2-data-contract.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class FatigueLabel(IntEnum):
    """Canonical fatigue states and special annotation states.

    Canonical classes and ordering strictly follow docs/m2-data-contract.md:
        0: ACTIVE
        1: DROWSY
        2: SLEEPING
    Special annotation state:
        -1: AMBIGUOUS
    """

    AMBIGUOUS = -1
    ACTIVE = 0
    DROWSY = 1
    SLEEPING = 2

    @classmethod
    def from_name(cls, name: str) -> FatigueLabel:
        """Resolve a FatigueLabel enum member from its canonical string name."""
        try:
            return cls[name]
        except KeyError:
            valid_names = [m.name for m in cls]
            raise ValueError(
                f"Invalid fatigue label '{name}'. Must be one of: {valid_names}"
            ) from None

    @classmethod
    def from_id(cls, label_id: int) -> FatigueLabel:
        """Resolve a FatigueLabel enum member from its integer label_id."""
        try:
            return cls(label_id)
        except ValueError:
            valid_ids = [m.value for m in cls]
            raise ValueError(
                f"Invalid fatigue label_id '{label_id}'. Must be one of: {valid_ids}"
            ) from None


CANONICAL_CLASSES: tuple[FatigueLabel, ...] = (
    FatigueLabel.ACTIVE,
    FatigueLabel.DROWSY,
    FatigueLabel.SLEEPING,
)


def validate_label_pair(label: str, label_id: int) -> None:
    """Validate that label name and label_id form a consistent canonical pair."""
    expected = FatigueLabel.from_name(label)
    if expected.value != label_id:
        raise ValueError(
            f"Label name '{label}' (id={expected.value}) does not match "
            f"supplied label_id {label_id}."
        )


def validate_iso8601_utc(timestamp_str: str) -> None:
    """Validate that timestamp_str is a non-empty UTC ISO-8601 string."""
    if not isinstance(timestamp_str, str):
        raise TypeError(f"recording_start_utc must be a string, got {type(timestamp_str).__name__}.")
    if not timestamp_str.strip():
        raise ValueError("recording_start_utc must be a non-empty string.")
    normalized = timestamp_str.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as e:
        raise ValueError(
            f"recording_start_utc is not a valid ISO-8601 string: '{timestamp_str}'"
        ) from e
    if dt.tzinfo is None or dt.utcoffset() != timezone.utc.utcoffset(dt):
        raise ValueError(
            f"recording_start_utc must specify UTC timezone (ending with 'Z' or '+00:00'): '{timestamp_str}'"
        )


@dataclass(frozen=True)
class CameraConfig:
    """Camera capture configuration metadata."""

    model: str
    width: int
    height: int
    target_fps: float

    def validate(self) -> None:
        """Validate camera hardware and dimension fields."""
        if not isinstance(self.model, str):
            raise TypeError(f"Camera model must be a string, got {type(self.model).__name__}.")
        if not self.model.strip():
            raise ValueError("Camera model must be a non-empty string.")
        if not isinstance(self.width, int) or isinstance(self.width, bool):
            raise TypeError(f"Camera width must be an integer, got {type(self.width).__name__}.")
        if self.width <= 0:
            raise ValueError(f"Camera width must be a positive integer, got {self.width}.")
        if not isinstance(self.height, int) or isinstance(self.height, bool):
            raise TypeError(f"Camera height must be an integer, got {type(self.height).__name__}.")
        if self.height <= 0:
            raise ValueError(f"Camera height must be a positive integer, got {self.height}.")
        if not isinstance(self.target_fps, (int, float)) or isinstance(self.target_fps, bool):
            raise TypeError(f"Camera target_fps must be numeric, got {type(self.target_fps).__name__}.")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0.0:
            raise ValueError(
                f"Camera target_fps must be positive and finite, got {self.target_fps}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert CameraConfig to serializable dictionary."""
        return {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "target_fps": float(self.target_fps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        """Instantiate CameraConfig from dictionary with validation."""
        if not isinstance(data, dict):
            raise TypeError(f"Camera data must be a dictionary, got {type(data).__name__}.")
        cfg = cls(
            model=data.get("model", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
            target_fps=data.get("target_fps", 0.0),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class PerclosConfig:
    """Trailing PERCLOS temporal measurement configuration."""

    window_mode: str  # "seconds" or "frames"
    window_value: float
    closure_threshold_ear: float

    def validate(self) -> None:
        """Validate trailing PERCLOS configuration parameters."""
        if not isinstance(self.window_mode, str):
            raise TypeError(f"Perclos window_mode must be a string, got {type(self.window_mode).__name__}.")
        if self.window_mode not in ("seconds", "frames"):
            raise ValueError(
                f"Perclos window_mode must be 'seconds' or 'frames', got '{self.window_mode}'."
            )
        if not isinstance(self.window_value, (int, float)) or isinstance(self.window_value, bool):
            raise TypeError(f"Perclos window_value must be numeric, got {type(self.window_value).__name__}.")
        if not math.isfinite(self.window_value) or self.window_value <= 0.0:
            raise ValueError(
                f"Perclos window_value must be positive and finite, got {self.window_value}."
            )
        if self.window_mode == "frames" and int(self.window_value) != self.window_value:
            raise ValueError(
                f"Perclos window_value in 'frames' mode must be an integer, got {self.window_value}."
            )
        if not isinstance(self.closure_threshold_ear, (int, float)) or isinstance(self.closure_threshold_ear, bool):
            raise TypeError(
                f"Perclos closure_threshold_ear must be numeric, got {type(self.closure_threshold_ear).__name__}."
            )
        if not math.isfinite(self.closure_threshold_ear) or self.closure_threshold_ear <= 0.0:
            raise ValueError(
                f"Perclos closure_threshold_ear must be positive and finite, got {self.closure_threshold_ear}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert PerclosConfig to serializable dictionary."""
        return {
            "window_mode": self.window_mode,
            "window_value": float(self.window_value),
            "closure_threshold_ear": float(self.closure_threshold_ear),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerclosConfig:
        """Instantiate PerclosConfig from dictionary with validation."""
        if not isinstance(data, dict):
            raise TypeError(f"Perclos data must be a dictionary, got {type(data).__name__}.")
        cfg = cls(
            window_mode=data.get("window_mode", ""),
            window_value=data.get("window_value", 0.0),
            closure_threshold_ear=data.get("closure_threshold_ear", 0.0),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class FeatureConfig:
    """Feature calculation configuration version and threshold metadata."""

    version: str
    eye_closure_threshold_ear: float
    perclos: PerclosConfig

    def validate(self) -> None:
        """Validate feature calculation configuration parameters."""
        if not isinstance(self.version, str):
            raise TypeError(f"FeatureConfig version must be a string, got {type(self.version).__name__}.")
        if not self.version.strip():
            raise ValueError("FeatureConfig version must be a non-empty string.")
        if not isinstance(self.eye_closure_threshold_ear, (int, float)) or isinstance(
            self.eye_closure_threshold_ear, bool
        ):
            raise TypeError(
                f"eye_closure_threshold_ear must be numeric, got {type(self.eye_closure_threshold_ear).__name__}."
            )
        if not math.isfinite(self.eye_closure_threshold_ear) or self.eye_closure_threshold_ear <= 0.0:
            raise ValueError(
                f"eye_closure_threshold_ear must be positive and finite, got {self.eye_closure_threshold_ear}."
            )
        if not isinstance(self.perclos, PerclosConfig):
            raise TypeError(f"perclos must be an instance of PerclosConfig, got {type(self.perclos).__name__}.")
        self.perclos.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert FeatureConfig to serializable dictionary."""
        return {
            "version": self.version,
            "eye_closure_threshold_ear": float(self.eye_closure_threshold_ear),
            "perclos": self.perclos.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureConfig:
        """Instantiate FeatureConfig from dictionary with validation."""
        if not isinstance(data, dict):
            raise TypeError(f"FeatureConfig data must be a dictionary, got {type(data).__name__}.")
        perclos_data = data.get("perclos", {})
        if not isinstance(perclos_data, dict):
            raise TypeError(f"feature_config.perclos must be a dictionary, got {type(perclos_data).__name__}.")
        cfg = cls(
            version=data.get("version", ""),
            eye_closure_threshold_ear=data.get("eye_closure_threshold_ear", 0.0),
            perclos=PerclosConfig.from_dict(perclos_data),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class ProtocolConfig:
    """Experimental and observation protocol metadata."""

    labeling_method: str
    notes: str = ""

    def validate(self) -> None:
        """Validate protocol information."""
        if not isinstance(self.labeling_method, str):
            raise TypeError(
                f"Protocol labeling_method must be a string, got {type(self.labeling_method).__name__}."
            )
        if not self.labeling_method.strip():
            raise ValueError("Protocol labeling_method must be a non-empty string.")
        if not isinstance(self.notes, str):
            raise TypeError(f"Protocol notes must be a string, got {type(self.notes).__name__}.")

    def to_dict(self) -> dict[str, Any]:
        """Convert ProtocolConfig to serializable dictionary."""
        return {
            "labeling_method": self.labeling_method,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProtocolConfig:
        """Instantiate ProtocolConfig from dictionary with validation."""
        if not isinstance(data, dict):
            raise TypeError(f"ProtocolConfig data must be a dictionary, got {type(data).__name__}.")
        cfg = cls(
            labeling_method=data.get("labeling_method", ""),
            notes=data.get("notes", ""),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class SessionMetadata:
    """Session metadata recorded in session_meta.json."""

    session_id: str
    subject_id: str
    camera: CameraConfig
    feature_config: FeatureConfig
    protocol: ProtocolConfig
    schema_version: str = "0.1.0"
    recording_start_utc: str = ""

    def __post_init__(self) -> None:
        """Default recording_start_utc if empty and validate metadata."""
        if not self.recording_start_utc:
            object.__setattr__(
                self,
                "recording_start_utc",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
        self.validate()

    def validate(self) -> None:
        """Enforce session metadata integrity invariants."""
        if not isinstance(self.schema_version, str):
            raise TypeError(f"schema_version must be a string, got {type(self.schema_version).__name__}.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string.")
        if not isinstance(self.session_id, str):
            raise TypeError(f"session_id must be a string, got {type(self.session_id).__name__}.")
        if not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(self.subject_id, str):
            raise TypeError(f"subject_id must be a string, got {type(self.subject_id).__name__}.")
        if not self.subject_id.strip():
            raise ValueError("subject_id must be a non-empty string.")
        if self.subject_id.strip().lower() == "unknown":
            raise ValueError(
                "subject_id must not be 'unknown'. Provide an explicit anonymized identifier (e.g., 'subject_001')."
            )
        validate_iso8601_utc(self.recording_start_utc)
        if not isinstance(self.camera, CameraConfig):
            raise TypeError(f"camera must be a CameraConfig, got {type(self.camera).__name__}.")
        self.camera.validate()
        if not isinstance(self.feature_config, FeatureConfig):
            raise TypeError(f"feature_config must be a FeatureConfig, got {type(self.feature_config).__name__}.")
        self.feature_config.validate()
        if not isinstance(self.protocol, ProtocolConfig):
            raise TypeError(f"protocol must be a ProtocolConfig, got {type(self.protocol).__name__}.")
        self.protocol.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert SessionMetadata to deterministic dictionary."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "recording_start_utc": self.recording_start_utc,
            "camera": self.camera.to_dict(),
            "feature_config": self.feature_config.to_dict(),
            "protocol": self.protocol.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize SessionMetadata to deterministic JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        """Instantiate SessionMetadata from dictionary with validation."""
        if not isinstance(data, dict):
            raise TypeError(f"SessionMetadata data must be a dictionary, got {type(data).__name__}.")
        camera_data = data.get("camera", {})
        feature_data = data.get("feature_config", {})
        protocol_data = data.get("protocol", {})
        if not isinstance(camera_data, dict):
            raise TypeError(f"camera metadata must be a dictionary, got {type(camera_data).__name__}.")
        if not isinstance(feature_data, dict):
            raise TypeError(f"feature_config metadata must be a dictionary, got {type(feature_data).__name__}.")
        if not isinstance(protocol_data, dict):
            raise TypeError(f"protocol metadata must be a dictionary, got {type(protocol_data).__name__}.")

        return cls(
            schema_version=data.get("schema_version", "0.1.0"),
            session_id=data.get("session_id", ""),
            subject_id=data.get("subject_id", ""),
            recording_start_utc=data.get("recording_start_utc", ""),
            camera=CameraConfig.from_dict(camera_data),
            feature_config=FeatureConfig.from_dict(feature_data),
            protocol=ProtocolConfig.from_dict(protocol_data),
        )

    @classmethod
    def from_json(cls, json_str: str) -> SessionMetadata:
        """Instantiate SessionMetadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class SampleRecord:
    """Canonical M2 training sample representation.

    All fields strictly follow the locked M2 data contract in docs/m2-data-contract.md.
    """

    sample_id: str
    session_id: str
    subject_id: str
    frame_index: int
    timestamp_ms: float
    frame_delta_ms: float | None
    face_detected: bool
    landmarks_valid: bool
    ear_left: float | None
    ear_right: float | None
    ear_avg: float | None
    is_eye_closed: bool | None
    blink_count: int
    blink_duration_ms: float
    perclos: float | None
    fps: float | None
    label: str
    label_id: int

    def __post_init__(self) -> None:
        """Validate sample fields upon construction."""
        self.validate()

    def validate(self) -> None:
        """Enforce strict D1 schema and missing-feature invariants."""
        # 1. Non-empty string identifiers
        if not isinstance(self.sample_id, str):
            raise TypeError(f"sample_id must be a string, got {type(self.sample_id).__name__}.")
        if not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string.")
        if not isinstance(self.session_id, str):
            raise TypeError(f"session_id must be a string, got {type(self.session_id).__name__}.")
        if not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(self.subject_id, str):
            raise TypeError(f"subject_id must be a string, got {type(self.subject_id).__name__}.")
        if not self.subject_id.strip():
            raise ValueError("subject_id must be a non-empty string.")
        if self.subject_id.strip().lower() == "unknown":
            raise ValueError(
                "subject_id must not be 'unknown'. Provide an anonymized identifier (e.g., 'subject_001')."
            )

        # 2. Frame index
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool):
            raise TypeError(f"frame_index must be an integer, got {type(self.frame_index).__name__}.")
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {self.frame_index}.")

        # 3. Monotonic session timestamp
        if not isinstance(self.timestamp_ms, (int, float)) or isinstance(self.timestamp_ms, bool):
            raise TypeError(f"timestamp_ms must be numeric, got {type(self.timestamp_ms).__name__}.")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0.0:
            raise ValueError(f"timestamp_ms must be finite and >= 0.0, got {self.timestamp_ms}.")

        # 4. Frame delta
        if self.frame_index == 0:
            if self.timestamp_ms != 0.0:
                raise ValueError(
                    f"First sample (frame_index=0) must have timestamp_ms == 0.0, got {self.timestamp_ms}."
                )
            if self.frame_delta_ms is not None:
                raise ValueError(
                    f"First sample (frame_index=0) must have frame_delta_ms is None, got {self.frame_delta_ms}."
                )
        else:
            if self.frame_delta_ms is None:
                raise ValueError(
                    f"Sample at frame_index {self.frame_index} must have a positive frame_delta_ms, got None."
                )
            if not isinstance(self.frame_delta_ms, (int, float)) or isinstance(self.frame_delta_ms, bool):
                raise TypeError(f"frame_delta_ms must be numeric, got {type(self.frame_delta_ms).__name__}.")
            if not math.isfinite(self.frame_delta_ms) or self.frame_delta_ms <= 0.0:
                raise ValueError(
                    f"frame_delta_ms must be finite and > 0.0 for frame_index > 0, got {self.frame_delta_ms}."
                )

        # 5. Face and landmark flags
        if not isinstance(self.face_detected, bool):
            raise TypeError(f"face_detected must be an actual boolean, got {type(self.face_detected).__name__}.")
        if not isinstance(self.landmarks_valid, bool):
            raise TypeError(f"landmarks_valid must be an actual boolean, got {type(self.landmarks_valid).__name__}.")

        # 6. Missing feature invariants
        if not self.landmarks_valid:
            if self.ear_left is not None:
                raise ValueError(
                    f"ear_left must be None when landmarks_valid is False, got {self.ear_left}."
                )
            if self.ear_right is not None:
                raise ValueError(
                    f"ear_right must be None when landmarks_valid is False, got {self.ear_right}."
                )
            if self.ear_avg is not None:
                raise ValueError(
                    f"ear_avg must be None when landmarks_valid is False, got {self.ear_avg}."
                )
            if self.is_eye_closed is not None:
                raise ValueError(
                    f"is_eye_closed must be None when landmarks_valid is False, got {self.is_eye_closed}."
                )
        else:
            for ear_name, ear_val in [
                ("ear_left", self.ear_left),
                ("ear_right", self.ear_right),
                ("ear_avg", self.ear_avg),
            ]:
                if ear_val is not None:
                    if not isinstance(ear_val, (int, float)) or isinstance(ear_val, bool):
                        raise TypeError(f"{ear_name} must be numeric or None, got {type(ear_val).__name__}.")
                    if not math.isfinite(ear_val):
                        raise ValueError(f"{ear_name} must be finite or None, got {ear_val}.")
                    if ear_val < 0.0:
                        raise ValueError(f"{ear_name} must be >= 0.0, got {ear_val}.")

            if self.is_eye_closed is not None and not isinstance(self.is_eye_closed, bool):
                raise TypeError(
                    f"is_eye_closed must be a boolean or None, got {type(self.is_eye_closed).__name__}."
                )

        # 7. Blink features
        if not isinstance(self.blink_count, int) or isinstance(self.blink_count, bool):
            raise TypeError(f"blink_count must be an integer, got {type(self.blink_count).__name__}.")
        if self.blink_count < 0:
            raise ValueError(f"blink_count must be an integer >= 0, got {self.blink_count}.")
        if not isinstance(self.blink_duration_ms, (int, float)) or isinstance(self.blink_duration_ms, bool):
            raise TypeError(f"blink_duration_ms must be numeric, got {type(self.blink_duration_ms).__name__}.")
        if not math.isfinite(self.blink_duration_ms) or self.blink_duration_ms < 0.0:
            raise ValueError(
                f"blink_duration_ms must be finite and >= 0.0, got {self.blink_duration_ms}."
            )

        # 8. PERCLOS
        if self.perclos is not None:
            if not isinstance(self.perclos, (int, float)) or isinstance(self.perclos, bool):
                raise TypeError(f"perclos must be numeric or None, got {type(self.perclos).__name__}.")
            if not math.isfinite(self.perclos):
                raise ValueError(f"perclos must be finite or None, got {self.perclos}.")
            if not (0.0 <= self.perclos <= 1.0):
                raise ValueError(f"perclos must be within [0.0, 1.0], got {self.perclos}.")

        # 9. FPS
        if self.fps is not None:
            if not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool):
                raise TypeError(f"fps must be numeric or None, got {type(self.fps).__name__}.")
            if not math.isfinite(self.fps):
                raise ValueError(f"fps must be finite or None, got {self.fps}.")
            if self.fps <= 0.0:
                raise ValueError(f"fps must be positive, got {self.fps}.")

        # 10. Label consistency
        if not isinstance(self.label, str):
            raise TypeError(f"label must be a string, got {type(self.label).__name__}.")
        if not isinstance(self.label_id, int) or isinstance(self.label_id, bool):
            raise TypeError(f"label_id must be an integer, got {type(self.label_id).__name__}.")
        validate_label_pair(self.label, self.label_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert SampleRecord to dictionary preserving canonical field order."""
        return {
            "sample_id": self.sample_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "frame_index": self.frame_index,
            "timestamp_ms": float(self.timestamp_ms),
            "frame_delta_ms": float(self.frame_delta_ms) if self.frame_delta_ms is not None else None,
            "face_detected": self.face_detected,
            "landmarks_valid": self.landmarks_valid,
            "ear_left": float(self.ear_left) if self.ear_left is not None else None,
            "ear_right": float(self.ear_right) if self.ear_right is not None else None,
            "ear_avg": float(self.ear_avg) if self.ear_avg is not None else None,
            "is_eye_closed": self.is_eye_closed,
            "blink_count": self.blink_count,
            "blink_duration_ms": float(self.blink_duration_ms),
            "perclos": float(self.perclos) if self.perclos is not None else None,
            "fps": float(self.fps) if self.fps is not None else None,
            "label": self.label,
            "label_id": self.label_id,
        }

    def to_json(self) -> str:
        """Serialize SampleRecord to JSON string without NaN/Infinity."""
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleRecord:
        """Instantiate SampleRecord from dictionary with validation."""
        if not isinstance(data, dict):
            raise TypeError(f"SampleRecord data must be a dictionary, got {type(data).__name__}.")
        record = cls(
            sample_id=data.get("sample_id", ""),
            session_id=data.get("session_id", ""),
            subject_id=data.get("subject_id", ""),
            frame_index=data.get("frame_index", -1),
            timestamp_ms=data.get("timestamp_ms", -1.0),
            frame_delta_ms=data.get("frame_delta_ms"),
            face_detected=data.get("face_detected", False),
            landmarks_valid=data.get("landmarks_valid", False),
            ear_left=data.get("ear_left"),
            ear_right=data.get("ear_right"),
            ear_avg=data.get("ear_avg"),
            is_eye_closed=data.get("is_eye_closed"),
            blink_count=data.get("blink_count", -1),
            blink_duration_ms=data.get("blink_duration_ms", -1.0),
            perclos=data.get("perclos"),
            fps=data.get("fps"),
            label=data.get("label", ""),
            label_id=data.get("label_id", -99),
        )
        record.validate()
        return record

    @classmethod
    def from_json(cls, json_str: str) -> SampleRecord:
        """Instantiate SampleRecord from JSON string."""
        return cls.from_dict(json.loads(json_str))
