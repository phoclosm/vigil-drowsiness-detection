"""Validated, metadata-only manifests for approved M2 development data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vigil.ml.data.schema import SampleRecord

MANIFEST_SPLITS = frozenset({"train", "validation", "held_out"})
MANIFEST_STRATEGIES = frozenset({"subject", "session"})
CANONICAL_LABEL_NAMES = frozenset({"ACTIVE", "DROWSY", "SLEEPING", "AMBIGUOUS"})


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class ManifestRecording:
    """One approved recording reference; raw media remain outside Git."""

    recording_id: str
    path: str
    subject_id: str
    session_id: str
    split: str

    def __post_init__(self) -> None:
        for name in ("recording_id", "path", "subject_id", "session_id"):
            _require_text(name, getattr(self, name))
        if self.split not in MANIFEST_SPLITS:
            raise ValueError(f"split must be one of {sorted(MANIFEST_SPLITS)}.")
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("path must be a relative path inside the manifest root.")

    def to_dict(self) -> dict[str, str]:
        return {
            "recording_id": self.recording_id,
            "path": self.path,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestRecording:
        if not isinstance(data, dict):
            raise TypeError("Each manifest recording must be an object.")
        return cls(
            recording_id=data.get("recording_id", ""),
            path=data.get("path", ""),
            subject_id=data.get("subject_id", ""),
            session_id=data.get("session_id", ""),
            split=data.get("split", ""),
        )


@dataclass(frozen=True)
class ApprovedDataManifest:
    """Dataset identity, authorization evidence, label mapping, and group splits."""

    dataset_name: str
    dataset_version: str
    source_uri: str
    authorization_evidence: str
    consent_evidence: str
    label_mapping: dict[str, str]
    split_strategy: str
    recordings: tuple[ManifestRecording, ...]

    def __post_init__(self) -> None:
        for name in (
            "dataset_name",
            "dataset_version",
            "source_uri",
            "authorization_evidence",
            "consent_evidence",
        ):
            _require_text(name, getattr(self, name))
        if self.split_strategy not in MANIFEST_STRATEGIES:
            raise ValueError(f"split_strategy must be one of {sorted(MANIFEST_STRATEGIES)}.")
        if set(self.label_mapping) != CANONICAL_LABEL_NAMES:
            raise ValueError(
                "label_mapping must define exactly ACTIVE, DROWSY, SLEEPING, and AMBIGUOUS."
            )
        for name, source_label in self.label_mapping.items():
            _require_text(f"label_mapping[{name}]", source_label)
        if not self.recordings:
            raise ValueError("recordings must contain at least one approved recording reference.")
        self.validate_group_assignments()

    def validate_group_assignments(self) -> None:
        seen_recordings: set[str] = set()
        seen_paths: set[str] = set()
        seen_sessions: set[str] = set()
        session_subjects: dict[str, str] = {}
        group_splits: dict[str, set[str]] = {}
        for recording in self.recordings:
            if recording.recording_id in seen_recordings:
                raise ValueError(f"Duplicate recording_id '{recording.recording_id}'.")
            if recording.path in seen_paths:
                raise ValueError(f"Duplicate manifest path '{recording.path}'.")
            if recording.session_id in seen_sessions:
                raise ValueError(
                    f"Session '{recording.session_id}' must have exactly one manifest recording."
                )
            seen_recordings.add(recording.recording_id)
            seen_paths.add(recording.path)
            seen_sessions.add(recording.session_id)
            previous_subject = session_subjects.setdefault(
                recording.session_id, recording.subject_id
            )
            if previous_subject != recording.subject_id:
                raise ValueError(
                    f"Session '{recording.session_id}' maps to multiple subjects."
                )
            group = (
                recording.subject_id
                if self.split_strategy == "subject"
                else recording.session_id
            )
            group_splits.setdefault(group, set()).add(recording.split)
        overlapping = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
        if overlapping:
            unit = "subjects" if self.split_strategy == "subject" else "sessions"
            raise ValueError(f"{unit} assigned to multiple manifest splits: {overlapping}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_uri": self.source_uri,
            "authorization_evidence": self.authorization_evidence,
            "consent_evidence": self.consent_evidence,
            "label_mapping": dict(sorted(self.label_mapping.items())),
            "split_strategy": self.split_strategy,
            "recordings": [recording.to_dict() for recording in self.recordings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovedDataManifest:
        if not isinstance(data, dict):
            raise TypeError("Manifest must be a JSON object.")
        recordings = data.get("recordings")
        if not isinstance(recordings, list):
            raise TypeError("Manifest recordings must be a list.")
        return cls(
            dataset_name=data.get("dataset_name", ""),
            dataset_version=data.get("dataset_version", ""),
            source_uri=data.get("source_uri", ""),
            authorization_evidence=data.get("authorization_evidence", ""),
            consent_evidence=data.get("consent_evidence", ""),
            label_mapping=data.get("label_mapping", {}),
            split_strategy=data.get("split_strategy", ""),
            recordings=tuple(ManifestRecording.from_dict(item) for item in recordings),
        )

    @classmethod
    def from_json(cls, text: str) -> ApprovedDataManifest:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path | str) -> ApprovedDataManifest:
        manifest_path = Path(path)
        try:
            return cls.from_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid data manifest '{manifest_path}': {exc}") from exc

    def recordings_for_split(self, split: str) -> tuple[ManifestRecording, ...]:
        if split not in MANIFEST_SPLITS:
            raise ValueError(f"split must be one of {sorted(MANIFEST_SPLITS)}.")
        return tuple(recording for recording in self.recordings if recording.split == split)

    def load_records(self, root: Path | str, split: str) -> list[SampleRecord]:
        """Load validated JSONL records for one manifest split."""
        root_path = Path(root).resolve()
        records: list[SampleRecord] = []
        for recording in self.recordings_for_split(split):
            path = (root_path / recording.path).resolve()
            if root_path not in path.parents:
                raise ValueError(f"Manifest path escapes root: '{recording.path}'.")
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                raise ValueError(f"Cannot read manifest recording '{path}': {exc}") from exc
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = SampleRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid manifest record '{path}' at line {line_number}: {exc}"
                    ) from exc
                if record.subject_id != recording.subject_id or record.session_id != recording.session_id:
                    raise ValueError(
                        f"Manifest identity mismatch for '{path}' at line {line_number}."
                    )
                records.append(record)
        return records

    def load_development_records(
        self, root: Path | str
    ) -> tuple[list[SampleRecord], list[SampleRecord]]:
        """Load only manifest-assigned train and validation records."""
        train_records = self.load_records(root, "train")
        validation_records = self.load_records(root, "validation")
        if not train_records or not validation_records:
            raise ValueError(
                "Manifest development partitions must both contain records; "
                "held-out records are excluded."
            )
        return train_records, validation_records
