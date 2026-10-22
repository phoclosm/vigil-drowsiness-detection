"""Deterministic evaluation-input quality auditing for Vigil M2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS, WindowConfig
from vigil.ml.dataset.windowing import (
    build_sample_feature_vector,
    enumerate_window_candidates,
)

AUDITED_FEATURE_CHANNELS = LOCKED_FEATURE_CHANNELS[:4]


def validate_audit_metadata(
    *,
    dataset_name: str | None,
    dataset_release: str | None,
    manifest_id: str | None,
    manifest_exclusions: int | None,
) -> None:
    """Validate explicitly supplied metadata without asserting manifest validation."""
    for name, value in (
        ("dataset_name", dataset_name),
        ("dataset_release", dataset_release),
        ("manifest_id", manifest_id),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string when supplied.")
    if manifest_exclusions is not None:
        if isinstance(manifest_exclusions, bool) or not isinstance(manifest_exclusions, int):
            raise TypeError("manifest_exclusions must be a non-negative integer when supplied.")
        if manifest_exclusions < 0:
            raise ValueError("manifest_exclusions must be a non-negative integer when supplied.")
        raise ValueError(
            "manifest_exclusions is unsupported until a validated manifest input is provided."
        )


@dataclass(frozen=True)
class FeatureCoverage:
    """Observed and missing counts for one locked feature channel."""

    valid_count: int
    missing_count: int
    valid_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_count": self.valid_count,
            "missing_count": self.missing_count,
            "valid_rate": self.valid_rate,
        }


@dataclass(frozen=True)
class WindowAudit:
    """Auditable candidate and accepted-window accounting."""

    candidate_count: int
    accepted_count: int
    excluded_count: int
    exclusion_reasons: dict[str, int]
    insufficient_history_segments: int
    session_partition_count: int
    temporal_gap_breaks: int
    feature_unusable_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "excluded_count": self.excluded_count,
            "exclusion_reasons": dict(sorted(self.exclusion_reasons.items())),
            "insufficient_history_segments": self.insufficient_history_segments,
            "session_partition_count": self.session_partition_count,
            "temporal_gap_breaks": self.temporal_gap_breaks,
            "feature_unusable_windows": self.feature_unusable_windows,
        }


@dataclass(frozen=True)
class EvaluationInputAudit:
    """Structured quality report for records supplied to independent evaluation."""

    record_count: int
    unique_subjects: list[str]
    unique_sessions: list[str]
    recording_count: int | None
    feature_coverage: dict[str, FeatureCoverage]
    windows: WindowAudit
    manifest_exclusions: int | None = None
    malformed_input_count: int = 0
    is_synthetic: bool | None = None
    dataset_name: str | None = None
    dataset_release: str | None = None
    manifest_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "record_count": self.record_count,
            "unique_subjects": self.unique_subjects,
            "unique_sessions": self.unique_sessions,
            "recording_count": self.recording_count,
            "feature_coverage": {
                name: self.feature_coverage[name].to_dict()
                for name in AUDITED_FEATURE_CHANNELS
                if name in self.feature_coverage
            },
            "windows": self.windows.to_dict(),
            "manifest_exclusions": self.manifest_exclusions,
            "malformed_input_count": self.malformed_input_count,
            "is_synthetic": self.is_synthetic if self.is_synthetic is not None else "unknown",
        }
        optional = {
            "dataset_name": self.dataset_name,
            "dataset_release": self.dataset_release,
            "manifest_id": self.manifest_id,
        }
        result.update({key: value for key, value in optional.items() if value is not None})
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "### M2 Evaluation Input-Quality Audit",
            "",
            f"- **Records**: {self.record_count}",
            f"- **Unique Subjects**: {len(self.unique_subjects)}",
            f"- **Unique Sessions**: {len(self.unique_sessions)}",
            f"- **Recordings**: {self.recording_count if self.recording_count is not None else 'not supplied'}",
            f"- **Manifest Exclusions**: {self.manifest_exclusions if self.manifest_exclusions is not None else 'not supplied'}",
            f"- **Malformed Input Rows**: {self.malformed_input_count}",
            f"- **Synthetic Fixtures**: {str(self.is_synthetic).lower() if self.is_synthetic is not None else 'unknown'}",
            "",
            "#### Feature Coverage",
            "| Channel | Valid | Missing | Valid Rate |",
            "|---|---:|---:|---:|",
        ]
        for name in AUDITED_FEATURE_CHANNELS:
            coverage = self.feature_coverage[name]
            lines.append(
                f"| `{name}` | {coverage.valid_count} | {coverage.missing_count} | "
                f"{coverage.valid_rate:.4f} |"
            )
        lines.extend([
            "",
            "#### Window Accounting",
            f"- **Candidate Windows**: {self.windows.candidate_count}",
            f"- **Accepted Windows**: {self.windows.accepted_count}",
            f"- **Excluded Candidate Windows**: {self.windows.excluded_count}",
            f"- **Insufficient-History Segments**: {self.windows.insufficient_history_segments}",
            f"- **Session Partitions**: {self.windows.session_partition_count}",
            f"- **Temporal Gap Breaks**: {self.windows.temporal_gap_breaks}",
            f"- **Feature-Unusable Window Flags**: {self.windows.feature_unusable_windows}",
            "",
            "##### Candidate Exclusion Reasons",
            "| Reason | Count |",
            "|---|---:|",
        ])
        for reason, count in sorted(self.windows.exclusion_reasons.items()):
            lines.append(f"| `{reason}` | {count} |")
        lines.extend([
            "",
            "Feature-unusable windows are quality flags, not exclusions: the existing",
            "pipeline permits missing ocular values and preserves their validity masks.",
            "Malformed JSONL/schema rows fail before this audit and are not counted as",
            "ordinary exclusions. Intentional manifest exclusions require a supplied",
            "validated manifest and are not inferred here.",
        ])
        return "\n".join(lines)


def audit_evaluation_input(
    records: list[SampleRecord],
    config: WindowConfig | None = None,
    *,
    is_synthetic: bool | None = None,
    dataset_name: str | None = None,
    dataset_release: str | None = None,
    manifest_id: str | None = None,
    manifest_exclusions: int | None = None,
    recording_ids: set[str] | None = None,
) -> EvaluationInputAudit:
    """Audit validated records using the same segmentation rules as extraction."""
    if is_synthetic is not None and not isinstance(is_synthetic, bool):
        raise TypeError("is_synthetic must be a boolean or None.")
    validate_audit_metadata(
        dataset_name=dataset_name,
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        manifest_exclusions=manifest_exclusions,
    )
    if config is None:
        config = WindowConfig()

    coverage_counts = {name: [0, 0] for name in AUDITED_FEATURE_CHANNELS}
    for record in records:
        features = build_sample_feature_vector(record)
        for name, value in zip(AUDITED_FEATURE_CHANNELS, features[4:]):
            coverage_counts[name][0 if value == 1.0 else 1] += 1

    candidates, insufficient_segments, temporal_gap_breaks, session_partition_count = (
        enumerate_window_candidates(records, config)
    )
    candidate_count = len(candidates)
    accepted_count = 0
    feature_unusable_windows = 0
    exclusion_reasons: dict[str, int] = {}

    for candidate in candidates:
        window_features = [
            build_sample_feature_vector(record)
            for record in candidate.records
        ]
        if all(all(row[index] == 0.0 for row in window_features) for index in (4, 5, 7)):
            feature_unusable_windows += 1
        if candidate.rejection_reason is not None:
            exclusion_reasons[candidate.rejection_reason] = (
                exclusion_reasons.get(candidate.rejection_reason, 0) + 1
            )
            continue
        accepted_count += 1

    return EvaluationInputAudit(
        record_count=len(records),
        unique_subjects=sorted({record.subject_id for record in records}),
        unique_sessions=sorted({record.session_id for record in records}),
        recording_count=len(recording_ids) if recording_ids is not None else None,
        feature_coverage={
            name: FeatureCoverage(
                valid_count=counts[0],
                missing_count=counts[1],
                valid_rate=counts[0] / len(records) if records else 0.0,
            )
            for name, counts in coverage_counts.items()
        },
        windows=WindowAudit(
            candidate_count=candidate_count,
            accepted_count=accepted_count,
            excluded_count=candidate_count - accepted_count,
            exclusion_reasons=exclusion_reasons,
            insufficient_history_segments=insufficient_segments,
            session_partition_count=session_partition_count,
            temporal_gap_breaks=temporal_gap_breaks,
            feature_unusable_windows=feature_unusable_windows,
        ),
        manifest_exclusions=manifest_exclusions,
        is_synthetic=is_synthetic,
        dataset_name=dataset_name,
        dataset_release=dataset_release,
        manifest_id=manifest_id,
    )
