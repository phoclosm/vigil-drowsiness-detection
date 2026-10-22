"""Regression tests for evaluation input-quality auditing."""

from __future__ import annotations

import pytest

from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.config import WindowConfig
from vigil.ml.dataset.windowing import extract_windows
from vigil.ml.evaluation.input_audit import audit_evaluation_input


def make_record(
    frame_index: int,
    *,
    session_id: str = "session_001",
    landmarks_valid: bool = True,
    is_eye_closed: bool | None = False,
    ear_avg: float | None = 0.3,
    blink_duration_ms: float = 0.0,
    perclos: float | None = 0.1,
    label_id: int = 0,
) -> SampleRecord:
    if not landmarks_valid:
        ear_avg = None
        is_eye_closed = None
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id="subject_001",
        frame_index=frame_index,
        timestamp_ms=float(frame_index * 33),
        frame_delta_ms=None if frame_index == 0 else 33.0,
        face_detected=landmarks_valid,
        landmarks_valid=landmarks_valid,
        ear_left=ear_avg,
        ear_right=ear_avg,
        ear_avg=ear_avg,
        is_eye_closed=is_eye_closed,
        blink_count=0,
        blink_duration_ms=blink_duration_ms,
        perclos=perclos,
        fps=30.0,
        label="AMBIGUOUS" if label_id == -1 else "ACTIVE",
        label_id=label_id,
    )


def test_audit_counts_validity_and_false_eye_closure_as_valid() -> None:
    records = [
        make_record(0, is_eye_closed=False),
        make_record(1, landmarks_valid=False, blink_duration_ms=0.0, perclos=None),
    ]

    audit = audit_evaluation_input(records)

    assert audit.record_count == 2
    assert audit.feature_coverage["ear_avg"].valid_count == 1
    assert audit.feature_coverage["is_eye_closed"].valid_count == 1
    assert audit.feature_coverage["blink_duration_ms"].valid_count == 2
    assert audit.feature_coverage["perclos"].missing_count == 1


def test_audit_accounts_for_window_exclusions_and_segments() -> None:
    records = [
        *[make_record(i, label_id=-1 if i == 30 else 0) for i in range(60)],
        *[make_record(i) for i in range(60, 75)],
        *[make_record(i, session_id="session_002") for i in range(10)],
    ]

    audit = audit_evaluation_input(records, WindowConfig(window_length=60, stride=15))

    assert audit.windows.candidate_count == 2
    assert audit.windows.accepted_count == 0
    assert audit.windows.exclusion_reasons == {"ambiguous_label": 2}
    assert audit.windows.insufficient_history_segments == 1
    assert audit.windows.session_partition_count == 2


def test_audit_marks_fully_missing_windows_without_excluding_them() -> None:
    records = [make_record(i, landmarks_valid=False, perclos=None) for i in range(60)]

    audit = audit_evaluation_input(records)

    assert audit.windows.candidate_count == 1
    assert audit.windows.accepted_count == 1
    assert audit.windows.feature_unusable_windows == 1


def test_audit_serialization_is_deterministic() -> None:
    records = [make_record(i) for i in range(60)]
    audit = audit_evaluation_input(records, is_synthetic=True)

    assert audit.to_json() == audit.to_json()
    assert audit.to_dict()["is_synthetic"] is True
    markdown = audit.to_markdown()
    assert "Candidate Windows" in markdown
    assert "`ear_avg`" in markdown


def test_audit_uses_same_eligibility_as_extractor_across_boundaries() -> None:
    cases = [
        [make_record(i) for i in range(80)],
        [make_record(i, label_id=-1 if i == 30 else 0) for i in range(80)],
        [
            *[make_record(i) for i in range(70)],
            *[make_record(i, session_id="session_002") for i in range(70)],
        ],
        [make_record(i) for i in range(59)],
        [
            *[make_record(i) for i in range(60)],
            *[make_record(i) for i in range(62, 122)],
        ],
    ]

    for records in cases:
        config = WindowConfig(window_length=60, stride=15)
        audit = audit_evaluation_input(records, config)
        windows = extract_windows(records, config)
        assert audit.windows.accepted_count == len(windows)
        assert audit.windows.candidate_count == (
            audit.windows.accepted_count + audit.windows.excluded_count
        )
        assert sum(audit.windows.exclusion_reasons.values()) == audit.windows.excluded_count


def test_unprovenanced_audit_serializes_unknown_not_false() -> None:
    audit = audit_evaluation_input([make_record(i) for i in range(60)])

    assert audit.to_dict()["is_synthetic"] == "unknown"
    assert "Synthetic Fixtures**: unknown" in audit.to_markdown()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"is_synthetic": "yes"}, TypeError),
        ({"dataset_name": " "}, ValueError),
        ({"dataset_release": ""}, ValueError),
        ({"manifest_id": " "}, ValueError),
        ({"manifest_exclusions": True}, TypeError),
        ({"manifest_exclusions": -1}, ValueError),
        ({"manifest_exclusions": 0}, ValueError),
    ],
)
def test_audit_rejects_invalid_or_unsupported_metadata(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        audit_evaluation_input([], **kwargs)  # type: ignore[arg-type]
