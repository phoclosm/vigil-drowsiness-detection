"""Leakage-safe grouped record partitioning for Vigil M2 datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from vigil.ml.data.schema import SampleRecord


@dataclass(frozen=True)
class DatasetGroupSplitResult:
    """Metadata describing the outcome of grouped dataset partitioning.

    Attributes:
        train_sessions: Sorted list of session IDs assigned to Training.
        val_sessions: Sorted list of session IDs assigned to Validation.
        train_subjects: Sorted list of subject IDs assigned to Training.
        val_subjects: Sorted list of subject IDs assigned to Validation.
        strategy: 'subject' (multi-subject primary), 'session' (single-subject fallback), or 'none'.
        subject_count: Total unique subjects detected.
        session_count: Total unique sessions detected.
        limitation_note: Explanatory note when partitioning is constrained.
    """

    train_sessions: list[str]
    val_sessions: list[str]
    train_subjects: list[str]
    val_subjects: list[str]
    strategy: str
    subject_count: int
    session_count: int
    limitation_note: str | None


def split_records_by_group(
    records: list[SampleRecord],
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[list[SampleRecord], list[SampleRecord], DatasetGroupSplitResult]:
    """Deterministically partition records into train and validation sets using grouped splitting.

    Enforces M2 Evaluation Protocol:
        - Primary: group by subject_id. All sessions for a subject remain in one partition.
        - Fallback: when only one subject is available, group by session_id.
        - Never split individual rows or windows randomly across partitions.
        - If fewer than 2 groups exist, emit empty validation set and report limitation.

    Args:
        records: List of SampleRecords to partition.
        train_ratio: Target proportion assigned to training (default 0.75).
        seed: Deterministic random seed (default 42).

    Returns:
        A tuple of (train_records, val_records, split_result).
    """
    if not records:
        split_result = DatasetGroupSplitResult(
            train_sessions=[],
            val_sessions=[],
            train_subjects=[],
            val_subjects=[],
            strategy="none",
            subject_count=0,
            session_count=0,
            limitation_note="No records provided for splitting.",
        )
        return ([], [], split_result)

    # Index records by subject and session
    subj_to_sessions: dict[str, set[str]] = defaultdict(set)
    session_to_records: dict[str, list[SampleRecord]] = defaultdict(list)
    session_to_subj: dict[str, str] = {}

    for r in records:
        subj_to_sessions[r.subject_id].add(r.session_id)
        session_to_records[r.session_id].append(r)
        session_to_subj[r.session_id] = r.subject_id

    unique_subjects = sorted(subj_to_sessions.keys())
    unique_sessions = sorted(session_to_records.keys())
    subject_count = len(unique_subjects)
    session_count = len(unique_sessions)

    rng = random.Random(seed)

    # Strategy 1: Multi-subject -> Subject-level grouping
    if subject_count > 1:
        shuffled_subjs = list(unique_subjects)
        rng.shuffle(shuffled_subjs)

        n_train_subjs = max(1, min(subject_count - 1, round(subject_count * train_ratio)))
        train_subjs = sorted(shuffled_subjs[:n_train_subjs])
        val_subjs = sorted(shuffled_subjs[n_train_subjs:])

        train_sessions = sorted(
            s for subj in train_subjs for s in sorted(subj_to_sessions[subj])
        )
        val_sessions = sorted(
            s for subj in val_subjs for s in sorted(subj_to_sessions[subj])
        )

        train_records = [r for s in train_sessions for r in session_to_records[s]]
        val_records = [r for s in val_sessions for r in session_to_records[s]]

        split_result = DatasetGroupSplitResult(
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            train_subjects=train_subjs,
            val_subjects=val_subjs,
            strategy="subject",
            subject_count=subject_count,
            session_count=session_count,
            limitation_note=None,
        )
        return (train_records, val_records, split_result)

    # Strategy 2: Single-subject -> Fall back to session-level grouping
    if session_count > 1:
        shuffled_sessions = list(unique_sessions)
        rng.shuffle(shuffled_sessions)

        n_train = max(1, min(session_count - 1, round(session_count * train_ratio)))
        train_sessions = sorted(shuffled_sessions[:n_train])
        val_sessions = sorted(shuffled_sessions[n_train:])

        train_records = [r for s in train_sessions for r in session_to_records[s]]
        val_records = [r for s in val_sessions for r in session_to_records[s]]

        split_result = DatasetGroupSplitResult(
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            train_subjects=unique_subjects,
            val_subjects=unique_subjects,
            strategy="session",
            subject_count=1,
            session_count=session_count,
            limitation_note=(
                "Single subject detected across all sessions. Partitioning fell back to "
                "session-level grouping. Evaluation is restricted to session generalization "
                "and does not evaluate cross-subject generalization."
            ),
        )
        return (train_records, val_records, split_result)

    # Strategy 3: Single session and single subject -> Partition impossible
    split_result = DatasetGroupSplitResult(
        train_sessions=unique_sessions,
        val_sessions=[],
        train_subjects=unique_subjects,
        val_subjects=[],
        strategy="none",
        subject_count=1,
        session_count=1,
        limitation_note=(
            "Only one session is available. A valid held-out validation partition is impossible "
            "without violating temporal session integrity. Evaluation partition is unavailable."
        ),
    )
    return (list(records), [], split_result)
