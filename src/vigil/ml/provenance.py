"""Validation helpers for model-development group provenance."""

from __future__ import annotations

from typing import Any

VALID_PROVENANCE_STRATEGIES = frozenset({"subject", "session"})
PROVENANCE_GROUP_KEYS = (
    "train_subjects",
    "val_subjects",
    "train_sessions",
    "val_sessions",
)


def validate_training_provenance(provenance: Any) -> dict[str, Any]:
    """Validate and normalize the explicit provenance schema used for evaluation."""
    if not isinstance(provenance, dict):
        raise TypeError("Training provenance must be a dictionary.")

    strategy = provenance.get("strategy")
    if strategy not in VALID_PROVENANCE_STRATEGIES:
        raise ValueError(
            "Training provenance strategy must be 'subject' or 'session'."
        )

    normalized: dict[str, Any] = {"strategy": strategy}
    for key in PROVENANCE_GROUP_KEYS:
        values = provenance.get(key)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(
                f"Training provenance field '{key}' must be a non-empty list of strings."
            )
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(
                f"Training provenance field '{key}' must contain non-empty strings."
            )
        normalized[key] = sorted(set(values))

    train_subjects = set(normalized["train_subjects"])
    val_subjects = set(normalized["val_subjects"])
    train_sessions = set(normalized["train_sessions"])
    val_sessions = set(normalized["val_sessions"])
    if strategy == "subject" and train_subjects & val_subjects:
        raise ValueError("Training provenance subject groups overlap.")
    if train_sessions & val_sessions:
        raise ValueError("Training provenance session groups overlap.")

    return normalized


def validate_evaluation_group_separation(
    records: list[Any],
    provenance: dict[str, Any],
) -> None:
    """Reject evaluation records that overlap model-development groups."""
    eval_subjects = {record.subject_id for record in records}
    eval_sessions = {record.session_id for record in records}
    development_subjects = (
        set(provenance["train_subjects"]) | set(provenance["val_subjects"])
    )
    development_sessions = (
        set(provenance["train_sessions"]) | set(provenance["val_sessions"])
    )

    if provenance["strategy"] == "subject":
        overlapping_subjects = sorted(eval_subjects & development_subjects)
        overlapping_sessions = sorted(eval_sessions & development_sessions)
        if overlapping_subjects or overlapping_sessions:
            raise ValueError(
                "Evaluation groups overlap model-development provenance for subject strategy: "
                f"subjects={overlapping_subjects}, sessions={overlapping_sessions}."
            )
    elif provenance["strategy"] == "session":
        overlapping_sessions = sorted(eval_sessions & development_sessions)
        if overlapping_sessions:
            raise ValueError(
                "Evaluation sessions overlap model-development provenance for session strategy: "
                f"sessions={overlapping_sessions}."
            )
