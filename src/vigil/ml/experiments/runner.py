"""Reproducible development experiment harness for M2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vigil.ml.data.manifest import ApprovedDataManifest
from vigil.ml.dataset.splitter import DatasetGroupSplitResult
from vigil.ml.training.config import TrainingConfig
from vigil.ml.training.trainer import (
    TrainingResult,
    train_temporal_classifier_on_partitions,
)


def _digest_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    """Resolve the active checkout root or fail instead of guessing."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Cannot determine the Git worktree root; refusing to validate artifact safety."
        ) from exc
    root_text = result.stdout.strip()
    if not root_text:
        raise RuntimeError(
            "Git returned an empty worktree root; refusing to validate artifact safety."
        )
    return Path(root_text).resolve()


def _validate_artifact_directory(artifact_dir: str) -> Path:
    artifact_path = Path(artifact_dir).resolve()
    if not artifact_path.exists() or not artifact_path.is_dir():
        raise ValueError(
            f"artifact_dir must be an existing directory reserved for generated artifacts: "
            f"'{artifact_dir}'."
        )
    repository_root = _repository_root()
    if artifact_path == repository_root or repository_root in artifact_path.parents:
        raise ValueError(
            "artifact_dir must resolve outside the Git worktree; generated checkpoints "
            "must remain outside Git."
        )
    return artifact_path


@dataclass(frozen=True)
class DevelopmentExperiment:
    """Auditable development-only experiment metadata and validation metrics."""

    git_sha: str
    manifest_name: str
    manifest_version: str
    seed: int
    train_subjects: list[str]
    val_subjects: list[str]
    train_sessions: list[str]
    val_sessions: list[str]
    model_config: dict[str, Any]
    optimizer: str
    learning_rate: float
    weight_decay: float
    epoch_count: int
    best_checkpoint_epoch: int
    metrics: dict[str, Any]
    artifact_sha256: str
    is_synthetic: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "manifest": {
                "dataset_name": self.manifest_name,
                "dataset_version": self.manifest_version,
            },
            "seed": self.seed,
            "split": {
                "train_subjects": self.train_subjects,
                "val_subjects": self.val_subjects,
                "train_sessions": self.train_sessions,
                "val_sessions": self.val_sessions,
            },
            "model_config": self.model_config,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epoch_count": self.epoch_count,
            "best_checkpoint_epoch": self.best_checkpoint_epoch,
            "metrics": self.metrics,
            "artifact_sha256": self.artifact_sha256,
            "is_synthetic": self.is_synthetic if self.is_synthetic is not None else "unknown",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def run_development_experiment(
    manifest: ApprovedDataManifest,
    *,
    data_root: str,
    artifact_dir: str,
    git_sha: str,
    config: TrainingConfig | None = None,
    is_synthetic: bool | None = None,
) -> tuple[TrainingResult, DevelopmentExperiment]:
    """Load manifest-approved partitions, train, and record reproducibility metadata."""
    if not isinstance(git_sha, str) or not git_sha.strip():
        raise ValueError("git_sha must be a non-empty string.")
    artifact_path = _validate_artifact_directory(artifact_dir)
    if is_synthetic is not None and not isinstance(is_synthetic, bool):
        raise TypeError("is_synthetic must be a boolean or None.")
    manifest.validate_group_assignments()
    train_records, val_records = manifest.load_development_records(data_root)
    effective_config = config or TrainingConfig()
    if effective_config.output_dir is not None:
        raise ValueError(
            "Do not set TrainingConfig.output_dir for the development harness; "
            "use artifact_dir."
        )
    effective_config = TrainingConfig(
        batch_size=effective_config.batch_size,
        learning_rate=effective_config.learning_rate,
        max_epochs=effective_config.max_epochs,
        weight_decay=effective_config.weight_decay,
        seed=effective_config.seed,
        train_ratio=effective_config.train_ratio,
        hidden_dim=effective_config.hidden_dim,
        output_dir=str(artifact_path),
    )
    train_subjects = sorted({record.subject_id for record in train_records})
    val_subjects = sorted({record.subject_id for record in val_records})
    train_sessions = sorted({record.session_id for record in train_records})
    val_sessions = sorted({record.session_id for record in val_records})
    split_result = DatasetGroupSplitResult(
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        strategy=manifest.split_strategy,
        subject_count=len(set(train_subjects + val_subjects)),
        session_count=len(set(train_sessions + val_sessions)),
        limitation_note=None,
    )
    training_result = train_temporal_classifier_on_partitions(
        train_records,
        val_records,
        split_result,
        effective_config,
    )
    checkpoint_path = training_result.checkpoint_path
    if checkpoint_path is None:
        raise RuntimeError("Training completed without writing the required checkpoint.")
    experiment = DevelopmentExperiment(
        git_sha=git_sha.strip(),
        manifest_name=manifest.dataset_name,
        manifest_version=manifest.dataset_version,
        seed=effective_config.seed,
        train_subjects=list(training_result.split_result.train_subjects),
        val_subjects=list(training_result.split_result.val_subjects),
        train_sessions=list(training_result.split_result.train_sessions),
        val_sessions=list(training_result.split_result.val_sessions),
        model_config={
            "name": "GRU",
            "input_dim": 8,
            "hidden_dim": effective_config.hidden_dim,
        },
        optimizer="Adam",
        learning_rate=effective_config.learning_rate,
        weight_decay=effective_config.weight_decay,
        epoch_count=len(training_result.train_losses),
        best_checkpoint_epoch=training_result.best_epoch,
        metrics=training_result.evaluation_report.to_dict(),
        artifact_sha256=_digest_file(checkpoint_path),
        is_synthetic=is_synthetic,
    )
    return training_result, experiment
