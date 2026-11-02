"""Deterministic provenance and identity records for M2 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, Mapping

from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS, LOCKED_STRIDE, LOCKED_WINDOW_LENGTH
CandidateType = Literal["gru", "lstm", "tcn"]
EvaluationPhase = Literal["development_validation", "final_held_out"]
CANONICAL_LABEL_MAPPING = (("ACTIVE", 0), ("DROWSY", 1), ("SLEEPING", 2), ("AMBIGUOUS", -1))


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Provenance values must contain only finite floats.")
        return 0.0 if value == 0.0 else value
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported provenance value type: {type(value).__name__}.")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def record_digest(record: SampleRecord) -> str:
    record.validate()
    return sha256_digest(record.to_dict())


def record_set_digest(records: list[SampleRecord] | tuple[SampleRecord, ...]) -> str:
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record IDs detected.")
    digests = [record_digest(record) for record in records]
    return sha256_digest(sorted(digests))


def manifest_digest(manifest: Any) -> str:
    return sha256_digest(manifest.to_dict())


_compute_manifest_digest = manifest_digest



def label_mapping_digest() -> str:
    return sha256_digest(dict(CANONICAL_LABEL_MAPPING))


@dataclass(frozen=True)
class PartitionIdentity:
    """Immutable record and group membership for all evaluation partitions."""

    train_record_digest: str
    validation_record_digest: str
    held_out_record_digest: str
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    held_out_subjects: tuple[str, ...]
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    held_out_sessions: tuple[str, ...]
    split_strategy: str
    manifest_digest: str

    @classmethod
    def from_partitions(
        cls,
        train_records: list[SampleRecord],
        validation_records: list[SampleRecord],
        held_out_records: list[SampleRecord],
        split_strategy: str,
        *,
        manifest_digest: str = "",
    ) -> PartitionIdentity:
        partitions = (train_records, validation_records, held_out_records)
        if any(not records for records in partitions):
            raise ValueError("Train, validation, and held-out partitions must be non-empty.")
        all_ids = [record.sample_id for records in partitions for record in records]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Record IDs overlap between partitions.")
        groups = []
        for records in partitions:
            groups.append((
                tuple(sorted({record.subject_id for record in records})),
                tuple(sorted({record.session_id for record in records})),
            ))
        subject_sets = [set(group[0]) for group in groups]
        session_sets = [set(group[1]) for group in groups]
        if any(subject_sets[i] & subject_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Subject groups overlap between partitions.")
        if any(session_sets[i] & session_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Session groups overlap between partitions.")
        return cls(
            train_record_digest=record_set_digest(train_records),
            validation_record_digest=record_set_digest(validation_records),
            held_out_record_digest=record_set_digest(held_out_records),
            train_subjects=groups[0][0],
            validation_subjects=groups[1][0],
            held_out_subjects=groups[2][0],
            train_sessions=groups[0][1],
            validation_sessions=groups[1][1],
            held_out_sessions=groups[2][1],
            split_strategy=split_strategy,
            manifest_digest=manifest_digest,
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_record_digest": self.train_record_digest,
            "validation_record_digest": self.validation_record_digest,
            "held_out_record_digest": self.held_out_record_digest,
            "train_subjects": list(self.train_subjects),
            "validation_subjects": list(self.validation_subjects),
            "held_out_subjects": list(self.held_out_subjects),
            "train_sessions": list(self.train_sessions),
            "validation_sessions": list(self.validation_sessions),
            "held_out_sessions": list(self.held_out_sessions),
            "split_strategy": self.split_strategy,
            "manifest_digest": self.manifest_digest,
        }


@dataclass(frozen=True)
class TrustedPartition:
    """Validated records and identity for one complete approved partition set."""

    train_records: tuple[SampleRecord, ...]
    validation_records: tuple[SampleRecord, ...]
    held_out_records: tuple[SampleRecord, ...]
    identity: PartitionIdentity
    manifest_verified: bool = False

    @classmethod
    def from_partitions(
        cls,
        train_records: list[SampleRecord] | tuple[SampleRecord, ...],
        validation_records: list[SampleRecord] | tuple[SampleRecord, ...],
        held_out_records: list[SampleRecord] | tuple[SampleRecord, ...],
        split_strategy: str,
        *,
        manifest_digest: str,
        approved_manifest: Any | None = None,
    ) -> TrustedPartition:
        manifest_verified = approved_manifest is not None
        if approved_manifest is not None:
            approved_manifest.validate_group_assignments()
            if approved_manifest.split_strategy != split_strategy:
                raise ValueError("Partition split strategy does not match the approved manifest.")
            expected_manifest_digest = _compute_manifest_digest(approved_manifest)
            if expected_manifest_digest != manifest_digest:
                raise ValueError("Manifest digest does not match the approved manifest.")
            split_groups = {
                split: {
                    (recording.subject_id, recording.session_id)
                    for recording in approved_manifest.recordings_for_split(split)
                }
                for split in ("train", "validation", "held_out")
            }
            for split, records in zip(
                ("train", "validation", "held_out"),
                (train_records, validation_records, held_out_records),
            ):
                if any(
                    (record.subject_id, record.session_id) not in split_groups[split]
                    for record in records
                ):
                    raise ValueError(
                        f"Records do not belong to the approved manifest '{split}' split."
                    )
        partitions = tuple(tuple(records) for records in (
            train_records,
            validation_records,
            held_out_records,
        ))
        for records in partitions:
            for record in records:
                record.validate()
        identity = PartitionIdentity.from_partitions(
            list(partitions[0]),
            list(partitions[1]),
            list(partitions[2]),
            split_strategy,
            manifest_digest=manifest_digest,
        )
        return cls(
            train_records=partitions[0],
            validation_records=partitions[1],
            held_out_records=partitions[2],
            identity=identity,
            manifest_verified=manifest_verified,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: Any,
        *,
        train_records: list[SampleRecord] | tuple[SampleRecord, ...],
        validation_records: list[SampleRecord] | tuple[SampleRecord, ...],
        held_out_records: list[SampleRecord] | tuple[SampleRecord, ...],
    ) -> TrustedPartition:
        """Create a trusted partition from a validated ApprovedDataManifest."""
        return cls.from_partitions(
            train_records,
            validation_records,
            held_out_records,
            manifest.split_strategy,
            manifest_digest=_compute_manifest_digest(manifest),
            approved_manifest=manifest,
        )

    @property
    def digest(self) -> str:
        return self.identity.digest

    def records_for_phase(self, phase: EvaluationPhase) -> tuple[SampleRecord, ...]:
        if phase == "development_validation":
            return self.validation_records
        if phase == "final_held_out":
            return self.held_out_records
        raise ValueError(f"Unsupported evaluation phase: {phase}.")

    def contains_exact_records(
        self,
        records: list[SampleRecord] | tuple[SampleRecord, ...],
        phase: EvaluationPhase,
    ) -> bool:
        expected = self.records_for_phase(phase)
        return (
            record_set_digest(list(records)) == record_set_digest(list(expected))
            and {
                (record.sample_id, record.subject_id, record.session_id)
                for record in records
            }
            == {
                (record.sample_id, record.subject_id, record.session_id)
                for record in expected
            }
        )


@dataclass(frozen=True)
class EvaluationIdentity:
    dataset_name: str
    dataset_version: str
    manifest_digest: str
    label_mapping_digest: str
    feature_channels: tuple[str, ...] = LOCKED_FEATURE_CHANNELS
    window_length: int = LOCKED_WINDOW_LENGTH
    stride: int = LOCKED_STRIDE
    split_strategy: str = "subject"
    train_record_digest: str = ""
    validation_record_digest: str = ""
    held_out_record_digest: str = ""
    partition_digest: str = ""
    protocol_version: str = "m2-day3-v1"

    def __post_init__(self) -> None:
        if self.feature_channels != LOCKED_FEATURE_CHANNELS:
            raise ValueError("Evaluation identity feature channels do not match the locked contract.")
        if self.window_length != LOCKED_WINDOW_LENGTH or self.stride != LOCKED_STRIDE:
            raise ValueError("Evaluation identity window settings do not match the locked contract.")
        if self.split_strategy not in {"subject", "session"}:
            raise ValueError("split_strategy must be 'subject' or 'session'.")

    @classmethod
    def from_partition(
        cls,
        *,
        dataset_name: str,
        dataset_version: str,
        partition: TrustedPartition,
        label_mapping_digest: str,
        protocol_version: str = "m2-day3-v1",
    ) -> EvaluationIdentity:
        identity = partition.identity
        return cls(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            manifest_digest=identity.manifest_digest,
            label_mapping_digest=label_mapping_digest,
            split_strategy=identity.split_strategy,
            train_record_digest=identity.train_record_digest,
            validation_record_digest=identity.validation_record_digest,
            held_out_record_digest=identity.held_out_record_digest,
            partition_digest=partition.digest,
            protocol_version=protocol_version,
        )

    def matches_partition(self, partition: TrustedPartition) -> bool:
        expected = EvaluationIdentity.from_partition(
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            partition=partition,
            label_mapping_digest=self.label_mapping_digest,
            protocol_version=self.protocol_version,
        )
        return self == expected

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "manifest_digest": self.manifest_digest,
            "label_mapping_digest": self.label_mapping_digest,
            "feature_channels": list(self.feature_channels),
            "window_length": self.window_length,
            "stride": self.stride,
            "split_strategy": self.split_strategy,
            "train_record_digest": self.train_record_digest,
            "validation_record_digest": self.validation_record_digest,
            "held_out_record_digest": self.held_out_record_digest,
            "partition_digest": self.partition_digest,
            "protocol_version": self.protocol_version,
        }


@dataclass(frozen=True)
class NormalizerProvenance:
    fit_partition: Literal["train"]
    fit_record_digest: str
    fit_record_count: int
    fit_subjects: tuple[str, ...]
    fit_sessions: tuple[str, ...]
    feature_channels: tuple[str, ...]
    source_manifest_digest: str
    parameter_digest: str
    training_partition_digest: str = ""
    status: Literal["verified", "legacy_unverified"] = "verified"

    def __post_init__(self) -> None:
        if self.fit_partition != "train":
            raise ValueError("Normalizer provenance must be fitted on the train partition.")
        if self.feature_channels != LOCKED_FEATURE_CHANNELS:
            raise ValueError("Normalizer provenance channels do not match the locked contract.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_partition": self.fit_partition,
            "fit_record_digest": self.fit_record_digest,
            "fit_record_count": self.fit_record_count,
            "fit_subjects": list(self.fit_subjects),
            "fit_sessions": list(self.fit_sessions),
            "feature_channels": list(self.feature_channels),
            "source_manifest_digest": self.source_manifest_digest,
            "parameter_digest": self.parameter_digest,
            "training_partition_digest": self.training_partition_digest,
            "status": self.status,
        }


@dataclass(frozen=True)
class CandidateDescriptor:
    model_type: CandidateType
    architecture_name: str
    architecture_version: str
    config_digest: str
    config_json: str
    input_dim: int
    window_length: int
    num_classes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architecture_name": self.architecture_name,
            "architecture_version": self.architecture_version,
            "config_digest": self.config_digest,
            "config_json": self.config_json,
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "num_classes": self.num_classes,
        }


def describe_candidate(model: Any) -> CandidateDescriptor:
    from vigil.ml.models.gru import GRUTemporalClassifier
    from vigil.ml.models.lstm import LSTMTemporalClassifier
    from vigil.ml.models.tcn import TCNTemporalClassifier

    model_types = {
        GRUTemporalClassifier: "gru",
        LSTMTemporalClassifier: "lstm",
        TCNTemporalClassifier: "tcn",
    }
    model_type = next((name for cls, name in model_types.items() if isinstance(model, cls)), None)
    if model_type is None:
        raise TypeError("Unsupported temporal classifier; expected GRU, LSTM, or TCN.")
    config = getattr(model, "config", None)
    if config is None or not is_dataclass(config):
        raise TypeError("Temporal classifier must expose a validated dataclass config.")
    config_json = canonical_json(asdict(config))
    input_dim = getattr(config, "input_dim", None)
    num_classes = getattr(config, "num_classes", None)
    if input_dim != len(LOCKED_FEATURE_CHANNELS) or num_classes != 3:
        raise ValueError("Candidate does not satisfy the locked input/output contract.")
    return CandidateDescriptor(
        model_type=model_type,  # type: ignore[arg-type]
        architecture_name=type(model).__name__,
        architecture_version="v1",
        config_digest=sha256_digest(asdict(config)),
        config_json=config_json,
        input_dim=input_dim,
        window_length=LOCKED_WINDOW_LENGTH,
        num_classes=num_classes,
    )


@dataclass(frozen=True)
class CandidateTrainingProvenance:
    candidate: CandidateDescriptor
    seed: int
    batch_size: int
    max_epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    selection_metric: str
    selection_partition: Literal["development_validation"] = "development_validation"
    hyperparameter_search_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "selection_metric": self.selection_metric,
            "selection_partition": self.selection_partition,
            "hyperparameter_search_digest": self.hyperparameter_search_digest,
        }


@dataclass(frozen=True)
class SelectionFreezeRecord:
    frozen_candidate_type: CandidateType
    frozen_candidate_identity_digest: str
    selected_checkpoint_sha256: str
    selected_model_config_digest: str
    evaluation_identity_digest: str
    partition_digest: str
    selection_protocol_version: str
    authorization_evidence_ref: str
    frozen_at_utc: str
    code_git_sha: str

    def to_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "frozen_candidate_type": self.frozen_candidate_type,
                "frozen_candidate_identity_digest": self.frozen_candidate_identity_digest,
                "selected_checkpoint_sha256": self.selected_checkpoint_sha256,
                "selected_model_config_digest": self.selected_model_config_digest,
                "evaluation_identity_digest": self.evaluation_identity_digest,
                "partition_digest": self.partition_digest,
                "selection_protocol_version": self.selection_protocol_version,
                "authorization_evidence_ref": self.authorization_evidence_ref,
                "frozen_at_utc": self.frozen_at_utc,
                "code_git_sha": self.code_git_sha,
            }.items()
        }

    def validate(self) -> None:
        """Validate required freeze-record fields without implying human governance."""
        required = (
            self.frozen_candidate_identity_digest,
            self.selected_checkpoint_sha256,
            self.selected_model_config_digest,
            self.evaluation_identity_digest,
            self.partition_digest,
            self.selection_protocol_version,
            self.authorization_evidence_ref,
            self.frozen_at_utc,
            self.code_git_sha,
        )
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValueError("Selection freeze record has missing required fields.")
