from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vigil.ml.data.schema import SampleRecord  # noqa: E402
from vigil.ml.dataset.normalizer import FeatureNormalizer  # noqa: E402
from vigil.ml.evaluation.comparison import (  # noqa: E402
    IndependentEvaluationMetadata,
    compare_candidates,
)
from vigil.ml.provenance_types import (  # noqa: E402
    CandidateTrainingProvenance,
    EvaluationIdentity,
    PartitionIdentity,
    describe_candidate,
    label_mapping_digest,
    record_digest,
    record_set_digest,
    sha256_digest,
)
from vigil.ml.experiments.runner import DevelopmentExperiment  # noqa: E402
from vigil.ml.models import (  # noqa: E402
    GRUConfig,
    GRUTemporalClassifier,
    LSTMConfig,
    LSTMTemporalClassifier,
    TCNConfig,
    TCNTemporalClassifier,
)


def make_record(frame_index: int, session_id: str, subject_id: str) -> SampleRecord:
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=float(frame_index * 33.3),
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=True,
        landmarks_valid=True,
        ear_left=0.3,
        ear_right=0.3,
        ear_avg=0.3,
        is_eye_closed=False,
        blink_count=0,
        blink_duration_ms=0.0,
        perclos=0.05,
        fps=30.0,
        label="ACTIVE",
        label_id=0,
    )


def session(session_id: str, subject_id: str, count: int = 60) -> list[SampleRecord]:
    return [make_record(index, session_id, subject_id) for index in range(count)]


def test_record_and_partition_digests_detect_content_and_membership_changes() -> None:
    train = session("train", "subject_train")
    validation = session("validation", "subject_validation")
    held_out = session("held-out", "subject_held-out")
    partition = PartitionIdentity.from_partitions(
        train, validation, held_out, "subject"
    )

    changed = list(validation)
    changed[0] = make_record(0, "validation-changed", "subject_validation")
    assert record_digest(validation[0]) != record_digest(changed[0])
    assert record_set_digest(validation) != record_set_digest(changed)
    with pytest.raises(ValueError, match="overlap"):
        PartitionIdentity.from_partitions(train, validation, train, "subject")

    assert partition.digest != sha256_digest({"partition": "different"})


def test_normalizer_records_training_lineage_and_refit_replaces_it() -> None:
    train = session("train", "subject_train")
    normalizer = FeatureNormalizer().fit(train)
    assert normalizer.provenance is not None
    assert normalizer.provenance.fit_partition == "train"
    assert normalizer.provenance.fit_record_digest == record_set_digest(train)

    replacement = session("replacement", "subject_replacement")
    normalizer.fit(replacement)
    assert normalizer.provenance is not None
    assert normalizer.provenance.fit_record_digest == record_set_digest(replacement)


def test_candidate_comparison_uses_verified_identity_and_shared_contract() -> None:
    train = session("train", "subject_train")
    validation = session("validation", "subject_validation")
    held_out = session("held-out", "subject_held-out")
    normalizer = FeatureNormalizer().fit(train)
    assert normalizer.provenance is not None
    identity = EvaluationIdentity(
        dataset_name="synthetic-contract",
        dataset_version="1",
        manifest_digest="manifest",
        label_mapping_digest=label_mapping_digest(),
        split_strategy="subject",
        train_record_digest=record_set_digest(train),
        validation_record_digest=record_set_digest(validation),
        held_out_record_digest=record_set_digest(held_out),
        partition_digest=PartitionIdentity.from_partitions(
            train, validation, held_out, "subject"
        ).digest,
    )
    models = {
        "gru": GRUTemporalClassifier(GRUConfig(hidden_dim=4)),
        "lstm": LSTMTemporalClassifier(LSTMConfig(hidden_dim=4)),
        "tcn": TCNTemporalClassifier(TCNConfig(hidden_dim=4)),
    }
    provenance = {
        name: CandidateTrainingProvenance(
            candidate=describe_candidate(model),
            seed=42,
            batch_size=4,
            max_epochs=1,
            optimizer="Adam",
            learning_rate=1e-3,
            weight_decay=0.0,
            selection_metric="macro_f1",
        )
        for name, model in models.items()
    }
    report = compare_candidates(
        models,
        validation,
        normalizer,
        evaluation_identity=identity,
        normalizer_provenance=normalizer.provenance,
        candidate_provenance=provenance,
        evaluation_phase="development_validation",
        metadata=IndependentEvaluationMetadata(
            development_strategy="subject",
            evaluation_subjects=["subject_validation"],
            evaluation_sessions=["validation"],
        ),
        code_git_sha="0" * 40,
        is_synthetic=True,
    )
    assert [item.candidate.model_type for item in report.candidates] == [
        "gru",
        "lstm",
        "tcn",
    ]
    assert {item.window_count for item in report.candidates} == {1}
    assert report.to_dict()["is_synthetic"] is True


def test_candidate_comparison_rejects_forged_candidate_identity() -> None:
    train = session("train", "subject_train")
    validation = session("validation", "subject_validation")
    held_out = session("held-out", "subject_held-out")
    normalizer = FeatureNormalizer().fit(train)
    assert normalizer.provenance is not None
    identity = EvaluationIdentity(
        dataset_name="synthetic-contract",
        dataset_version="1",
        manifest_digest="manifest",
        label_mapping_digest=label_mapping_digest(),
        train_record_digest=record_set_digest(train),
        validation_record_digest=record_set_digest(validation),
        held_out_record_digest=record_set_digest(held_out),
        partition_digest=PartitionIdentity.from_partitions(
            train, validation, held_out, "subject"
        ).digest,
    )
    model = GRUTemporalClassifier(GRUConfig(hidden_dim=4))
    forged = CandidateTrainingProvenance(
        candidate=describe_candidate(LSTMTemporalClassifier(LSTMConfig(hidden_dim=4))),
        seed=42,
        batch_size=4,
        max_epochs=1,
        optimizer="Adam",
        learning_rate=1e-3,
        weight_decay=0.0,
        selection_metric="macro_f1",
    )
    with pytest.raises(ValueError, match="does not match"):
        compare_candidates(
            {"gru": model},
            validation,
            normalizer,
            evaluation_identity=identity,
            normalizer_provenance=normalizer.provenance,
            candidate_provenance={"gru": forged},
            evaluation_phase="development_validation",
            metadata=IndependentEvaluationMetadata(
                development_strategy="subject",
                evaluation_subjects=["subject_validation"],
                evaluation_sessions=["validation"],
            ),
            code_git_sha="0" * 40,
            is_synthetic=True,
        )


def test_legacy_experiment_records_remain_readable_without_fabricated_provenance() -> None:
    legacy = DevelopmentExperiment.from_dict({
        "git_sha": "0" * 40,
        "manifest": {"dataset_name": "legacy", "dataset_version": "1"},
        "seed": 42,
        "split": {
            "train_subjects": ["train"],
            "val_subjects": ["validation"],
            "train_sessions": ["train-session"],
            "val_sessions": ["validation-session"],
        },
        "model_config": {"name": "GRU", "input_dim": 8, "hidden_dim": 64},
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epoch_count": 1,
        "best_checkpoint_epoch": 1,
        "metrics": {},
        "artifact_sha256": "digest",
        "is_synthetic": "unknown",
    })
    assert legacy.schema_version == 0
    assert legacy.model_type == "gru"
    assert legacy.normalizer_provenance is None
