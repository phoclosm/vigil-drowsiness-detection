"""Unit and integration tests for Vigil M2 GRU temporal sequence classifier and training pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vigil.ml.baseline.evaluation import BaselineEvaluationReport
from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS, LOCKED_WINDOW_LENGTH
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier
from vigil.ml.training.config import TrainingConfig
from vigil.ml.training.trainer import (
    TrainingResult,
    seed_everything,
    train_temporal_classifier,
)


def make_test_record(
    frame_index: int,
    session_id: str = "session_001",
    subject_id: str = "subject_001",
    timestamp_ms: float | None = None,
    ear_avg: float = 0.30,
    is_eye_closed: bool = False,
    blink_duration_ms: float = 0.0,
    perclos: float = 0.05,
    label: str = "ACTIVE",
    label_id: int = 0,
) -> SampleRecord:
    """Helper creating a valid canonical SampleRecord for temporal tests."""
    t_ms = float(frame_index * 33.3) if timestamp_ms is None else timestamp_ms
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=t_ms,
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=True,
        landmarks_valid=True,
        ear_left=ear_avg,
        ear_right=ear_avg,
        ear_avg=ear_avg,
        is_eye_closed=is_eye_closed,
        blink_count=0,
        blink_duration_ms=blink_duration_ms,
        perclos=perclos,
        fps=30.0,
        label=label,
        label_id=label_id,
    )


def generate_synthetic_session(
    session_id: str,
    subject_id: str,
    num_frames: int = 80,
    label_id: int = 0,
) -> list[SampleRecord]:
    """Generate a contiguous synthetic session with a specified label pattern."""
    label_map = {0: "ACTIVE", 1: "DROWSY", 2: "SLEEPING"}
    label_name = label_map[label_id]
    ear_val = 0.32 if label_id == 0 else (0.22 if label_id == 1 else 0.12)
    closed = label_id == 2
    perclos_val = 0.05 if label_id == 0 else (0.25 if label_id == 1 else 0.80)

    return [
        make_test_record(
            frame_index=i,
            session_id=session_id,
            subject_id=subject_id,
            timestamp_ms=float(i * 33.3),
            ear_avg=ear_val,
            is_eye_closed=closed,
            perclos=perclos_val,
            label=label_name,
            label_id=label_id,
        )
        for i in range(num_frames)
    ]


class TestGRUModelArchitecture:
    """Tests for GRUTemporalClassifier model architecture, shapes, and contract."""

    def test_forward_output_shape_and_logits(self) -> None:
        """Verify model produces (batch_size, 3) raw logits for (batch_size, 60, 8) inputs."""
        model = GRUTemporalClassifier(GRUConfig(hidden_dim=32))
        x = torch.randn(4, LOCKED_WINDOW_LENGTH, len(LOCKED_FEATURE_CHANNELS))

        logits = model(x)
        assert logits.shape == (4, 3)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all().item() is True

    def test_predict_and_predict_proba(self) -> None:
        """Verify predict_proba normalizes to 1.0 and predict returns integer classes in (0, 1, 2)."""
        model = GRUTemporalClassifier(GRUConfig(hidden_dim=32))
        x = torch.randn(5, 60, 8)

        probs = model.predict_proba(x)
        assert probs.shape == (5, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

        preds = model.predict(x)
        assert preds.shape == (5,)
        assert preds.dtype == torch.int64
        for p in preds.tolist():
            assert p in (0, 1, 2)

    def test_input_shape_validation(self) -> None:
        """Reject inputs with incorrect dimensionality, sequence length, or feature count."""
        model = GRUTemporalClassifier()

        # Non-tensor
        with pytest.raises(TypeError, match="torch.Tensor"):
            model([[0.0] * 8] * 60)  # type: ignore[arg-type]

        # 2D tensor instead of 3D
        with pytest.raises(ValueError, match="3-dimensional"):
            model(torch.randn(60, 8))

        # 4D tensor
        with pytest.raises(ValueError, match="3-dimensional"):
            model(torch.randn(2, 60, 8, 1))

        # Wrong sequence length (30 instead of 60)
        with pytest.raises(ValueError, match="sequence length"):
            model(torch.randn(2, 30, 8))

        # Wrong feature channel count (7 instead of 8)
        with pytest.raises(ValueError, match="feature dimension"):
            model(torch.randn(2, 60, 7))

    def test_non_finite_inputs_rejected(self) -> None:
        """Reject input tensors containing NaN or infinity."""
        model = GRUTemporalClassifier()
        x_nan = torch.zeros(2, 60, 8)
        x_nan[0, 10, 2] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            model(x_nan)

        x_inf = torch.zeros(2, 60, 8)
        x_inf[1, 50, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            model(x_inf)

    def test_configuration_validation(self) -> None:
        """Enforce strict configuration parameter constraints."""
        with pytest.raises(ValueError, match="input_dim"):
            GRUConfig(input_dim=10)
        with pytest.raises(ValueError, match="hidden_dim"):
            GRUConfig(hidden_dim=0)
        with pytest.raises(ValueError, match="num_layers"):
            GRUConfig(num_layers=0)
        with pytest.raises(ValueError, match="num_classes"):
            GRUConfig(num_classes=2)
        with pytest.raises(ValueError, match="dropout"):
            GRUConfig(dropout=1.0)
        with pytest.raises(TypeError, match="bidirectional"):
            GRUConfig(bidirectional="True")  # type: ignore[arg-type]

    def test_deterministic_eval_mode(self) -> None:
        """Verify identical outputs for identical inputs in eval mode."""
        seed_everything(42)
        model = GRUTemporalClassifier(GRUConfig(hidden_dim=32))
        model.eval()

        x = torch.randn(3, 60, 8)
        out1 = model(x)
        out2 = model(x)
        assert torch.equal(out1, out2)

    def test_missing_torch_raises_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify clear, actionable error is raised when PyTorch is not available."""
        from vigil.ml.models import gru

        monkeypatch.setattr(gru, "HAS_TORCH", False)
        with pytest.raises(ImportError, match="PyTorch is required to instantiate GRUTemporalClassifier"):
            gru.GRUTemporalClassifier()


class TestTrainingPipeline:
    """Tests for train_temporal_classifier reproducible training pipeline."""

    def test_training_pipeline_end_to_end(self) -> None:
        """Run a 3-epoch training session on synthetic records and verify all outputs."""
        # 2 subjects, each with 2 sessions of 80 frames
        records: list[SampleRecord] = []
        # Subject 1 (Train): Active and Drowsy sessions
        records.extend(generate_synthetic_session("s1_active", "subject_1", 80, label_id=0))
        records.extend(generate_synthetic_session("s1_drowsy", "subject_1", 80, label_id=1))
        # Subject 2 (Validation): Active and Sleeping sessions
        records.extend(generate_synthetic_session("s2_active", "subject_2", 80, label_id=0))
        records.extend(generate_synthetic_session("s2_sleeping", "subject_2", 80, label_id=2))

        cfg = TrainingConfig(
            max_epochs=3,
            batch_size=4,
            learning_rate=1e-3,
            hidden_dim=32,
            seed=42,
        )

        result = train_temporal_classifier(records, cfg)

        assert isinstance(result, TrainingResult)
        assert 1 <= result.best_epoch <= 3
        assert len(result.train_losses) == 3
        assert len(result.val_losses) == 3
        assert result.best_val_loss == min(result.val_losses)

        # Losses must be finite and positive
        for loss in result.train_losses + result.val_losses:
            assert loss > 0.0

        # Evaluation report verification
        report = result.evaluation_report
        assert isinstance(report, BaselineEvaluationReport)
        assert 0.0 <= report.accuracy <= 1.0
        assert len(report.confusion_matrix) == 3
        assert len(report.confusion_matrix[0]) == 3

        # Subject-first partition verification (no leakage)
        train_subjs = set(result.split_result.train_subjects)
        val_subjs = set(result.split_result.val_subjects)
        assert train_subjs.isdisjoint(val_subjs)
        assert result.split_result.strategy == "subject"

    def test_loss_reduction_over_epochs(self) -> None:
        """Verify that training loss decreases across epochs on a learnable synthetic task."""
        records: list[SampleRecord] = []
        # Train: 3 sessions for subject_1
        records.extend(generate_synthetic_session("s1_a", "subject_1", 90, label_id=0))
        records.extend(generate_synthetic_session("s1_b", "subject_1", 90, label_id=1))
        records.extend(generate_synthetic_session("s1_c", "subject_1", 90, label_id=2))
        # Val: 2 sessions for subject_2
        records.extend(generate_synthetic_session("s2_a", "subject_2", 90, label_id=0))
        records.extend(generate_synthetic_session("s2_b", "subject_2", 90, label_id=1))

        cfg = TrainingConfig(
            max_epochs=8,
            batch_size=8,
            learning_rate=3e-3,
            hidden_dim=32,
            seed=123,
        )

        result = train_temporal_classifier(records, cfg)
        # Final training loss should be lower than initial training loss
        assert result.train_losses[-1] < result.train_losses[0]

    def test_deterministic_seeded_training(self) -> None:
        """Verify identical loss trajectories when training with the same seed."""
        records: list[SampleRecord] = []
        records.extend(generate_synthetic_session("s1_a", "subject_1", 75, label_id=0))
        records.extend(generate_synthetic_session("s2_a", "subject_2", 75, label_id=1))

        cfg = TrainingConfig(max_epochs=2, batch_size=4, hidden_dim=16, seed=42)

        res1 = train_temporal_classifier(records, cfg)
        res2 = train_temporal_classifier(records, cfg)

        assert res1.train_losses == pytest.approx(res2.train_losses, rel=1e-5)
        assert res1.val_losses == pytest.approx(res2.val_losses, rel=1e-5)
        assert res1.best_val_loss == pytest.approx(res2.best_val_loss, rel=1e-5)

    def test_checkpoint_saved_to_output_dir(self, tmp_path: Path) -> None:
        """Verify that best checkpoint is saved to disk when output_dir is specified."""
        records: list[SampleRecord] = []
        records.extend(generate_synthetic_session("s1_a", "subject_1", 75, label_id=0))
        records.extend(generate_synthetic_session("s2_a", "subject_2", 75, label_id=1))

        out_dir = str(tmp_path / "checkpoint_test")
        cfg = TrainingConfig(max_epochs=1, batch_size=4, output_dir=out_dir)

        result = train_temporal_classifier(records, cfg)
        assert result.checkpoint_path is not None
        assert os.path.exists(result.checkpoint_path)

        # Loaded checkpoint should restore model state
        loaded = torch.load(result.checkpoint_path, map_location="cpu")
        assert isinstance(loaded, dict)
        assert "gru.weight_ih_l0" in loaded

    def test_zero_support_class_reporting(self) -> None:
        """Verify that classes with no ground-truth support in validation are reported."""
        records: list[SampleRecord] = []
        # Training and validation together only contain classes 0 and 1; class 2 (SLEEPING) has zero support
        records.extend(generate_synthetic_session("s1_a", "subject_1", 75, label_id=0))
        records.extend(generate_synthetic_session("s1_b", "subject_1", 75, label_id=1))
        records.extend(generate_synthetic_session("s2_a", "subject_2", 75, label_id=0))

        cfg = TrainingConfig(max_epochs=1, batch_size=4)
        result = train_temporal_classifier(records, cfg)

        assert 2 in result.evaluation_report.zero_support_classes

    def test_insufficient_records_or_windows_rejected(self) -> None:
        """Reject training runs that lack usable windows or partitions."""
        # Empty records
        with pytest.raises(ValueError, match="No sample records"):
            train_temporal_classifier([])

        # Single session: cannot create validation partition without leakage
        single_sess = generate_synthetic_session("single_s", "sub_1", 80)
        with pytest.raises(ValueError, match="insufficient partitions"):
            train_temporal_classifier(single_sess)

        # Sessions shorter than W=60: 0 usable windows extracted
        short1 = generate_synthetic_session("s1", "sub_1", 40)
        short2 = generate_synthetic_session("s2", "sub_2", 40)
        with pytest.raises(ValueError, match="0 usable windows"):
            train_temporal_classifier(short1 + short2)

    def test_missing_torch_raises_actionable_error_in_training(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify clear, actionable error is raised when PyTorch is not available during training."""
        from vigil.ml.training import trainer

        monkeypatch.setattr(trainer, "HAS_TORCH", False)
        records = generate_synthetic_session("s1", "sub_1", 75)
        with pytest.raises(ImportError, match="PyTorch is required for temporal classifier training"):
            trainer.train_temporal_classifier(records)
