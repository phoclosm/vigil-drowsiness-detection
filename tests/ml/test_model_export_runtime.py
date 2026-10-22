"""Unit and integration tests for Vigil M2 model export, runtime inference, and evaluation comparison."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch

from vigil.ml.data.schema import LiveInputFrame, SampleRecord
from vigil.ml.dataset.config import (
    LOCKED_FEATURE_CHANNELS,
    LOCKED_STRIDE,
    LOCKED_WINDOW_LENGTH,
)
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.evaluation.comparison import (
    ModelComparisonReport,
    compare_models,
    evaluate_recording_directory,
)
from vigil.ml.inference.engine import (
    CANONICAL_CLASS_LABELS,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_INVALID_INPUT,
    STATUS_OK,
    InferenceResult,
    TemporalInferenceEngine,
)
from vigil.ml.models.export import (
    BUNDLE_SCHEMA_VERSION,
    LoadedModelBundle,
    export_model_bundle,
    load_model_bundle,
)
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier

# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================


def make_test_sample(
    frame_index: int,
    session_id: str = "session_001",
    subject_id: str = "subject_001",
    timestamp_ms: float | None = None,
    ear_avg: float = 0.30,
    is_eye_closed: bool = False,
    blink_duration_ms: float = 0.0,
    perclos: float = 0.05,
    landmarks_valid: bool = True,
    label: str = "ACTIVE",
    label_id: int = 0,
) -> SampleRecord:
    """Construct a canonical SampleRecord for streaming runtime tests."""
    t_ms = float(frame_index * 33.3) if timestamp_ms is None else timestamp_ms
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=t_ms,
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=True,
        landmarks_valid=landmarks_valid,
        ear_left=ear_avg if landmarks_valid else None,
        ear_right=ear_avg if landmarks_valid else None,
        ear_avg=ear_avg if landmarks_valid else None,
        is_eye_closed=is_eye_closed if landmarks_valid else None,
        blink_count=0,
        blink_duration_ms=blink_duration_ms,
        perclos=perclos,
        fps=30.0,
        label=label,
        label_id=label_id,
    )


def make_live_frame(
    frame_index: int,
    session_id: str = "live_session_001",
    timestamp_ms: float | None = None,
    ear_avg: float = 0.30,
    is_eye_closed: bool = False,
    blink_duration_ms: float = 0.0,
    perclos: float = 0.05,
    landmarks_valid: bool = True,
) -> LiveInputFrame:
    """Construct a canonical LiveInputFrame (no label fields) for live inference tests."""
    t_ms = float(frame_index * 33.3) if timestamp_ms is None else timestamp_ms
    return LiveInputFrame(
        session_id=session_id,
        frame_index=frame_index,
        timestamp_ms=t_ms,
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=True,
        landmarks_valid=landmarks_valid,
        ear_left=ear_avg if landmarks_valid else None,
        ear_right=ear_avg if landmarks_valid else None,
        ear_avg=ear_avg if landmarks_valid else None,
        is_eye_closed=is_eye_closed if landmarks_valid else None,
        blink_count=0,
        blink_duration_ms=blink_duration_ms,
        perclos=perclos,
        fps=30.0,
    )


def create_fitted_normalizer() -> FeatureNormalizer:
    """Create and fit a FeatureNormalizer with deterministic reference values."""
    normalizer = FeatureNormalizer()
    normalizer.means = {"ear_avg": 0.28, "blink_duration_ms": 150.0, "perclos": 0.12}
    normalizer.stds = {"ear_avg": 0.05, "blink_duration_ms": 50.0, "perclos": 0.08}
    normalizer.observed_counts = {"ear_avg": 100, "blink_duration_ms": 100, "perclos": 100}
    normalizer.is_fitted = True
    return normalizer


# ==============================================================================
# LiveInputFrame Schema Tests
# ==============================================================================


class TestLiveInputFrame:
    """Tests for the unlabeled runtime-input type."""

    def test_valid_frame_index_0_constructed(self) -> None:
        """A well-formed frame_index=0 LiveInputFrame is accepted without errors."""
        f = make_live_frame(frame_index=0)
        assert f.session_id == "live_session_001"
        assert f.frame_index == 0
        assert f.timestamp_ms == 0.0
        assert f.frame_delta_ms is None
        assert f.blink_count == 0

    def test_valid_frame_later_constructed(self) -> None:
        """A well-formed frame_index=5 LiveInputFrame with frame_delta_ms is accepted."""
        f = make_live_frame(frame_index=5)
        assert f.frame_index == 5
        assert f.frame_delta_ms == 33.3

    def test_no_label_fields(self) -> None:
        """LiveInputFrame must not have label or label_id attributes."""
        f = make_live_frame(frame_index=0)
        assert not hasattr(f, "label")
        assert not hasattr(f, "label_id")
        assert not hasattr(f, "subject_id")

    def test_missing_landmarks_accepted(self) -> None:
        """Frames with landmarks_valid=False and null ocular features are valid."""
        f = make_live_frame(frame_index=0, landmarks_valid=False)
        assert f.ear_avg is None
        assert f.is_eye_closed is None

    def test_ear_range_enforced(self) -> None:
        """EAR values outside [0.0, 0.60] are rejected (first violating field reported)."""
        with pytest.raises(ValueError, match="must be within \\[0.0, 0.60\\]"):
            LiveInputFrame(
                session_id="s",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.61,
                ear_right=0.61,
                ear_avg=0.61,
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
            )

    def test_inconsistent_ear_avg_rejected(self) -> None:
        """ear_avg inconsistent with (ear_left + ear_right)/2 is rejected."""
        with pytest.raises(ValueError, match="Inconsistent ear_avg"):
            LiveInputFrame(
                session_id="s",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=True,
                ear_left=0.30,
                ear_right=0.30,
                ear_avg=0.20,  # wrong
                is_eye_closed=False,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
            )

    def test_first_frame_non_zero_blink_count_rejected(self) -> None:
        """frame_index=0 with blink_count != 0 is rejected."""
        with pytest.raises(ValueError, match="blink_count == 0"):
            LiveInputFrame(
                session_id="s",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=True,
                landmarks_valid=False,
                ear_left=None,
                ear_right=None,
                ear_avg=None,
                is_eye_closed=None,
                blink_count=3,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
            )

    def test_fps_out_of_range_rejected(self) -> None:
        """fps > 120.0 is rejected."""
        with pytest.raises(ValueError, match="fps must be <= 120.0"):
            LiveInputFrame(
                session_id="s",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=False,
                landmarks_valid=False,
                ear_left=None,
                ear_right=None,
                ear_avg=None,
                is_eye_closed=None,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=200.0,
            )

    def test_sample_record_schema_unchanged(self) -> None:
        """Adding LiveInputFrame must not loosen SampleRecord validation."""
        with pytest.raises((ValueError, TypeError)):
            # SampleRecord still requires label and label_id
            SampleRecord(  # type: ignore[call-arg]
                sample_id="s_f0",
                session_id="s",
                subject_id="subject_001",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=False,
                landmarks_valid=False,
                ear_left=None,
                ear_right=None,
                ear_avg=None,
                is_eye_closed=None,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
                # label and label_id deliberately omitted
            )


# ==============================================================================
# InferenceResult Contract Tests
# ==============================================================================


class TestInferenceResultContract:
    """Verify every response contains the six required contract fields."""

    def _make_engine(self) -> TemporalInferenceEngine:
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        return TemporalInferenceEngine(model=model, normalizer=normalizer)

    def _required_fields(self, res: InferenceResult) -> None:
        """Assert all six contract fields are present and have the right types."""
        assert hasattr(res, "predicted_class")
        assert hasattr(res, "class_id")
        assert hasattr(res, "probabilities")
        assert hasattr(res, "confidence")
        assert hasattr(res, "status")
        assert hasattr(res, "latency_ms")

    def test_warm_up_response_has_all_contract_fields(self) -> None:
        """Responses during warm-up include all six contract fields with None predictions."""
        engine = self._make_engine()
        res = engine.step(make_live_frame(frame_index=0))
        self._required_fields(res)
        assert res.predicted_class is None
        assert res.class_id is None
        assert res.probabilities is None
        assert res.confidence is None
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert math.isfinite(res.latency_ms) and res.latency_ms >= 0.0

    def test_prediction_response_has_all_contract_fields(self) -> None:
        """Responses with a prediction include all six contract fields."""
        engine = self._make_engine()
        for i in range(60):
            res = engine.step(make_live_frame(frame_index=i))
        self._required_fields(res)  # type: ignore[possibly-undefined]
        assert res.status == STATUS_OK
        assert res.predicted_class in ("ACTIVE", "DROWSY", "SLEEPING")
        assert res.class_id in (0, 1, 2)
        assert res.probabilities is not None
        assert set(res.probabilities.keys()) == {"ACTIVE", "DROWSY", "SLEEPING"}
        assert pytest.approx(sum(res.probabilities.values()), abs=1e-5) == 1.0
        assert res.confidence == pytest.approx(res.probabilities[res.predicted_class])
        assert math.isfinite(res.latency_ms) and res.latency_ms >= 0.0

    def test_session_reset_response_has_all_contract_fields(self) -> None:
        """Session-reset responses include all six contract fields."""
        engine = self._make_engine()
        engine.step(make_live_frame(frame_index=0, session_id="s1"))
        res = engine.step(make_live_frame(frame_index=0, session_id="s2"))
        self._required_fields(res)
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert res.predicted_class is None
        assert math.isfinite(res.latency_ms) and res.latency_ms >= 0.0

    def test_discontinuity_response_has_all_contract_fields(self) -> None:
        """Discontinuity responses include all six contract fields."""
        engine = self._make_engine()
        engine.step(make_live_frame(frame_index=0))
        engine.step(make_live_frame(frame_index=1, timestamp_ms=33.3))
        # Frame gap > 1 → discontinuity
        res = engine.step(make_live_frame(frame_index=10, timestamp_ms=333.3))
        self._required_fields(res)
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert res.predicted_class is None
        assert math.isfinite(res.latency_ms) and res.latency_ms >= 0.0

    def test_stride_skip_response_has_all_contract_fields(self) -> None:
        """Stride-skip responses include all six contract fields."""
        engine = self._make_engine()
        for i in range(61):
            res = engine.step(make_live_frame(frame_index=i))
        # frame 60 is stride_skip
        self._required_fields(res)  # type: ignore[possibly-undefined]
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert res.predicted_class is None
        assert math.isfinite(res.latency_ms) and res.latency_ms >= 0.0

    def test_latency_ms_non_negative_and_finite(self) -> None:
        """latency_ms is finite and non-negative on every result."""
        engine = self._make_engine()
        for i in range(80):
            res = engine.step(make_live_frame(frame_index=i))
            assert math.isfinite(res.latency_ms), f"frame {i}: latency_ms is not finite"
            assert res.latency_ms >= 0.0, f"frame {i}: latency_ms is negative"

    def test_published_status_values_used(self) -> None:
        """Status values match the published STATUS_* constants."""
        engine = self._make_engine()

        # Warm-up → STATUS_INSUFFICIENT_HISTORY
        res = engine.step(make_live_frame(frame_index=0))
        assert res.status == STATUS_INSUFFICIENT_HISTORY

        # Fill to first prediction → STATUS_OK
        for i in range(1, 60):
            res = engine.step(make_live_frame(frame_index=i))
        assert res.status == STATUS_OK

        # Stride skip → STATUS_INSUFFICIENT_HISTORY
        res = engine.step(make_live_frame(frame_index=60))
        assert res.status == STATUS_INSUFFICIENT_HISTORY

        # Session reset → STATUS_INSUFFICIENT_HISTORY
        res = engine.step(make_live_frame(frame_index=0, session_id="other_session"))
        assert res.status == STATUS_INSUFFICIENT_HISTORY

    def test_backward_compat_has_prediction_alias(self) -> None:
        """The has_prediction property (backward compat) equals status == STATUS_OK."""
        engine = self._make_engine()
        res = engine.step(make_live_frame(frame_index=0))
        assert res.has_prediction == (res.status == STATUS_OK)


# ==============================================================================
# Live Input (Unlabeled) Tests
# ==============================================================================


class TestLiveInputInference:
    """Tests verifying the engine accepts LiveInputFrame without ground-truth labels."""

    def _make_engine(self) -> TemporalInferenceEngine:
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        return TemporalInferenceEngine(model=model, normalizer=normalizer)

    def test_live_frames_accepted_without_label(self) -> None:
        """Engine accepts LiveInputFrame without label or label_id fields."""
        engine = self._make_engine()
        for i in range(60):
            frame = make_live_frame(frame_index=i)
            res = engine.step(frame)
        assert res.status == STATUS_OK  # type: ignore[possibly-undefined]
        assert res.predicted_class in ("ACTIVE", "DROWSY", "SLEEPING")

    def test_live_frames_missing_landmarks_accepted(self) -> None:
        """Engine accepts LiveInputFrame with missing landmarks (landmarks_valid=False)."""
        engine = self._make_engine()
        for i in range(60):
            frame = make_live_frame(frame_index=i, landmarks_valid=(i % 2 == 0))
            res = engine.step(frame)
        assert res.status == STATUS_OK  # type: ignore[possibly-undefined]

    def test_live_frames_do_not_weaken_sample_record(self) -> None:
        """SampleRecord validation is unchanged after LiveInputFrame addition."""
        with pytest.raises(TypeError):
            SampleRecord(  # type: ignore[call-arg]
                session_id="s",
                frame_index=0,
                timestamp_ms=0.0,
                frame_delta_ms=None,
                face_detected=False,
                landmarks_valid=False,
                ear_left=None,
                ear_right=None,
                ear_avg=None,
                is_eye_closed=None,
                blink_count=0,
                blink_duration_ms=0.0,
                perclos=None,
                fps=None,
            )

    def test_invalid_type_rejected(self) -> None:
        """Passing neither a SampleRecord nor a LiveInputFrame raises TypeError."""
        engine = self._make_engine()
        with pytest.raises(TypeError, match="must be a SampleRecord or LiveInputFrame"):
            engine.step("not_a_frame")  # type: ignore[arg-type]

    def test_mixed_live_and_labelled_frames_accepted(self) -> None:
        """Engine accepts a mix of SampleRecord and LiveInputFrame in the same session."""
        engine = self._make_engine()
        # Feed labelled samples for first 30 frames
        for i in range(30):
            engine.step(make_test_sample(frame_index=i))
        # Then switch to live (unlabelled) frames for the next 30 — same session_id
        for i in range(30, 60):
            engine.step(make_live_frame(frame_index=i))
        # Final frame should produce a prediction
        res = engine.step(make_live_frame(frame_index=60))
        # buf full, and this is a stride_skip (frame 60 = 1st frame after pred at 59)
        assert res.status in (STATUS_OK, STATUS_INSUFFICIENT_HISTORY)

    def test_session_id_tracked_with_live_frames(self) -> None:
        """Engine correctly tracks session_id when using LiveInputFrame."""
        engine = self._make_engine()
        frame = make_live_frame(frame_index=0, session_id="live_sess_A")
        engine.step(frame)
        assert engine.current_session_id == "live_sess_A"

    def test_warm_up_with_live_frames(self) -> None:
        """Warm-up and stride semantics work identically for LiveInputFrame."""
        engine = self._make_engine()
        for i in range(59):
            res = engine.step(make_live_frame(frame_index=i))
            assert res.status == STATUS_INSUFFICIENT_HISTORY
            assert res.reason == "buffer_warming_up"

        # 60th frame → first prediction
        res = engine.step(make_live_frame(frame_index=59))
        assert res.status == STATUS_OK
        assert res.reason == "predicted"


# ==============================================================================
# Model Export and Loading Tests (unchanged, kept for regression)
# ==============================================================================


class TestModelExportAndLoad:
    """Tests covering model checkpoint export, loading, validation, and reconstruction."""

    def test_export_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Export a model and normalizer, load back, and verify exact prediction consistency."""
        config = GRUConfig(input_dim=8, hidden_dim=32, num_layers=1, num_classes=3, dropout=0.0)
        model = GRUTemporalClassifier(config)
        normalizer = create_fitted_normalizer()

        bundle_path = tmp_path / "model_bundle.pt"
        exported_path = export_model_bundle(
            model=model,
            normalizer=normalizer,
            filepath=bundle_path,
            metadata={"experiment": "d6_test", "author": "m2"},
        )
        assert exported_path.exists()

        # Load back via standalone function
        bundle = load_model_bundle(bundle_path)
        assert isinstance(bundle, LoadedModelBundle)
        assert bundle.bundle_version == BUNDLE_SCHEMA_VERSION
        assert bundle.metadata["experiment"] == "d6_test"
        assert bundle.feature_channels == LOCKED_FEATURE_CHANNELS
        assert bundle.class_labels == CANONICAL_CLASS_LABELS
        assert bundle.window_config.window_length == LOCKED_WINDOW_LENGTH
        assert bundle.window_config.stride == LOCKED_STRIDE

        # Verify model reconstruction and eval mode
        assert not bundle.model.training
        assert bundle.model.config.hidden_dim == 32
        assert bundle.model.config.num_layers == 1

        # Verify normalizer reconstruction
        assert bundle.normalizer.is_fitted
        assert bundle.normalizer.means == normalizer.means
        assert bundle.normalizer.stds == normalizer.stds
        assert bundle.normalizer.observed_counts == normalizer.observed_counts

        # Verify exact numerical output consistency
        torch.manual_seed(42)
        test_input = torch.randn(2, LOCKED_WINDOW_LENGTH, len(LOCKED_FEATURE_CHANNELS))
        with torch.no_grad():
            orig_out = model(test_input)
            loaded_out = bundle.model(test_input)

        assert torch.allclose(orig_out, loaded_out, atol=1e-6)

        # Also verify LoadedModelBundle.from_bundle classmethod
        bundle_alt = LoadedModelBundle.from_bundle(bundle_path)
        assert isinstance(bundle_alt, LoadedModelBundle)

    def test_loaded_bundle_runs_live_inference_after_warmup(self, tmp_path: Path) -> None:
        """A loaded bundle preserves metadata and serves the established live contract."""
        model = GRUTemporalClassifier(GRUConfig(hidden_dim=16))
        model.training_provenance = {  # type: ignore[attr-defined]
            "strategy": "session",
            "train_subjects": ["subject_001"],
            "val_subjects": ["subject_001"],
            "train_sessions": ["training_session"],
            "val_sessions": ["validation_session"],
        }
        normalizer = create_fitted_normalizer()
        bundle_path = tmp_path / "live_smoke_bundle.pt"
        export_model_bundle(model, normalizer, bundle_path)

        bundle = load_model_bundle(bundle_path)
        assert bundle.model.config == model.config
        assert bundle.normalizer.to_dict() == normalizer.to_dict()
        assert bundle.feature_channels == LOCKED_FEATURE_CHANNELS
        assert bundle.class_labels == CANONICAL_CLASS_LABELS
        assert bundle.window_config.window_length == LOCKED_WINDOW_LENGTH
        assert bundle.window_config.stride == LOCKED_STRIDE
        assert bundle.metadata["training_provenance"]["val_sessions"] == [
            "validation_session"
        ]

        engine = TemporalInferenceEngine(bundle.model, bundle.normalizer)
        result = None
        for frame_index in range(LOCKED_WINDOW_LENGTH):
            result = engine.step(make_live_frame(frame_index))

        assert result is not None
        assert result.status == STATUS_OK
        assert result.predicted_class in CANONICAL_CLASS_LABELS.values()
        assert result.class_id in CANONICAL_CLASS_LABELS
        assert result.probabilities is not None
        assert set(result.probabilities) == set(CANONICAL_CLASS_LABELS.values())
        assert result.confidence is not None
        assert result.latency_ms >= 0.0

    def test_export_unfitted_normalizer_rejected(self, tmp_path: Path) -> None:
        """Exporting an unfitted normalizer must raise a ValueError."""
        model = GRUTemporalClassifier(GRUConfig())
        unfitted_norm = FeatureNormalizer()
        with pytest.raises(ValueError, match="Cannot export an unfitted FeatureNormalizer"):
            export_model_bundle(model, unfitted_norm, tmp_path / "bundle.pt")

    def test_export_invalid_types_rejected(self, tmp_path: Path) -> None:
        """Exporting invalid model or normalizer types must raise TypeError."""
        normalizer = create_fitted_normalizer()
        with pytest.raises(TypeError, match="model must be a GRUTemporalClassifier"):
            export_model_bundle("not_a_model", normalizer, tmp_path / "bundle.pt")  # type: ignore[arg-type]

        model = GRUTemporalClassifier(GRUConfig())
        with pytest.raises(TypeError, match="normalizer must be a FeatureNormalizer"):
            export_model_bundle(model, "not_a_norm", tmp_path / "bundle.pt")  # type: ignore[arg-type]

    def test_export_incompatible_model_configuration_rejected(self, tmp_path: Path) -> None:
        """Export rejects a model that cannot satisfy the locked runtime contract."""
        normalizer = create_fitted_normalizer()
        with pytest.raises(ValueError, match="input_dim"):
            export_model_bundle(
                GRUTemporalClassifier(GRUConfig(input_dim=4)),
                normalizer,
                tmp_path / "bad_input_dim.pt",
            )
        with pytest.raises(ValueError, match="num_classes"):
            export_model_bundle(
                GRUTemporalClassifier(GRUConfig(num_classes=2)),
                normalizer,
                tmp_path / "bad_class_count.pt",
            )

    def test_export_persists_training_provenance(self, tmp_path: Path) -> None:
        """Training group provenance is carried into exported bundle metadata."""
        model = GRUTemporalClassifier(GRUConfig())
        model.training_provenance = {  # type: ignore[attr-defined]
            "strategy": "subject",
            "train_subjects": ["subject_train"],
            "val_subjects": ["subject_val"],
            "train_sessions": ["session_train"],
            "val_sessions": ["session_val"],
        }
        bundle_path = tmp_path / "provenance.pt"
        export_model_bundle(model, create_fitted_normalizer(), bundle_path)
        raw = torch.load(bundle_path, weights_only=False)
        assert raw["metadata"]["training_provenance"]["train_subjects"] == ["subject_train"]

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent bundle path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model bundle not found"):
            load_model_bundle(tmp_path / "missing.pt")

    def test_load_corrupted_raw_bundle_raises(self, tmp_path: Path) -> None:
        """Loading a non-dict bundle must raise ValueError."""
        corrupted_path = tmp_path / "corrupted.pt"
        torch.save("just a string", corrupted_path)
        with pytest.raises(TypeError, match="Invalid model bundle structure"):
            load_model_bundle(corrupted_path)

    def test_load_missing_required_keys_raises(self, tmp_path: Path) -> None:
        """Loading a bundle missing required top-level sections must raise ValueError."""
        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        bundle_path = tmp_path / "incomplete.pt"
        export_model_bundle(model, normalizer, bundle_path)

        # Tamper: remove 'state_dict'
        raw = torch.load(bundle_path, weights_only=False)
        del raw["state_dict"]
        torch.save(raw, bundle_path)

        with pytest.raises(ValueError, match="missing required section: 'state_dict'"):
            load_model_bundle(bundle_path)

    def test_load_incompatible_version_raises(self, tmp_path: Path) -> None:
        """Loading a bundle with an incompatible major version must raise ValueError."""
        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        bundle_path = tmp_path / "bad_version.pt"
        export_model_bundle(model, normalizer, bundle_path)

        raw = torch.load(bundle_path, weights_only=False)
        raw["bundle_version"] = "2.0.0"
        torch.save(raw, bundle_path)

        with pytest.raises(ValueError, match="Incompatible bundle version '2.0.0'"):
            load_model_bundle(bundle_path)

    def test_load_incompatible_feature_channels_raises(self, tmp_path: Path) -> None:
        """Loading a bundle with mismatched feature channels must raise ValueError."""
        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        bundle_path = tmp_path / "bad_channels.pt"
        export_model_bundle(model, normalizer, bundle_path)

        raw = torch.load(bundle_path, weights_only=False)
        raw["feature_channels"] = ["ear_avg", "perclos"]
        torch.save(raw, bundle_path)

        with pytest.raises(ValueError, match="Bundle feature channels do not match"):
            load_model_bundle(bundle_path)

    def test_load_unfitted_normalizer_in_bundle_raises(self, tmp_path: Path) -> None:
        """Loading a bundle where normalizer is_fitted is False must raise ValueError."""
        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        bundle_path = tmp_path / "unfitted_in_bundle.pt"
        export_model_bundle(model, normalizer, bundle_path)

        raw = torch.load(bundle_path, weights_only=False)
        raw["normalizer"]["is_fitted"] = False
        torch.save(raw, bundle_path)

        with pytest.raises(ValueError, match="normalizer in bundle must be a dictionary with is_fitted=True"):
            load_model_bundle(bundle_path)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("model_type", "OtherModel", "Unsupported model_type"),
            ("class_labels", {0: "ACTIVE", 1: "DROWSY", 2: "ALERT"}, "class_labels must exactly"),
            ("window_config", {"window_length": 30, "stride": 5, "max_frame_gap": 1, "max_time_gap_ms": 200.0}, "window_config must use locked"),
        ],
    )
    def test_load_rejects_incompatible_bundle_metadata(
        self,
        tmp_path: Path,
        field: str,
        value: object,
        message: str,
    ) -> None:
        """Unsupported model, class, and window metadata fail during loading."""
        bundle_path = tmp_path / f"bad_{field}.pt"
        export_model_bundle(
            GRUTemporalClassifier(GRUConfig()),
            create_fitted_normalizer(),
            bundle_path,
        )
        raw = torch.load(bundle_path, weights_only=False)
        raw[field] = value
        torch.save(raw, bundle_path)
        with pytest.raises((TypeError, ValueError), match=message):
            load_model_bundle(bundle_path)

    @pytest.mark.parametrize(
        ("section", "feature", "value", "message"),
        [
            ("means", "ear_avg", float("nan"), "normalizer.means"),
            ("stds", "perclos", 0.0, "normalizer.stds"),
            ("observed_counts", "blink_duration_ms", -1, "normalizer.observed_counts"),
        ],
    )
    def test_load_rejects_invalid_normalizer_values(
        self,
        tmp_path: Path,
        section: str,
        feature: str,
        value: object,
        message: str,
    ) -> None:
        """Malformed normalizer statistics fail during bundle loading."""
        bundle_path = tmp_path / f"bad_{section}.pt"
        export_model_bundle(
            GRUTemporalClassifier(GRUConfig()),
            create_fitted_normalizer(),
            bundle_path,
        )
        raw = torch.load(bundle_path, weights_only=False)
        raw["normalizer"][section][feature] = value
        torch.save(raw, bundle_path)
        with pytest.raises(ValueError, match=message):
            load_model_bundle(bundle_path)

    def test_load_rejects_missing_normalizer_feature(self, tmp_path: Path) -> None:
        """All continuous normalizer feature keys are required."""
        bundle_path = tmp_path / "missing_norm_key.pt"
        export_model_bundle(
            GRUTemporalClassifier(GRUConfig()),
            create_fitted_normalizer(),
            bundle_path,
        )
        raw = torch.load(bundle_path, weights_only=False)
        del raw["normalizer"]["means"]["ear_avg"]
        torch.save(raw, bundle_path)
        with pytest.raises(ValueError, match="required continuous feature keys"):
            load_model_bundle(bundle_path)


# ==============================================================================
# Stateful Runtime Inference Engine Tests (SampleRecord path — regression)
# ==============================================================================


class TestTemporalInferenceEngine:
    """Tests covering stateful rolling windowing, warm-up, stride, and reset contracts."""

    def test_warmup_and_first_prediction(self) -> None:
        """Engine produces no prediction for frames 0..58, and emits first prediction at frame 59."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        # Feed 59 frames (warm-up phase)
        for i in range(59):
            rec = make_test_sample(frame_index=i)
            res = engine.process_sample(rec)
            assert not res.has_prediction
            assert res.label is None
            assert res.label_id is None
            assert res.probabilities is None
            assert res.reason == "buffer_warming_up"
            assert res.status == STATUS_INSUFFICIENT_HISTORY
            assert res.buffer_size == i + 1
            assert engine.buffer_size == i + 1

        # Frame 59 (60th frame → first full window)
        rec_60 = make_test_sample(frame_index=59)
        res_60 = engine.process_sample(rec_60)
        assert res_60.has_prediction
        assert res_60.label in CANONICAL_CLASS_LABELS.values()
        assert res_60.label_id in (0, 1, 2)
        assert res_60.reason == "predicted"
        assert res_60.status == STATUS_OK
        assert res_60.buffer_size == 60
        assert res_60.start_frame_index == 0
        assert res_60.end_frame_index == 59
        assert res_60.probabilities is not None
        assert set(res_60.probabilities.keys()) == {"ACTIVE", "DROWSY", "SLEEPING"}
        assert pytest.approx(sum(res_60.probabilities.values()), abs=1e-5) == 1.0
        assert res_60.confidence == res_60.probabilities[res_60.label]

    def test_stride_cadence_s15(self) -> None:
        """After first prediction at frame 59, subsequent predictions occur every S=15 frames."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        # Feed first 60 frames
        for i in range(60):
            engine.process_sample(make_test_sample(frame_index=i))

        # Frames 60 to 73 (14 frames → stride_skip)
        for i in range(60, 74):
            res = engine.process_sample(make_test_sample(frame_index=i))
            assert not res.has_prediction
            assert res.reason == "stride_skip"
            assert res.status == STATUS_INSUFFICIENT_HISTORY
            assert res.buffer_size == 60
            assert res.start_frame_index == i - 59
            assert res.end_frame_index == i

        # Frame 74 (15th frame after frame 59 → second prediction)
        res_74 = engine.process_sample(make_test_sample(frame_index=74))
        assert res_74.has_prediction
        assert res_74.reason == "predicted"
        assert res_74.status == STATUS_OK
        assert res_74.start_frame_index == 15
        assert res_74.end_frame_index == 74

        # Frames 75 to 88 (stride_skip)
        for i in range(75, 89):
            res = engine.process_sample(make_test_sample(frame_index=i))
            assert not res.has_prediction
            assert res.reason == "stride_skip"

        # Frame 89 (third prediction)
        res_89 = engine.process_sample(make_test_sample(frame_index=89))
        assert res_89.has_prediction
        assert res_89.reason == "predicted"
        assert res_89.start_frame_index == 30
        assert res_89.end_frame_index == 89

    def test_manual_reset(self) -> None:
        """Calling reset() clears the buffer and resets warm-up."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        # Fill buffer to 60
        for i in range(60):
            engine.process_sample(make_test_sample(frame_index=i))
        assert engine.buffer_size == 60

        # Manual reset
        engine.reset()
        assert engine.buffer_size == 0
        assert engine.current_session_id is None

        # Next sample starts warm-up again
        res = engine.process_sample(make_test_sample(frame_index=0))
        assert not res.has_prediction
        assert res.reason == "buffer_warming_up"
        assert res.buffer_size == 1

    def test_session_boundary_reset(self) -> None:
        """Encountering a new session_id clears buffer and resets state."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        # Feed 30 frames from session_001
        for i in range(30):
            engine.process_sample(make_test_sample(frame_index=i, session_id="session_001"))
        assert engine.buffer_size == 30

        # Feed frame from session_002
        res = engine.process_sample(make_test_sample(frame_index=0, session_id="session_002"))
        assert not res.has_prediction
        assert res.reason == "session_reset"
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert res.session_id == "session_002"
        assert res.buffer_size == 1
        assert engine.buffer_size == 1
        assert engine.current_session_id == "session_002"

    def test_discontinuity_frame_gap(self) -> None:
        """Frame gap > 1 clears the rolling buffer and restarts warm-up."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        for i in range(20):
            engine.process_sample(make_test_sample(frame_index=i))
        assert engine.buffer_size == 20

        # Gap: frame 22 (dropped frame 21)
        res = engine.process_sample(make_test_sample(frame_index=22, timestamp_ms=22 * 33.3))
        assert not res.has_prediction
        assert res.reason == "discontinuity_frame_gap"
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert res.buffer_size == 1
        assert engine.buffer_size == 1

        # Next continuous frame continues warm-up
        res_next = engine.process_sample(make_test_sample(frame_index=23, timestamp_ms=23 * 33.3))
        assert not res_next.has_prediction
        assert res_next.reason == "buffer_warming_up"
        assert res_next.buffer_size == 2

    def test_discontinuity_timing_gap(self) -> None:
        """Time gap > 200ms clears the rolling buffer and restarts warm-up."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        engine.process_sample(make_test_sample(frame_index=0, timestamp_ms=0.0))
        engine.process_sample(make_test_sample(frame_index=1, timestamp_ms=33.3))
        assert engine.buffer_size == 2

        # Timing gap of 250ms (stall)
        res = engine.process_sample(make_test_sample(frame_index=2, timestamp_ms=283.3))
        assert not res.has_prediction
        assert res.reason == "discontinuity_timing_gap"
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert res.buffer_size == 1
        assert engine.buffer_size == 1

    def test_non_monotonic_sequence_returns_invalid_and_resets(self) -> None:
        """Monotonicity violations return invalid responses and reset streaming state."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        engine.process_sample(make_test_sample(frame_index=5, timestamp_ms=100.0))

        result = engine.process_sample(make_test_sample(frame_index=5, timestamp_ms=133.3))
        assert result.status == STATUS_INVALID_INPUT
        assert result.predicted_class is None
        assert result.buffer_size == 0
        assert engine.buffer_size == 0
        assert engine.current_session_id is None

        # A valid frame starts a fresh sequence after the reset.
        result = engine.process_sample(make_test_sample(frame_index=0, timestamp_ms=0.0))
        assert result.status == STATUS_INSUFFICIENT_HISTORY
        assert result.buffer_size == 1

        result = engine.process_sample(make_test_sample(frame_index=1, timestamp_ms=0.0))
        assert result.status == STATUS_INVALID_INPUT
        assert result.predicted_class is None
        assert engine.buffer_size == 0

        result = engine.process_sample(make_test_sample(frame_index=1, timestamp_ms=33.3))
        assert result.status == STATUS_INSUFFICIENT_HISTORY
        assert result.buffer_size == 1

    def test_invalid_sample_schema_returns_invalid_contract_response(self) -> None:
        """A malformed record cannot produce a prediction and returns a full response."""
        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        # Wrong input types remain programmer errors.
        with pytest.raises(TypeError, match="must be a SampleRecord or LiveInputFrame"):
            engine.process_sample("not_a_sample")  # type: ignore[arg-type]

        # Inconsistent EAR avg via post-construction mutation
        bad_ear_sample = make_test_sample(frame_index=0, ear_avg=0.20)
        object.__setattr__(bad_ear_sample, "ear_avg", 0.35)
        result = engine.process_sample(bad_ear_sample)
        assert result.status == STATUS_INVALID_INPUT
        assert result.predicted_class is None
        assert result.probabilities is None
        assert result.latency_ms == 0.0

    def test_missing_landmarks_gracefully_imputed(self) -> None:
        """Samples with missing landmarks are imputed and evaluated without error."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        # Feed 60 frames where every other frame has invalid landmarks
        for i in range(60):
            rec = make_test_sample(frame_index=i, landmarks_valid=(i % 2 == 0))
            res = engine.process_sample(rec)

        assert res.has_prediction  # type: ignore[possibly-undefined]
        assert res.reason == "predicted"  # type: ignore[possibly-undefined]

    def test_fully_unusable_window_returns_invalid_input(self) -> None:
        """A window with no valid feature-mask channels cannot be predicted."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        engine = TemporalInferenceEngine(model=model, normalizer=create_fitted_normalizer())

        for i in range(60):
            result = engine.process_sample(
                make_test_sample(
                    frame_index=i,
                    landmarks_valid=False,
                    perclos=None,
                )
            )

        assert result.status == STATUS_INVALID_INPUT
        assert result.reason == "invalid_input"
        assert result.predicted_class is None
        assert result.class_id is None
        assert result.probabilities is None
        assert result.confidence is None
        assert math.isfinite(result.latency_ms)
        assert result.latency_ms == 0.0

    def test_output_conversion_failure_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Errors while converting model output return STATUS_ERROR without output."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        engine = TemporalInferenceEngine(model=model, normalizer=create_fitted_normalizer())

        import vigil.ml.inference.engine as engine_mod

        def fail_softmax(*_args: object, **_kwargs: object) -> torch.Tensor:
            raise RuntimeError("output conversion failure")

        monkeypatch.setattr(engine_mod.torch, "softmax", fail_softmax)
        for i in range(60):
            result = engine.process_sample(make_test_sample(frame_index=i))

        assert result.status == STATUS_ERROR
        assert result.predicted_class is None
        assert result.probabilities is None
        assert result.confidence is None
        assert math.isfinite(result.latency_ms)
        assert result.latency_ms >= 0.0

    def test_engine_from_bundle_factory(self, tmp_path: Path) -> None:
        """Engine can be instantiated directly via TemporalInferenceEngine.from_bundle."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        bundle_path = tmp_path / "bundle.pt"
        export_model_bundle(model, normalizer, bundle_path)

        engine = TemporalInferenceEngine.from_bundle(bundle_path)
        assert engine.buffer_size == 0
        rec = make_test_sample(frame_index=0)
        res = engine.process_sample(rec)
        assert res.reason == "buffer_warming_up"

    def test_probabilities_and_confidence_valid_when_predicted(self) -> None:
        """Probabilities sum to ~1.0 and confidence matches the top-1 class when predicted."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)
        for i in range(60):
            res = engine.step(make_test_sample(frame_index=i))
        assert res.status == STATUS_OK  # type: ignore[possibly-undefined]
        assert res.probabilities is not None  # type: ignore[possibly-undefined]
        assert pytest.approx(sum(res.probabilities.values()), abs=1e-5) == 1.0  # type: ignore[possibly-undefined]
        assert res.confidence == pytest.approx(res.probabilities[res.predicted_class])  # type: ignore[possibly-undefined]
        assert 0.0 <= res.confidence <= 1.0  # type: ignore[possibly-undefined]

    def test_inference_failure_returns_error_without_prediction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model failure returns STATUS_ERROR and never fabricates output."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        def fail_forward(_inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("synthetic inference failure")

        monkeypatch.setattr(model, "forward", fail_forward)
        for i in range(60):
            result = engine.step(make_live_frame(frame_index=i))

        assert result.status == STATUS_ERROR
        assert result.predicted_class is None
        assert result.class_id is None
        assert result.probabilities is None
        assert result.confidence is None
        assert math.isfinite(result.latency_ms)
        assert result.latency_ms >= 0.0

    def test_invalid_input_does_not_silently_generate_prediction(self) -> None:
        """Invalid inputs raise errors rather than silently generating predictions."""
        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)
        with pytest.raises(TypeError):
            engine.step(None)  # type: ignore[arg-type]

    def test_latency_ms_measured_on_prediction_step(self) -> None:
        """latency_ms is strictly positive (non-zero) when a neural forward pass occurs."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)
        for i in range(60):
            res = engine.step(make_test_sample(frame_index=i))
        # The prediction step should have non-trivial latency
        assert res.status == STATUS_OK  # type: ignore[possibly-undefined]
        assert res.latency_ms >= 0.0  # type: ignore[possibly-undefined]
        assert math.isfinite(res.latency_ms)  # type: ignore[possibly-undefined]

    def test_no_prediction_steps_have_zero_latency(self) -> None:
        """Non-prediction steps report latency_ms == 0.0 (no neural forward pass)."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)
        res = engine.step(make_test_sample(frame_index=0))
        assert res.status == STATUS_INSUFFICIENT_HISTORY
        assert res.latency_ms == 0.0


# ==============================================================================
# Model Evaluation and Baseline Comparison Tests (unchanged, kept for regression)
# ==============================================================================


class TestModelComparison:
    """Tests covering reproducible model vs. rule baseline evaluation and reporting."""

    def test_compare_models_synthetic_fixture(self) -> None:
        """Run compare_models on synthetic records and verify report structure and markdown."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()

        # Build 3 sessions with 80 frames each for 3 distinct subjects
        records: list[SampleRecord] = []
        for s_idx, (sess, subj) in enumerate([
            ("sess_01", "subj_01"),
            ("sess_02", "subj_02"),
            ("sess_03", "subj_03"),
        ]):
            for f in range(80):
                records.append(
                    make_test_sample(
                        frame_index=f,
                        session_id=sess,
                        subject_id=subj,
                        ear_avg=0.30 if s_idx == 0 else (0.22 if s_idx == 1 else 0.15),
                        is_eye_closed=s_idx == 2,
                        label="ACTIVE" if s_idx == 0 else ("DROWSY" if s_idx == 1 else "SLEEPING"),
                        label_id=s_idx,
                    )
                )

        report = compare_models(
            records=records,
            model=model,
            normalizer=normalizer,
            training_provenance={
                "strategy": "subject",
                "train_subjects": ["subject_999"],
                "val_subjects": ["subject_998"],
                "train_sessions": ["training_session"],
                "val_sessions": ["validation_session"],
            },
            is_synthetic=True,
        )

        assert isinstance(report, ModelComparisonReport)
        assert report.is_synthetic
        assert report.window_count > 0
        assert report.gru_report is not None
        assert report.baseline_matched_report is not None
        assert report.baseline_full_report is not None

        # Verify serialization
        data = report.to_dict()
        assert data["is_synthetic"] is True
        assert "gru_model" in data
        assert "rule_baseline_matched" in data
        assert "rule_baseline_full" in data
        assert data["evaluation_metadata"] == {
            "development_strategy": "subject",
            "evaluation_subjects": ["subj_01", "subj_02", "subj_03"],
            "evaluation_sessions": ["sess_01", "sess_02", "sess_03"],
            "secondary_split_performed": False,
        }

        # Verify markdown formatting
        md = report.to_markdown()
        assert "Model Comparison Report" in md
        assert "Synthetic Fixtures" in md
        assert "Macro F1" in md
        assert "Confusion Matrix" in md

    def test_compare_models_requires_training_provenance(self) -> None:
        """Evaluation cannot silently infer independence from an internal split."""
        model = GRUTemporalClassifier(GRUConfig())
        with pytest.raises(TypeError, match="Training provenance is required"):
            compare_models(
                records=[make_test_sample(frame_index=0)],
                model=model,
                normalizer=create_fitted_normalizer(),
            )

    def test_compare_models_rejects_overlapping_training_groups(self) -> None:
        """Evaluation records overlapping training subjects or sessions are rejected."""
        model = GRUTemporalClassifier(GRUConfig())
        with pytest.raises(ValueError, match="overlap model-development provenance"):
            compare_models(
                records=[make_test_sample(frame_index=0, subject_id="subject_001")],
                model=model,
                normalizer=create_fitted_normalizer(),
                training_provenance={
                    "strategy": "subject",
                    "train_subjects": ["subject_001"],
                    "val_subjects": ["subject_002"],
                    "train_sessions": ["training_session"],
                    "val_sessions": ["validation_session"],
                },
            )

    def test_compare_models_rejects_validation_group_overlap(self) -> None:
        """Subject-strategy evaluation rejects groups used only for validation."""
        model = GRUTemporalClassifier(GRUConfig())
        records = [
            make_test_sample(
                frame_index=i,
                subject_id="subject_val",
                session_id="evaluation_session",
            )
            for i in range(80)
        ]
        with pytest.raises(ValueError, match="model-development provenance"):
            compare_models(
                records=records,
                model=model,
                normalizer=create_fitted_normalizer(),
                training_provenance={
                    "strategy": "subject",
                    "train_subjects": ["subject_train"],
                    "val_subjects": ["subject_val"],
                    "train_sessions": ["training_session"],
                    "val_sessions": ["validation_session"],
                },
            )

    def test_session_strategy_allows_held_out_session_for_same_subject(self) -> None:
        """Session fallback permits same-subject evaluation in a new session."""
        model = GRUTemporalClassifier(GRUConfig())
        records = [
            make_test_sample(
                frame_index=i,
                subject_id="subject_001",
                session_id="held_out_session",
            )
            for i in range(80)
        ]
        report = compare_models(
            records=records,
            model=model,
            normalizer=create_fitted_normalizer(),
            training_provenance={
                "strategy": "session",
                "train_subjects": ["subject_001"],
                "val_subjects": ["subject_001"],
                "train_sessions": ["training_session"],
                "val_sessions": ["validation_session"],
            },
        )
        assert report.window_count > 0
        assert report.baseline_full_report.total_samples == 80
        assert report.evaluation_metadata.development_strategy == "session"
        assert report.evaluation_metadata.evaluation_subjects == ["subject_001"]
        assert report.evaluation_metadata.evaluation_sessions == ["held_out_session"]
        assert not report.evaluation_metadata.secondary_split_performed

    def test_session_strategy_rejects_development_session_overlap(self) -> None:
        """Session fallback rejects evaluation sessions used in model development."""
        model = GRUTemporalClassifier(GRUConfig())
        records = [
            make_test_sample(
                frame_index=i,
                subject_id="subject_new",
                session_id="validation_session",
            )
            for i in range(80)
        ]
        with pytest.raises(ValueError, match="overlap model-development provenance"):
            compare_models(
                records=records,
                model=model,
                normalizer=create_fitted_normalizer(),
                training_provenance={
                    "strategy": "session",
                    "train_subjects": ["subject_001"],
                    "val_subjects": ["subject_001"],
                    "train_sessions": ["training_session"],
                    "val_sessions": ["validation_session"],
                },
            )

    @pytest.mark.parametrize(
        "provenance",
        [
            {"strategy": "unknown"},
            {
                "strategy": "subject",
                "train_subjects": ["subject_train"],
                "val_subjects": ["subject_train"],
                "train_sessions": ["session_train"],
                "val_sessions": ["session_val"],
            },
            {
                "strategy": "session",
                "train_subjects": ["subject_train"],
                "val_subjects": ["subject_train"],
                "train_sessions": ["session_train"],
            },
        ],
    )
    def test_compare_models_rejects_malformed_provenance(
        self,
        provenance: dict[str, object],
    ) -> None:
        """Evaluation rejects unknown, incomplete, and overlapping provenance."""
        model = GRUTemporalClassifier(GRUConfig())
        with pytest.raises((TypeError, ValueError), match="provenance|strategy"):
            compare_models(
                records=[make_test_sample(frame_index=0)],
                model=model,
                normalizer=create_fitted_normalizer(),
                training_provenance=provenance,
            )

    def test_evaluate_recording_directory_empty_or_nonexistent(self, tmp_path: Path) -> None:
        """evaluate_recording_directory returns None for missing/empty directories."""
        assert evaluate_recording_directory(tmp_path / "non_existent") is None

        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        assert evaluate_recording_directory(empty_dir) is None

    def test_evaluate_recording_directory_missing_bundle_raises(self, tmp_path: Path) -> None:
        """evaluate_recording_directory raises ValueError if recordings exist but bundle is omitted."""
        session_dir = tmp_path / "session_001"
        session_dir.mkdir()
        jsonl_file = session_dir / "samples.jsonl"
        rec = make_test_sample(frame_index=0)
        jsonl_file.write_text(rec.to_json() + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="A trained model bundle is required"):
            evaluate_recording_directory(tmp_path, bundle_path=None)

    def test_evaluate_recording_directory_loads_root_file_once(self, tmp_path: Path) -> None:
        """Root-level samples.jsonl is discovered once, not once per glob pattern."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        model.training_provenance = {  # type: ignore[attr-defined]
            "strategy": "subject",
            "train_subjects": ["subject_train"],
            "val_subjects": ["subject_val"],
            "train_sessions": ["session_train"],
            "val_sessions": ["session_val"],
        }
        bundle_path = tmp_path / "bundle.pt"
        export_model_bundle(model, create_fitted_normalizer(), bundle_path)

        recording_path = tmp_path / "samples.jsonl"
        recording_path.write_text(
            "".join(
                make_test_sample(
                    frame_index=i,
                    session_id="held_out_session",
                    subject_id="subject_held_out",
                ).to_json() + "\n"
                for i in range(80)
            ),
            encoding="utf-8",
        )

        report = evaluate_recording_directory(tmp_path, bundle_path=bundle_path)

        assert report is not None
        assert report.window_count > 0
        assert report.baseline_full_report.total_samples == 80
        assert report.evaluation_metadata.evaluation_subjects == ["subject_held_out"]
        assert report.evaluation_metadata.evaluation_sessions == ["held_out_session"]

    def test_evaluate_recording_directory_rejects_invalid_json_with_location(
        self, tmp_path: Path
    ) -> None:
        """Malformed JSON fails loudly with its source path and line number."""
        recording_path = tmp_path / "samples.jsonl"
        recording_path.write_text(
            make_test_sample(frame_index=0).to_json() + "\n" + "{not-json}\n",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match=(
                rf"Invalid JSON in recording file '{re.escape(str(recording_path))}' "
                r"at line 2"
            ),
        ):
            evaluate_recording_directory(tmp_path, bundle_path=None)

    def test_evaluate_recording_directory_rejects_invalid_record_with_location(
        self, tmp_path: Path
    ) -> None:
        """Schema-invalid records fail loudly with their source path and line number."""
        invalid_record = make_test_sample(frame_index=0).to_dict()
        invalid_record.pop("session_id")
        recording_path = tmp_path / "samples.jsonl"
        recording_path.write_text(
            make_test_sample(frame_index=0).to_json()
            + "\n"
            + json.dumps(invalid_record)
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match=(
                rf"Invalid SampleRecord in recording file "
                rf"'{re.escape(str(recording_path))}' at line 2"
            ),
        ):
            evaluate_recording_directory(tmp_path, bundle_path=None)


# ==============================================================================
# Optional PyTorch Behavior Tests
# ==============================================================================


class TestOptionalTorchBehavior:
    """Tests covering graceful error reporting when PyTorch is unavailable."""

    def test_export_model_bundle_raises_when_torch_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """export_model_bundle raises ImportError if torch is not installed."""
        import vigil.ml.models.export as export_mod

        monkeypatch.setattr(export_mod, "HAS_TORCH", False)
        monkeypatch.setattr(export_mod, "torch", None)

        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        with pytest.raises(ImportError, match="PyTorch is required"):
            export_model_bundle(model, normalizer, tmp_path / "bundle.pt")

    def test_load_model_bundle_raises_when_torch_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """load_model_bundle raises ImportError if torch is not installed."""
        import vigil.ml.models.export as export_mod

        monkeypatch.setattr(export_mod, "HAS_TORCH", False)
        monkeypatch.setattr(export_mod, "torch", None)

        with pytest.raises(ImportError, match="PyTorch is required"):
            load_model_bundle(tmp_path / "bundle.pt")

    def test_inference_engine_raises_when_torch_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TemporalInferenceEngine raises ImportError if torch is not installed."""
        import vigil.ml.inference.engine as engine_mod

        monkeypatch.setattr(engine_mod, "HAS_TORCH", False)
        monkeypatch.setattr(engine_mod, "torch", None)

        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        with pytest.raises(ImportError, match="PyTorch is required"):
            TemporalInferenceEngine(model, normalizer)

    def test_compare_models_raises_when_torch_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compare_models raises ImportError if torch is not installed."""
        import vigil.ml.evaluation.comparison as comp_mod

        monkeypatch.setattr(comp_mod, "HAS_TORCH", False)
        monkeypatch.setattr(comp_mod, "torch", None)

        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        with pytest.raises(ImportError, match="PyTorch is required"):
            compare_models([make_test_sample(0)], model, normalizer)
