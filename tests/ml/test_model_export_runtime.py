"""Unit and integration tests for Vigil M2 model export, runtime inference, and evaluation comparison."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vigil.ml.data.schema import SampleRecord
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


def create_fitted_normalizer() -> FeatureNormalizer:
    """Create and fit a FeatureNormalizer with deterministic reference values."""
    normalizer = FeatureNormalizer()
    normalizer.means = {"ear_avg": 0.28, "blink_duration_ms": 150.0, "perclos": 0.12}
    normalizer.stds = {"ear_avg": 0.05, "blink_duration_ms": 50.0, "perclos": 0.08}
    normalizer.observed_counts = {"ear_avg": 100, "blink_duration_ms": 100, "perclos": 100}
    normalizer.is_fitted = True
    return normalizer


# ==============================================================================
# Model Export and Loading Tests
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


# ==============================================================================
# Stateful Runtime Inference Engine Tests
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
            assert res.buffer_size == i + 1
            assert engine.buffer_size == i + 1

        # Frame 59 (60th frame -> first full window)
        rec_60 = make_test_sample(frame_index=59)
        res_60 = engine.process_sample(rec_60)
        assert res_60.has_prediction
        assert res_60.label in CANONICAL_CLASS_LABELS.values()
        assert res_60.label_id in (0, 1, 2)
        assert res_60.reason == "predicted"
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

        # Frames 60 to 73 (14 frames -> stride_skip)
        for i in range(60, 74):
            res = engine.process_sample(make_test_sample(frame_index=i))
            assert not res.has_prediction
            assert res.reason == "stride_skip"
            assert res.buffer_size == 60
            assert res.start_frame_index == i - 59
            assert res.end_frame_index == i

        # Frame 74 (15th frame after frame 59 -> second prediction)
        res_74 = engine.process_sample(make_test_sample(frame_index=74))
        assert res_74.has_prediction
        assert res_74.reason == "predicted"
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
        assert res.buffer_size == 1
        assert engine.buffer_size == 1

    def test_non_monotonic_sequence_rejected(self) -> None:
        """Duplicate or decreasing frame indices or timestamps must raise ValueError."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        engine.process_sample(make_test_sample(frame_index=5, timestamp_ms=100.0))

        # Duplicate frame index
        with pytest.raises(ValueError, match="Duplicate or decreasing frame index"):
            engine.process_sample(make_test_sample(frame_index=5, timestamp_ms=133.3))

        # Decreasing frame index
        with pytest.raises(ValueError, match="Duplicate or decreasing frame index"):
            engine.process_sample(make_test_sample(frame_index=4, timestamp_ms=133.3))

        # Non-monotonic timestamp
        with pytest.raises(ValueError, match="Non-monotonic timestamp detected"):
            engine.process_sample(make_test_sample(frame_index=6, timestamp_ms=90.0))

    def test_invalid_sample_schema_rejected(self) -> None:
        """Passing an invalid record or wrong type raises immediately."""
        model = GRUTemporalClassifier(GRUConfig())
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        with pytest.raises(TypeError, match="sample must be a SampleRecord"):
            engine.process_sample("not_a_sample")  # type: ignore[arg-type]

        # Inconsistent EAR avg via post-construction mutation
        bad_ear_sample = make_test_sample(frame_index=0, ear_avg=0.20)
        object.__setattr__(bad_ear_sample, "ear_avg", 0.35)
        with pytest.raises(ValueError, match="Inconsistent ear_avg"):
            engine.process_sample(bad_ear_sample)

    def test_missing_landmarks_gracefully_imputed(self) -> None:
        """Samples with missing landmarks are imputed and evaluated without error."""
        model = GRUTemporalClassifier(GRUConfig(input_dim=8, hidden_dim=16, num_classes=3))
        normalizer = create_fitted_normalizer()
        engine = TemporalInferenceEngine(model=model, normalizer=normalizer)

        # Feed 60 frames where every other frame has invalid landmarks
        for i in range(60):
            rec = make_test_sample(frame_index=i, landmarks_valid=(i % 2 == 0))
            res = engine.process_sample(rec)

        assert res.has_prediction
        assert res.reason == "predicted"

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


# ==============================================================================
# Model Evaluation and Baseline Comparison Tests
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

        # Verify markdown formatting
        md = report.to_markdown()
        assert "Model Comparison Report" in md
        assert "Synthetic Fixtures" in md
        assert "Macro F1" in md
        assert "Confusion Matrix" in md

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

