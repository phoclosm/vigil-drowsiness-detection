"""Unit tests for Vigil M2 temporal window dataset, windowing, and normalizer."""

from __future__ import annotations

import pytest
import torch

from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.config import (
    LOCKED_FEATURE_CHANNELS,
    LOCKED_STRIDE,
    LOCKED_WINDOW_LENGTH,
    WindowConfig,
)
from vigil.ml.dataset.dataset import FatigueWindowDataset
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.dataset.splitter import DatasetGroupSplitResult, split_records_by_group
from vigil.ml.dataset.windowing import (
    WindowSample,
    build_sample_feature_vector,
    extract_windows,
)


def make_sequence_sample(
    frame_index: int,
    session_id: str = "session_001",
    subject_id: str = "subject_001",
    timestamp_ms: float | None = None,
    face_detected: bool = True,
    landmarks_valid: bool = True,
    ear_avg: float | None = 0.30,
    is_eye_closed: bool | None = False,
    blink_duration_ms: float = 0.0,
    perclos: float | None = 0.05,
    label: str = "ACTIVE",
    label_id: int = 0,
) -> SampleRecord:
    t_ms = float(frame_index * 33.3) if timestamp_ms is None else timestamp_ms
    if not landmarks_valid:
        ear_avg = None
        is_eye_closed = None
    return SampleRecord(
        sample_id=f"{session_id}_f{frame_index}",
        session_id=session_id,
        subject_id=subject_id,
        frame_index=frame_index,
        timestamp_ms=t_ms,
        frame_delta_ms=None if frame_index == 0 else 33.3,
        face_detected=face_detected,
        landmarks_valid=landmarks_valid,
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


class TestWindowConfig:
    """Tests for WindowConfig locked parameters and validation."""

    def test_default_config_matches_locked_contract(self) -> None:
        cfg = WindowConfig()
        assert cfg.window_length == 60
        assert cfg.stride == 15
        assert cfg.max_frame_gap == 1
        assert cfg.max_time_gap_ms == 200.0

    def test_feature_channels_order_and_count(self) -> None:
        assert len(LOCKED_FEATURE_CHANNELS) == 8
        assert LOCKED_FEATURE_CHANNELS == (
            "ear_avg",
            "is_eye_closed",
            "blink_duration_ms",
            "perclos",
            "ear_avg_valid",
            "is_eye_closed_valid",
            "blink_duration_valid",
            "perclos_valid",
        )

    def test_invalid_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="window_length"):
            WindowConfig(window_length=0)
        with pytest.raises(ValueError, match="stride"):
            WindowConfig(stride=0)
        with pytest.raises(ValueError, match="max_frame_gap"):
            WindowConfig(max_frame_gap=0)
        with pytest.raises(ValueError, match="max_time_gap_ms"):
            WindowConfig(max_time_gap_ms=-50.0)


class TestFeatureNormalizer:
    """Tests for train-only FeatureNormalizer."""

    def test_fit_and_normalize_continuous_features(self) -> None:
        # Create training records with varied continuous values
        train_records = [
            make_sequence_sample(i, ear_avg=0.20 + (i * 0.01), blink_duration_ms=float(i * 10), perclos=float(i * 0.02))
            for i in range(10)
        ]
        norm = FeatureNormalizer().fit(train_records)

        assert norm.is_fitted is True
        assert norm.observed_counts["ear_avg"] == 10
        assert norm.means["ear_avg"] == pytest.approx(0.245, rel=1e-3)
        assert norm.stds["ear_avg"] > 0.0

        # Transforming a valid value
        val_norm = norm.normalize_continuous("ear_avg", norm.means["ear_avg"], is_valid=True)
        assert val_norm == pytest.approx(0.0, abs=1e-5)

        # Transforming missing value returns 0.0 imputation (the mean)
        val_missing = norm.normalize_continuous("ear_avg", None, is_valid=False)
        assert val_missing == 0.0

    def test_constant_feature_handling(self) -> None:
        # Constant ear_avg = 0.30
        train_records = [make_sequence_sample(i, ear_avg=0.30) for i in range(5)]
        norm = FeatureNormalizer().fit(train_records)

        assert norm.means["ear_avg"] == 0.30
        assert norm.stds["ear_avg"] == 1.0  # Safe constant feature fallback
        val_norm = norm.normalize_continuous("ear_avg", 0.30, is_valid=True)
        assert val_norm == 0.0

    def test_all_missing_feature_handling(self) -> None:
        # All records have landmarks_valid=False (ear_avg missing)
        train_records = [make_sequence_sample(i, landmarks_valid=False, ear_avg=None) for i in range(5)]
        norm = FeatureNormalizer().fit(train_records)

        assert norm.observed_counts["ear_avg"] == 0
        assert norm.means["ear_avg"] == 0.0
        assert norm.stds["ear_avg"] == 1.0
        assert norm.normalize_continuous("ear_avg", None, is_valid=False) == 0.0


class TestWindowExtractionIntegrity:
    """Tests for sliding window extraction and boundary containment."""

    def test_window_shape_stride_and_target_alignment(self) -> None:
        # 90 continuous samples in one session -> should produce 3 windows ((90-60)/15 + 1 = 3)
        records = [
            make_sequence_sample(
                i,
                label="DROWSY" if i >= 80 else "ACTIVE",
                label_id=1 if i >= 80 else 0,
            )
            for i in range(90)
        ]
        windows = extract_windows(records, WindowConfig(window_length=60, stride=15))

        assert len(windows) == 3

        # Window 0: frames 0..59, target is frame 59 label (ACTIVE = 0)
        w0 = windows[0]
        assert len(w0.features) == 60
        assert len(w0.features[0]) == 8
        assert w0.start_frame_index == 0
        assert w0.end_frame_index == 59
        assert w0.target == 0

        # Window 1: frames 15..74, target is frame 74 label (ACTIVE = 0)
        w1 = windows[1]
        assert w1.start_frame_index == 15
        assert w1.end_frame_index == 74
        assert w1.target == 0

        # Window 2: frames 30..89, target is frame 89 label (DROWSY = 1) -> strictly aligned to final sample!
        w2 = windows[2]
        assert w2.start_frame_index == 30
        assert w2.end_frame_index == 89
        assert w2.target == 1

    def test_windows_never_cross_session_boundaries(self) -> None:
        # Session 1: 50 samples (< 60) -> 0 windows
        # Session 2: 70 samples (>= 60) -> 1 window (frames 0..59)
        s1 = [make_sequence_sample(i, session_id="session_1") for i in range(50)]
        s2 = [make_sequence_sample(i, session_id="session_2") for i in range(70)]

        all_records = s1 + s2
        windows = extract_windows(all_records, WindowConfig(window_length=60, stride=15))

        assert len(windows) == 1
        assert windows[0].session_id == "session_2"

    def test_windows_split_at_frame_gaps(self) -> None:
        # Continuous frames 0..40, then gap to 45..110
        # Segment 1: frames 0..40 (41 frames < 60) -> 0 windows
        # Segment 2: frames 45..110 (66 frames >= 60) -> 1 window
        records_part1 = [make_sequence_sample(i) for i in range(41)]
        records_part2 = [make_sequence_sample(i, timestamp_ms=float(i * 33.3)) for i in range(45, 111)]

        windows = extract_windows(records_part1 + records_part2, WindowConfig(window_length=60, stride=15))

        assert len(windows) == 1
        assert windows[0].start_frame_index == 45
        assert windows[0].end_frame_index == 104

    def test_exclusion_of_windows_overlapping_ambiguous_labels(self) -> None:
        # 75 continuous samples with frame 30 labeled AMBIGUOUS (-1)
        records = [
            make_sequence_sample(
                i,
                label="AMBIGUOUS" if i == 30 else "ACTIVE",
                label_id=-1 if i == 30 else 0,
            )
            for i in range(75)
        ]
        # Potential windows without filtering:
        # Win 0: frames 0..59 (overlaps frame 30 -> MUST BE DISCARDED)
        # Win 1: frames 15..74 (overlaps frame 30 -> MUST BE DISCARDED)
        windows = extract_windows(records, WindowConfig(window_length=60, stride=15))

        assert len(windows) == 0

    def test_valid_zero_versus_missing_feature_masks(self) -> None:
        # Sample with valid zero PERCLOS (perclos=0.0)
        s_valid_zero = make_sequence_sample(0, perclos=0.0)
        vec_valid_zero = build_sample_feature_vector(s_valid_zero)
        # index 3 (perclos): 0.0, index 7 (perclos_valid): 1.0 (VALID ZERO!)
        assert vec_valid_zero[3] == 0.0
        assert vec_valid_zero[7] == 1.0

        # Sample with missing PERCLOS (perclos=None)
        s_missing = make_sequence_sample(1, perclos=None)
        vec_missing = build_sample_feature_vector(s_missing)
        # index 3: 0.0 (imputed), index 7 (perclos_valid): 0.0 (MASK DETECTS MISSING!)
        assert vec_missing[3] == 0.0
        assert vec_missing[7] == 0.0


class TestFatigueWindowDataset:
    """Tests for PyTorch dataset contract and tensor shapes."""

    def test_pytorch_tensor_shapes_dtypes_and_no_nan(self) -> None:
        records = [make_sequence_sample(i) for i in range(75)]
        normalizer = FeatureNormalizer().fit(records)
        ds = FatigueWindowDataset.from_records(records, normalizer=normalizer)

        assert len(ds) == 2  # (75 - 60)/15 + 1 = 2

        features, target = ds[0]

        # Shape contract: (60, 8)
        assert features.shape == (60, 8)
        assert features.dtype == torch.float32

        # Target contract: scalar int64
        assert target.dtype == torch.int64
        assert target.item() in (0, 1, 2)

        # Invariant: No NaN or infinity
        assert torch.isfinite(features).all().item() is True
        assert torch.isfinite(target).all().item() is True

        # Metadata access
        meta = ds.get_metadata(0)
        assert meta["subject_id"] == "subject_001"
        assert meta["start_frame_index"] == 0
        assert meta["end_frame_index"] == 59


class TestDatasetGroupSplitter:
    """Tests for leakage-safe grouped dataset splitting."""

    def test_subject_grouping_takes_precedence_over_session_grouping(self) -> None:
        # 3 subjects, each with 2 sessions of 10 samples
        records: list[SampleRecord] = []
        for subj_idx in range(1, 4):
            for sess_idx in range(1, 3):
                sess_id = f"s{subj_idx}_{sess_idx}"
                subj_id = f"subject_{subj_idx}"
                for frame in range(10):
                    records.append(make_sequence_sample(frame, session_id=sess_id, subject_id=subj_id))

        train_recs, val_recs, split_meta = split_records_by_group(records, train_ratio=0.70, seed=42)

        assert isinstance(split_meta, DatasetGroupSplitResult)
        assert split_meta.strategy == "subject"
        assert split_meta.subject_count == 3
        assert split_meta.session_count == 6

        # Cross-subject leakage check
        train_subjects = {r.subject_id for r in train_recs}
        val_subjects = {r.subject_id for r in val_recs}
        assert train_subjects.isdisjoint(val_subjects)
        assert len(train_subjects) + len(val_subjects) == 3

    def test_single_subject_falls_back_to_session_grouping(self) -> None:
        # 1 subject with 3 sessions
        records: list[SampleRecord] = []
        for sess_idx in range(1, 4):
            sess_id = f"sess_{sess_idx}"
            for frame in range(10):
                records.append(make_sequence_sample(frame, session_id=sess_id, subject_id="single_subj"))

        train_recs, val_recs, split_meta = split_records_by_group(records, train_ratio=0.70, seed=42)

        assert split_meta.strategy == "session"
        assert split_meta.subject_count == 1
        assert split_meta.session_count == 3
        assert "Single subject detected" in str(split_meta.limitation_note)

        # Check session disjointness
        train_sessions = {r.session_id for r in train_recs}
        val_sessions = {r.session_id for r in val_recs}
        assert train_sessions.isdisjoint(val_sessions)
        assert len(train_sessions) + len(val_sessions) == 3

    def test_single_session_reports_limitation_and_empty_validation(self) -> None:
        records = [make_sequence_sample(i, session_id="single_session", subject_id="subj_1") for i in range(10)]

        train_recs, val_recs, split_meta = split_records_by_group(records)

        assert split_meta.strategy == "none"
        assert len(train_recs) == 10
        assert len(val_recs) == 0
        assert "Evaluation partition is unavailable" in str(split_meta.limitation_note)

    def test_normalizer_fitted_strictly_on_train_records(self) -> None:
        """Verify normalizer parameters are fit strictly on training records."""
        train_records = [make_sequence_sample(i, ear_avg=0.25) for i in range(10)]
        val_records = [make_sequence_sample(i, ear_avg=0.40) for i in range(10)]

        # Fit strictly on train
        norm = FeatureNormalizer().fit(train_records)
        assert norm.means["ear_avg"] == 0.25  # Unaffected by val_records mean of 0.40

        # Apply unchanged to validation
        val_transformed = norm.normalize_continuous("ear_avg", val_records[0].ear_avg, is_valid=True)
        # (0.40 - 0.25) / 1.0 (std=1.0 for constant) = 0.15
        assert pytest.approx(val_transformed, rel=1e-3) == 0.15


class TestTemporalDatasetIntegrityRegression:
    """Regression tests for D4 hardening: session-subject consistency, sequence integrity, and locked contract."""

    def test_mixed_subject_session_rejected_in_splitter(self) -> None:
        """Reject splitting if a single session contains records from multiple subjects."""
        rec1 = make_sequence_sample(0, session_id="session_X", subject_id="subject_A")
        rec2 = make_sequence_sample(1, session_id="session_X", subject_id="subject_B")

        with pytest.raises(ValueError, match="inconsistent subject IDs"):
            split_records_by_group([rec1, rec2])

    def test_mixed_subject_session_rejected_in_windowing(self) -> None:
        """Reject window extraction if a single session contains records from multiple subjects."""
        rec1 = make_sequence_sample(0, session_id="session_X", subject_id="subject_A")
        rec2 = make_sequence_sample(1, session_id="session_X", subject_id="subject_B")

        with pytest.raises(ValueError, match="inconsistent subject IDs"):
            extract_windows([rec1, rec2])

    def test_duplicate_frame_index_rejected(self) -> None:
        """Reject temporal sequences containing duplicate frame indices within a session."""
        rec1 = make_sequence_sample(10, session_id="session_1", timestamp_ms=300.0)
        rec2 = make_sequence_sample(10, session_id="session_1", timestamp_ms=333.3)

        with pytest.raises(ValueError, match="Duplicate frame index 10"):
            extract_windows([rec1, rec2])

    def test_timestamp_reversal_rejected(self) -> None:
        """Reject temporal sequences where timestamps decrease when ordered by frame index."""
        rec1 = make_sequence_sample(10, session_id="session_1", timestamp_ms=500.0)
        rec2 = make_sequence_sample(11, session_id="session_1", timestamp_ms=450.0)

        with pytest.raises(ValueError, match="Non-monotonic timestamp"):
            extract_windows([rec1, rec2])

    def test_timestamp_stagnant_rejected(self) -> None:
        """Reject temporal sequences where timestamps remain identical across consecutive frames."""
        rec1 = make_sequence_sample(10, session_id="session_1", timestamp_ms=500.0)
        rec2 = make_sequence_sample(11, session_id="session_1", timestamp_ms=500.0)

        with pytest.raises(ValueError, match="Non-monotonic timestamp"):
            extract_windows([rec1, rec2])

    def test_valid_frame_gap_and_timing_gap_preserved(self) -> None:
        """Preserve valid frame-index gaps and timing-gap segmentation without error."""
        # Segment 1: frames 0..65 -> yields (66-60)/15 + 1 = 1 window
        # Gap: frame 65 (2164.5ms) to frame 70 (2800.0ms) -> frame gap 5 > 1 and time gap > 200ms
        # Segment 2: frames 70..135 -> yields (66-60)/15 + 1 = 1 window
        recs1 = [make_sequence_sample(i, timestamp_ms=float(i * 33.3)) for i in range(66)]
        recs2 = [make_sequence_sample(i, timestamp_ms=float(2500.0 + (i - 70) * 33.3)) for i in range(70, 136)]

        windows = extract_windows(recs1 + recs2)
        assert len(windows) == 2
        assert windows[0].start_frame_index == 0
        assert windows[0].end_frame_index == 59
        assert windows[1].start_frame_index == 70
        assert windows[1].end_frame_index == 129

    def test_locked_window_size_and_stride_enforced(self) -> None:
        """Enforce locked window_length=60 and stride=15 in WindowConfig."""
        assert LOCKED_WINDOW_LENGTH == 60
        assert LOCKED_STRIDE == 15

        cfg = WindowConfig()
        assert cfg.window_length == 60
        assert cfg.stride == 15

        with pytest.raises(ValueError, match="locked to 60"):
            WindowConfig(window_length=30)
        with pytest.raises(ValueError, match="locked to 60"):
            WindowConfig(window_length=120)

        with pytest.raises(ValueError, match="locked to 15"):
            WindowConfig(stride=5)
        with pytest.raises(ValueError, match="locked to 15"):
            WindowConfig(stride=30)

    def test_invalid_split_ratio_rejected(self) -> None:
        """Reject non-finite or out-of-range train_ratio values in split_records_by_group."""
        recs = [make_sequence_sample(i) for i in range(10)]

        with pytest.raises(ValueError, match="train_ratio"):
            split_records_by_group(recs, train_ratio=0.0)
        with pytest.raises(ValueError, match="train_ratio"):
            split_records_by_group(recs, train_ratio=1.0)
        with pytest.raises(ValueError, match="train_ratio"):
            split_records_by_group(recs, train_ratio=-0.25)
        with pytest.raises(ValueError, match="train_ratio"):
            split_records_by_group(recs, train_ratio=1.25)
        with pytest.raises(ValueError, match="train_ratio"):
            split_records_by_group(recs, train_ratio=float("nan"))
        with pytest.raises(TypeError, match="train_ratio"):
            split_records_by_group(recs, train_ratio="0.75")  # type: ignore[arg-type]

    def test_malformed_window_sample_rejected(self) -> None:
        """Validate timestep count, channel count, finite values, and canonical targets in WindowSample."""
        valid_features = [[0.0] * 8 for _ in range(60)]

        # Invalid timestep count (59 instead of 60)
        with pytest.raises(ValueError, match="60 timesteps"):
            WindowSample(
                features=[[0.0] * 8 for _ in range(59)],
                target=0,
                subject_id="sub_1",
                session_id="sess_1",
                start_frame_index=0,
                end_frame_index=58,
                start_timestamp_ms=0.0,
                end_timestamp_ms=1900.0,
            )

        # Invalid channel count (7 instead of 8)
        with pytest.raises(ValueError, match="8 channels"):
            WindowSample(
                features=[[0.0] * 7 for _ in range(60)],
                target=0,
                subject_id="sub_1",
                session_id="sess_1",
                start_frame_index=0,
                end_frame_index=59,
                start_timestamp_ms=0.0,
                end_timestamp_ms=1900.0,
            )

        # Non-finite value in features
        nan_features = [[0.0] * 8 for _ in range(60)]
        nan_features[10][2] = float("nan")
        with pytest.raises(ValueError, match="Non-finite value"):
            WindowSample(
                features=nan_features,
                target=0,
                subject_id="sub_1",
                session_id="sess_1",
                start_frame_index=0,
                end_frame_index=59,
                start_timestamp_ms=0.0,
                end_timestamp_ms=1900.0,
            )

        # Non-canonical target label (-1 or 99)
        with pytest.raises(ValueError, match="canonical class"):
            WindowSample(
                features=valid_features,
                target=-1,
                subject_id="sub_1",
                session_id="sess_1",
                start_frame_index=0,
                end_frame_index=59,
                start_timestamp_ms=0.0,
                end_timestamp_ms=1900.0,
            )
        with pytest.raises(ValueError, match="canonical class"):
            WindowSample(
                features=valid_features,
                target=3,
                subject_id="sub_1",
                session_id="sess_1",
                start_frame_index=0,
                end_frame_index=59,
                start_timestamp_ms=0.0,
                end_timestamp_ms=1900.0,
            )

    def test_dataset_validates_windows_in_init_and_getitem(self) -> None:
        """Validate that FatigueWindowDataset validates timestep count and canonical targets."""
        valid_features = [[0.0] * 8 for _ in range(60)]
        valid_w = WindowSample(
            features=valid_features,
            target=0,
            subject_id="sub_1",
            session_id="sess_1",
            start_frame_index=0,
            end_frame_index=59,
            start_timestamp_ms=0.0,
            end_timestamp_ms=1900.0,
        )

        ds = FatigueWindowDataset([valid_w])
        features, target = ds[0]
        assert features.shape == (60, 8)
        assert target.item() == 0

        # Artificially alter a window to test getitem guard
        class CorruptedWindow:
            def __init__(self, features: list[list[float]], target: int) -> None:
                self.features = features
                self.target = target

        corrupt_target_window = CorruptedWindow(valid_features, 99)
        with pytest.raises(ValueError, match="non-canonical target"):
            FatigueWindowDataset([corrupt_target_window])  # type: ignore[list-item]

        corrupt_steps_window = CorruptedWindow([[0.0] * 8 for _ in range(40)], 0)
        with pytest.raises(ValueError, match="invalid timestep count"):
            FatigueWindowDataset([corrupt_steps_window])  # type: ignore[list-item]
