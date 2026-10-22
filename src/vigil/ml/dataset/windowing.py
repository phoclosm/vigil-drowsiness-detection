"""Deterministic temporal sliding window extraction for Vigil M2."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from vigil.ml.data.schema import FatigueLabel, SampleRecord
from vigil.ml.dataset.config import WindowConfig
from vigil.ml.dataset.normalizer import FeatureNormalizer


def _iter_continuous_segments(
    records: list[SampleRecord],
    config: WindowConfig,
) -> tuple[list[tuple[str, str, list[SampleRecord]]], int]:
    """Validate records and yield deterministic continuous session segments."""
    if not records:
        return [], 0

    session_groups: dict[str, list[SampleRecord]] = defaultdict(list)
    session_to_subj: dict[str, str] = {}
    for record in records:
        if record.session_id in session_to_subj:
            if session_to_subj[record.session_id] != record.subject_id:
                raise ValueError(
                    f"Session '{record.session_id}' contains inconsistent subject IDs: "
                    f"'{session_to_subj[record.session_id]}' vs '{record.subject_id}'."
                )
        else:
            session_to_subj[record.session_id] = record.subject_id
        session_groups[record.session_id].append(record)

    segments: list[tuple[str, str, list[SampleRecord]]] = []
    temporal_gap_breaks = 0

    for session_id in sorted(session_groups):
        sorted_records = sorted(
            session_groups[session_id],
            key=lambda record: (record.frame_index, record.timestamp_ms),
        )

        for previous, current in pairwise(sorted_records):
            if current.frame_index == previous.frame_index:
                raise ValueError(
                    f"Duplicate frame index {current.frame_index} detected in session '{session_id}'."
                )
            if current.timestamp_ms <= previous.timestamp_ms:
                raise ValueError(
                    f"Non-monotonic timestamp detected in session '{session_id}': "
                    f"frame {previous.frame_index} ({previous.timestamp_ms}ms) followed by "
                    f"frame {current.frame_index} ({current.timestamp_ms}ms)."
                )

        current_segment: list[SampleRecord] = []
        for record in sorted_records:
            if not current_segment:
                current_segment.append(record)
                continue

            previous = current_segment[-1]
            frame_gap = record.frame_index - previous.frame_index
            time_gap_ms = record.timestamp_ms - previous.timestamp_ms
            if frame_gap > config.max_frame_gap or time_gap_ms > config.max_time_gap_ms:
                segments.append((session_id, session_to_subj[session_id], current_segment))
                temporal_gap_breaks += 1
                current_segment = [record]
            else:
                current_segment.append(record)

        if current_segment:
            segments.append((session_id, session_to_subj[session_id], current_segment))

    return segments, temporal_gap_breaks


@dataclass(frozen=True)
class WindowSample:
    """A single temporal window with locked dimensions (W, 8) and target label.

    Attributes:
        features: 2D list of shape (window_length, 8) containing float feature channels.
        target: Canonical class ID (0: ACTIVE, 1: DROWSY, 2: SLEEPING) of the final sample.
        subject_id: Subject identifier.
        session_id: Session identifier.
        start_frame_index: Frame index of the first sample in the window.
        end_frame_index: Frame index of the final sample in the window.
        start_timestamp_ms: Session-relative timestamp of the first sample.
        end_timestamp_ms: Session-relative timestamp of the final sample.
    """

    features: list[list[float]]
    target: int
    subject_id: str
    session_id: str
    start_frame_index: int
    end_frame_index: int
    start_timestamp_ms: float
    end_timestamp_ms: float

    def __post_init__(self) -> None:
        """Validate locked window dimensions, finite values, and canonical target."""
        if len(self.features) != 60:
            raise ValueError(
                f"WindowSample features must contain exactly 60 timesteps, got {len(self.features)}."
            )
        for t, row in enumerate(self.features):
            if len(row) != 8:
                raise ValueError(
                    f"WindowSample features row {t} must contain exactly 8 channels, got {len(row)}."
                )
            for val in row:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite value {val} detected in WindowSample features row {t}.")
        if self.target not in (
            FatigueLabel.ACTIVE,
            FatigueLabel.DROWSY,
            FatigueLabel.SLEEPING,
        ):
            raise ValueError(
                f"WindowSample target must be a canonical class (0: ACTIVE, 1: DROWSY, 2: SLEEPING), got {self.target}."
            )


@dataclass(frozen=True)
class WindowCandidate:
    """A candidate window and its canonical eligibility decision."""

    session_id: str
    records: list[SampleRecord]
    rejection_reason: str | None


def enumerate_window_candidates(
    records: list[SampleRecord],
    config: WindowConfig | None = None,
) -> tuple[list[WindowCandidate], int, int, int]:
    """Enumerate all candidates using production temporal and label rules."""
    if config is None:
        config = WindowConfig()
    segments, temporal_gap_breaks = _iter_continuous_segments(records, config)
    candidates: list[WindowCandidate] = []
    insufficient_history_segments = 0

    for session_id, _, segment in segments:
        if len(segment) < config.window_length:
            insufficient_history_segments += 1
            continue
        for start_idx in range(0, len(segment) - config.window_length + 1, config.stride):
            window_slice = segment[start_idx : start_idx + config.window_length]
            reason = (
                "ambiguous_label"
                if any(record.label_id == FatigueLabel.AMBIGUOUS for record in window_slice)
                else None
            )
            candidates.append(WindowCandidate(session_id, window_slice, reason))

    session_partition_count = len({session_id for session_id, _, _ in segments})
    return candidates, insufficient_history_segments, temporal_gap_breaks, session_partition_count


def build_sample_feature_vector(
    sample: SampleRecord,
    normalizer: FeatureNormalizer | None = None,
) -> list[float]:
    """Extract and format the exact 8 locked feature channels for a single sample.

    Channels:
        0. ear_avg (normalized, imputed as 0.0 if missing)
        1. is_eye_closed (binary indicator 0.0 or 1.0, 0.0 if missing)
        2. blink_duration_ms (normalized, imputed as 0.0 if missing)
        3. perclos (normalized, imputed as 0.0 if missing)
        4. ear_avg_valid (1.0 if valid, 0.0 if missing)
        5. is_eye_closed_valid (1.0 if valid, 0.0 if missing)
        6. blink_duration_valid (1.0 if valid, 0.0 if missing)
        7. perclos_valid (1.0 if valid, 0.0 if missing)
    """
    # 0 & 4: ear_avg
    ear_valid = bool(sample.landmarks_valid and sample.ear_avg is not None and math.isfinite(sample.ear_avg))
    if normalizer is not None:
        ear_val = normalizer.normalize_continuous("ear_avg", sample.ear_avg, ear_valid)
    else:
        ear_val = float(sample.ear_avg) if (ear_valid and sample.ear_avg is not None) else 0.0

    # 1 & 5: is_eye_closed
    eye_closed_valid = bool(sample.landmarks_valid and sample.is_eye_closed is not None)
    eye_closed_val = 1.0 if (eye_closed_valid and sample.is_eye_closed is True) else 0.0

    # 2 & 6: blink_duration_ms
    blink_valid = bool(sample.blink_duration_ms is not None and math.isfinite(sample.blink_duration_ms))
    if normalizer is not None:
        blink_val = normalizer.normalize_continuous("blink_duration_ms", sample.blink_duration_ms, blink_valid)
    else:
        blink_val = float(sample.blink_duration_ms) if (blink_valid and sample.blink_duration_ms is not None) else 0.0

    # 3 & 7: perclos
    perclos_valid = bool(sample.perclos is not None and math.isfinite(sample.perclos))
    if normalizer is not None:
        perclos_val = normalizer.normalize_continuous("perclos", sample.perclos, perclos_valid)
    else:
        perclos_val = float(sample.perclos) if (perclos_valid and sample.perclos is not None) else 0.0

    # Explicit masks: distinguish valid zero from unavailable observation
    return [
        float(ear_val),
        float(eye_closed_val),
        float(blink_val),
        float(perclos_val),
        1.0 if ear_valid else 0.0,
        1.0 if eye_closed_valid else 0.0,
        1.0 if blink_valid else 0.0,
        1.0 if perclos_valid else 0.0,
    ]


def extract_windows(
    records: list[SampleRecord],
    config: WindowConfig | None = None,
    normalizer: FeatureNormalizer | None = None,
) -> list[WindowSample]:
    """Extract deterministic sliding windows conforming to M2 D4 invariants.

    Invariants:
        - Windows NEVER cross session or subject boundaries.
        - Splits continuous segments on frame-index discontinuities (> max_frame_gap)
          and excessive timing stalls (> max_time_gap_ms).
        - Discards any window that overlaps an AMBIGUOUS annotation (label_id == -1).
        - Target is aligned strictly to the final sample's canonical label.
        - Emits no windows if a continuous segment has fewer than window_length samples.

    Args:
        records: List of SampleRecords.
        config: WindowConfig defining W, S, and gap thresholds.
        normalizer: Optional fitted FeatureNormalizer.

    Returns:
        List of WindowSample objects.
    """
    if config is None:
        config = WindowConfig()

    if not records:
        return []

    extracted_windows: list[WindowSample] = []
    candidates, _, _, _ = enumerate_window_candidates(records, config)

    for candidate in candidates:
        if candidate.rejection_reason is not None:
            continue
        window_slice = candidate.records
        final_sample = window_slice[-1]
        feature_matrix = [
            build_sample_feature_vector(sample, normalizer)
            for sample in window_slice
        ]
        first_sample = window_slice[0]
        extracted_windows.append(
            WindowSample(
                features=feature_matrix,
                target=int(final_sample.label_id),
                subject_id=final_sample.subject_id,
                session_id=candidate.session_id,
                start_frame_index=first_sample.frame_index,
                end_frame_index=final_sample.frame_index,
                start_timestamp_ms=first_sample.timestamp_ms,
                end_timestamp_ms=final_sample.timestamp_ms,
            )
        )

    return extracted_windows
