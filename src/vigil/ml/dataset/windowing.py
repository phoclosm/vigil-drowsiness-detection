"""Deterministic temporal sliding window extraction for Vigil M2."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from vigil.ml.data.schema import FatigueLabel, SampleRecord
from vigil.ml.dataset.config import WindowConfig
from vigil.ml.dataset.normalizer import FeatureNormalizer


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

    # Group records by session_id
    session_groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for r in records:
        session_groups[r.session_id].append(r)

    extracted_windows: list[WindowSample] = []

    for session_id in sorted(session_groups.keys()):
        session_records = session_groups[session_id]

        # Sort strictly by frame_index and timestamp_ms
        sorted_records = sorted(
            session_records,
            key=lambda r: (r.frame_index, r.timestamp_ms),
        )

        # Break session into continuous segments based on frame and timing gaps
        segments: list[list[SampleRecord]] = []
        current_segment: list[SampleRecord] = []

        for r in sorted_records:
            if not current_segment:
                current_segment.append(r)
                continue

            prev = current_segment[-1]
            frame_gap = r.frame_index - prev.frame_index
            time_gap_ms = r.timestamp_ms - prev.timestamp_ms

            # Check continuity constraints
            if frame_gap > config.max_frame_gap or time_gap_ms > config.max_time_gap_ms:
                # Discontinuity detected: end segment and begin new segment
                segments.append(current_segment)
                current_segment = [r]
            else:
                current_segment.append(r)

        if current_segment:
            segments.append(current_segment)

        # Slide window across each continuous segment
        for segment in segments:
            seg_len = len(segment)
            if seg_len < config.window_length:
                # Emit no windows if segment is shorter than W
                continue

            for start_idx in range(0, seg_len - config.window_length + 1, config.stride):
                window_slice = segment[start_idx : start_idx + config.window_length]

                # Invariant: Discard windows that overlap any AMBIGUOUS annotation
                if any(r.label_id == FatigueLabel.AMBIGUOUS for r in window_slice):
                    continue

                # Final sample alignment
                final_sample = window_slice[-1]
                target_label_id = final_sample.label_id

                if target_label_id not in (
                    FatigueLabel.ACTIVE,
                    FatigueLabel.DROWSY,
                    FatigueLabel.SLEEPING,
                ):
                    # Target must be a canonical class
                    continue

                # Construct (60, 8) feature matrix
                feature_matrix: list[list[float]] = []
                for sample in window_slice:
                    feature_matrix.append(build_sample_feature_vector(sample, normalizer))

                first_sample = window_slice[0]
                extracted_windows.append(
                    WindowSample(
                        features=feature_matrix,
                        target=int(target_label_id),
                        subject_id=final_sample.subject_id,
                        session_id=session_id,
                        start_frame_index=first_sample.frame_index,
                        end_frame_index=final_sample.frame_index,
                        start_timestamp_ms=first_sample.timestamp_ms,
                        end_timestamp_ms=final_sample.timestamp_ms,
                    )
                )

    return extracted_windows
