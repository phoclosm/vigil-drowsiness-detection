"""PyTorch-compatible dataset for Vigil M2 temporal fatigue windowing."""

from __future__ import annotations

import math
from typing import Any

from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS, WindowConfig
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.dataset.windowing import WindowSample, extract_windows

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    TorchDataset = object  # type: ignore[misc,assignment]
    HAS_TORCH = False


class FatigueWindowDataset(TorchDataset):
    """Deterministic PyTorch-compatible dataset built from Vigil M2 sample records.

    Contract:
        - Input tensor shape: (W, 8) where W = 60, dtype = torch.float32.
        - Output target: canonical class ID (0: ACTIVE, 1: DROWSY, 2: SLEEPING), dtype = torch.long.
        - Target is aligned strictly to the final sample in each window.
        - NaN and infinity values are strictly prevented.
        - Windows never cross session or subject boundaries.
    """

    def __init__(self, windows: list[WindowSample]) -> None:
        """Initialize the dataset with pre-extracted WindowSamples.

        Args:
            windows: List of WindowSample objects.
        """
        self.windows = list(windows)
        self.feature_names = LOCKED_FEATURE_CHANNELS
        self.window_length = 60
        self.feature_dim = len(LOCKED_FEATURE_CHANNELS)

        # Pre-validate windows for shape, channels, finite values, and canonical targets
        for i, w in enumerate(self.windows):
            if len(w.features) != self.window_length:
                raise ValueError(
                    f"Window {i} has invalid timestep count {len(w.features)}, expected {self.window_length}."
                )
            if w.target not in (0, 1, 2):
                raise ValueError(
                    f"Window {i} has non-canonical target {w.target}. Must be 0 (ACTIVE), 1 (DROWSY), or 2 (SLEEPING)."
                )
            for row in w.features:
                if len(row) != self.feature_dim:
                    raise ValueError(
                        f"Window {i} has unexpected channel count {len(row)}, expected {self.feature_dim}."
                    )
                for val in row:
                    if not math.isfinite(val):
                        raise ValueError(f"Non-finite value {val} detected in window {i}.")

    def __len__(self) -> int:
        """Return the number of extracted temporal windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Fetch the (features, target) pair for window index `idx`.

        Returns:
            A tuple of (features_tensor, target_tensor) where features_tensor is of
            shape (60, 8) with dtype torch.float32 and target_tensor is torch.long.
        """
        window = self.windows[idx]

        if len(window.features) != self.window_length:
            raise ValueError(
                f"Window at index {idx} has invalid timestep count {len(window.features)}, expected {self.window_length}."
            )
        if any(len(row) != self.feature_dim for row in window.features):
            raise ValueError(
                f"Window at index {idx} has invalid channel count, expected {self.feature_dim}."
            )
        if window.target not in (0, 1, 2):
            raise ValueError(
                f"Window at index {idx} has non-canonical target {window.target}. Expected 0, 1, or 2."
            )

        if HAS_TORCH:
            features = torch.tensor(window.features, dtype=torch.float32)
            target = torch.tensor(window.target, dtype=torch.long)
            return (features, target)

        # Fallback when torch is not installed
        return (window.features, window.target)

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Fetch metadata for window `idx` (subject, session, timestamps, frame bounds)."""
        w = self.windows[idx]
        return {
            "subject_id": w.subject_id,
            "session_id": w.session_id,
            "start_frame_index": w.start_frame_index,
            "end_frame_index": w.end_frame_index,
            "start_timestamp_ms": w.start_timestamp_ms,
            "end_timestamp_ms": w.end_timestamp_ms,
            "target": w.target,
        }

    @classmethod
    def from_records(
        cls,
        records: list[SampleRecord],
        config: WindowConfig | None = None,
        normalizer: FeatureNormalizer | None = None,
    ) -> FatigueWindowDataset:
        """Factory creating a dataset directly from raw SampleRecords.

        Args:
            records: List of SampleRecords.
            config: WindowConfig with window length, stride, and gap settings.
            normalizer: Optional fitted FeatureNormalizer.

        Returns:
            FatigueWindowDataset instance.
        """
        windows = extract_windows(records, config=config, normalizer=normalizer)
        return cls(windows=windows)
