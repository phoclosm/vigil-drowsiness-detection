"""Temporal dataset, sliding windowing, and normalizer for Vigil M2."""

from vigil.ml.dataset.config import (
    LOCKED_FEATURE_CHANNELS,
    LOCKED_STRIDE,
    LOCKED_WINDOW_LENGTH,
    WindowConfig,
)
from vigil.ml.dataset.dataset import FatigueWindowDataset
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.dataset.splitter import DatasetGroupSplitResult, split_records_by_group
from vigil.ml.dataset.windowing import WindowSample, extract_windows

__all__ = [
    "LOCKED_FEATURE_CHANNELS",
    "LOCKED_STRIDE",
    "LOCKED_WINDOW_LENGTH",
    "DatasetGroupSplitResult",
    "FatigueWindowDataset",
    "FeatureNormalizer",
    "WindowConfig",
    "WindowSample",
    "extract_windows",
    "split_records_by_group",
]
