"""Reproducible training and optimization API for Vigil M2 temporal sequence models."""

from vigil.ml.training.config import TrainingConfig
from vigil.ml.training.trainer import (
    TrainingResult,
    seed_everything,
    train_temporal_classifier,
)

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "seed_everything",
    "train_temporal_classifier",
]
