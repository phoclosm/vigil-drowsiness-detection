"""Training configuration and parameters for Vigil M2 temporal classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and configuration for reproducible temporal classifier training.

    Attributes:
        batch_size: Minibatch size for training and validation. Default 32.
        learning_rate: Initial learning rate for the Adam optimizer. Default 1e-3.
        max_epochs: Maximum training epochs. Default 30.
        weight_decay: L2 regularization penalty for Adam. Default 1e-4.
        seed: Deterministic random seed for Python, NumPy, PyTorch, and DataLoaders. Default 42.
        train_ratio: Proportion of groups assigned to training partition. Default 0.75.
        hidden_dim: Number of hidden units in the GRU cell. Default 64.
        output_dir: Optional filesystem directory path to save best model checkpoint. Default None.
    """

    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 30
    weight_decay: float = 1e-4
    seed: int = 42
    train_ratio: float = 0.75
    hidden_dim: int = 64
    output_dir: str | None = None

    def __post_init__(self) -> None:
        """Validate hyperparameter configuration."""
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise TypeError(f"batch_size must be an integer, got {type(self.batch_size).__name__}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")

        if not isinstance(self.learning_rate, (int, float)) or isinstance(self.learning_rate, bool):
            raise TypeError(f"learning_rate must be numeric, got {type(self.learning_rate).__name__}.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive and finite, got {self.learning_rate}.")

        if not isinstance(self.max_epochs, int) or isinstance(self.max_epochs, bool):
            raise TypeError(f"max_epochs must be an integer, got {type(self.max_epochs).__name__}.")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}.")

        if not isinstance(self.weight_decay, (int, float)) or isinstance(self.weight_decay, bool):
            raise TypeError(f"weight_decay must be numeric, got {type(self.weight_decay).__name__}.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative and finite, got {self.weight_decay}.")

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"seed must be an integer, got {type(self.seed).__name__}.")

        if not isinstance(self.train_ratio, (int, float)) or isinstance(self.train_ratio, bool):
            raise TypeError(f"train_ratio must be numeric, got {type(self.train_ratio).__name__}.")
        if not math.isfinite(self.train_ratio) or self.train_ratio <= 0.0 or self.train_ratio >= 1.0:
            raise ValueError(f"train_ratio must be a finite float in (0.0, 1.0), got {self.train_ratio}.")

        if not isinstance(self.hidden_dim, int) or isinstance(self.hidden_dim, bool):
            raise TypeError(f"hidden_dim must be an integer, got {type(self.hidden_dim).__name__}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")
