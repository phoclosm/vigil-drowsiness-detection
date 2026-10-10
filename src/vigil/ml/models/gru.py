"""Compact Gated Recurrent Unit (GRU) temporal sequence classifier for Vigil M2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vigil.ml.dataset.config import (
    LOCKED_FEATURE_CHANNELS,
    LOCKED_WINDOW_LENGTH,
)

try:
    import torch
    from torch import nn
    HAS_TORCH = True
    _BaseModule = nn.Module
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False
    _BaseModule = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class GRUConfig:
    """Architectural configuration for GRUTemporalClassifier.

    Attributes:
        input_dim: Number of input feature channels. Locked to 8.
        hidden_dim: Number of hidden units in the GRU cell. Default 64.
        num_layers: Number of stacked GRU layers. Default 1.
        num_classes: Number of output fatigue classes. Locked to 3 (ACTIVE, DROWSY, SLEEPING).
        dropout: Dropout probability applied between stacked layers (if num_layers > 1). Default 0.0.
        bidirectional: Whether to use a bidirectional GRU. Default False (strictly causal).
    """

    input_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 1
    num_classes: int = 3
    dropout: float = 0.0
    bidirectional: bool = False

    def __post_init__(self) -> None:
        """Validate architectural parameters."""
        if not isinstance(self.input_dim, int) or isinstance(self.input_dim, bool):
            raise TypeError(f"input_dim must be an integer, got {type(self.input_dim).__name__}.")
        if self.input_dim != len(LOCKED_FEATURE_CHANNELS):
            raise ValueError(
                f"input_dim is locked to {len(LOCKED_FEATURE_CHANNELS)} by the M2 feature contract, got {self.input_dim}."
            )

        if not isinstance(self.hidden_dim, int) or isinstance(self.hidden_dim, bool):
            raise TypeError(f"hidden_dim must be an integer, got {type(self.hidden_dim).__name__}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")

        if not isinstance(self.num_layers, int) or isinstance(self.num_layers, bool):
            raise TypeError(f"num_layers must be an integer, got {type(self.num_layers).__name__}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")

        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise TypeError(f"num_classes must be an integer, got {type(self.num_classes).__name__}.")
        if self.num_classes != 3:
            raise ValueError(f"num_classes must be locked to 3 canonical classes, got {self.num_classes}.")

        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError(f"dropout must be numeric, got {type(self.dropout).__name__}.")
        if not math.isfinite(self.dropout) or self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")

        if not isinstance(self.bidirectional, bool):
            raise TypeError(f"bidirectional must be a boolean, got {type(self.bidirectional).__name__}.")


class GRUTemporalClassifier(_BaseModule):
    """Compact GRU sequence classifier for temporal drowsiness detection.

    Contract:
        - Accepts batches shaped `(batch_size, 60, 8)` with `torch.float32`.
        - Emits raw logits shaped `(batch_size, 3)` suitable for `torch.nn.CrossEntropyLoss`.
        - Class ordering: `0: ACTIVE`, `1: DROWSY`, `2: SLEEPING`.
        - Softmax is NOT applied in `forward()`. Use `predict_proba()` for normalized probabilities.
        - Strict shape and non-finite validation.
    """

    def __init__(self, config: GRUConfig | None = None) -> None:
        """Initialize the GRU temporal sequence classifier.

        Args:
            config: Architectural configuration. Default uses GRUConfig(hidden_dim=64).

        Raises:
            ImportError: If PyTorch is not available.
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required to instantiate GRUTemporalClassifier. "
                "Install PyTorch with 'pip install torch'."
            )

        super().__init__()
        self.config = config if config is not None else GRUConfig()

        dropout_rate = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=self.config.bidirectional,
        )

        dir_factor = 2 if self.config.bidirectional else 1
        self.fc = nn.Linear(self.config.hidden_dim * dir_factor, self.config.num_classes)

    def forward(self, x: Any) -> Any:
        """Compute raw logits for a batch of temporal window tensors.

        Args:
            x: Input tensor shaped `(batch_size, 60, 8)` with dtype `torch.float32`.

        Returns:
            Raw unnormalized logits shaped `(batch_size, 3)` with dtype `torch.float32`.

        Raises:
            TypeError: If x is not a torch.Tensor.
            ValueError: If x has incorrect shape or non-finite elements.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GRUTemporalClassifier forward pass.")

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}.")

        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must be 3-dimensional with shape (batch_size, 60, 8), got shape {tuple(x.shape)}."
            )

        _batch_size, seq_len, num_features = x.shape

        if seq_len != LOCKED_WINDOW_LENGTH:
            raise ValueError(
                f"Input sequence length must match locked window length {LOCKED_WINDOW_LENGTH}, got {seq_len}."
            )

        if num_features != self.config.input_dim:
            raise ValueError(
                f"Input feature dimension must match locked channels {self.config.input_dim}, got {num_features}."
            )

        if not torch.isfinite(x).all():
            raise ValueError("Input tensor contains non-finite values (NaN or inf).")

        # Forward through recurrent backbone
        # out shape: (batch_size, seq_len, hidden_dim * dir_factor)
        out, _ = self.gru(x)

        # Final timestep representation (aligned with the end of the observation window)
        last_timestep = out[:, -1, :]

        # Compute raw logits
        logits = self.fc(last_timestep)

        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits (NaN or inf).")

        return logits

    def predict_proba(self, x: Any) -> Any:
        """Compute softmax probability distributions over the 3 canonical fatigue classes.

        Args:
            x: Input tensor shaped `(batch_size, 60, 8)`.

        Returns:
            Normalized class probabilities shaped `(batch_size, 3)`.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: Any) -> Any:
        """Predict the canonical class ID (0, 1, or 2) with highest logit for each window.

        Args:
            x: Input tensor shaped `(batch_size, 60, 8)`.

        Returns:
            Predicted class tensor shaped `(batch_size,)` with dtype `torch.long`.
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
