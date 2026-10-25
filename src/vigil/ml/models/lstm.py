"""Compact causal LSTM temporal sequence classifier for Vigil M2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS
from vigil.ml.models.temporal_common import validate_logits, validate_temporal_input

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
class LSTMConfig:
    """Architecture configuration for the causal LSTM candidate."""

    input_dim: int = len(LOCKED_FEATURE_CHANNELS)
    hidden_dim: int = 64
    num_layers: int = 1
    num_classes: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim != len(LOCKED_FEATURE_CHANNELS):
            raise ValueError(
                f"input_dim is locked to {len(LOCKED_FEATURE_CHANNELS)}, got {self.input_dim}."
            )
        if not isinstance(self.hidden_dim, int) or isinstance(self.hidden_dim, bool):
            raise TypeError("hidden_dim must be an integer.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not isinstance(self.num_layers, int) or isinstance(self.num_layers, bool):
            raise TypeError("num_layers must be an integer.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.num_classes != 3:
            raise ValueError("num_classes must be locked to 3 canonical classes.")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError("dropout must be numeric.")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0.0, 1.0).")


class LSTMTemporalClassifier(_BaseModule):
    """Strictly causal LSTM classifier with the shared model contract."""

    def __init__(self, config: LSTMConfig | None = None) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required to instantiate LSTMTemporalClassifier."
            )
        super().__init__()
        self.config = config or LSTMConfig()
        dropout_rate = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.fc = nn.Linear(self.config.hidden_dim, self.config.num_classes)

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTMTemporalClassifier forward pass.")
        validate_temporal_input(x, self.config.input_dim)
        output, _ = self.lstm(x)
        return validate_logits(self.fc(output[:, -1, :]))

    def predict_proba(self, x: Any) -> Any:
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: Any) -> Any:
        return torch.argmax(self.forward(x), dim=-1)
