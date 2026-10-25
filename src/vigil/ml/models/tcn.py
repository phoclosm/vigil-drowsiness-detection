"""Compact causal temporal convolutional classifier for Vigil M2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS
from vigil.ml.models.temporal_common import validate_logits, validate_temporal_input

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    HAS_TORCH = True
    _BaseModule = nn.Module
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False
    _BaseModule = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class TCNConfig:
    """Architecture configuration for the causal TCN candidate."""

    input_dim: int = len(LOCKED_FEATURE_CHANNELS)
    hidden_dim: int = 32
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
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
        if not isinstance(self.kernel_size, int) or isinstance(self.kernel_size, bool):
            raise TypeError("kernel_size must be an integer.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not self.dilations or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.dilations
        ):
            raise ValueError("dilations must contain positive integers.")
        if self.num_classes != 3:
            raise ValueError("num_classes must be locked to 3 canonical classes.")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError("dropout must be numeric.")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0.0, 1.0).")


class TCNTemporalClassifier(_BaseModule):
    """Strictly causal dilated convolutional classifier."""

    def __init__(self, config: TCNConfig | None = None) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to instantiate TCNTemporalClassifier.")
        super().__init__()
        self.config = config or TCNConfig()
        layers: list[nn.Module] = []
        channels = self.config.input_dim
        for dilation in self.config.dilations:
            layers.extend(
                [
                    nn.Conv1d(
                        channels,
                        self.config.hidden_dim,
                        self.config.kernel_size,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            channels = self.config.hidden_dim
        self.network = nn.ModuleList(layers)
        self.fc = nn.Linear(self.config.hidden_dim, self.config.num_classes)

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TCNTemporalClassifier forward pass.")
        validate_temporal_input(x, self.config.input_dim)
        output = x.transpose(1, 2)
        layer_index = 0
        for dilation in self.config.dilations:
            padding = dilation * (self.config.kernel_size - 1)
            output = F.pad(output, (padding, 0))
            output = self.network[layer_index](output)
            output = self.network[layer_index + 1](output)
            output = self.network[layer_index + 2](output)
            layer_index += 3
        logits = self.fc(output[:, :, -1])
        return validate_logits(logits)

    def predict_proba(self, x: Any) -> Any:
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: Any) -> Any:
        return torch.argmax(self.forward(x), dim=-1)
