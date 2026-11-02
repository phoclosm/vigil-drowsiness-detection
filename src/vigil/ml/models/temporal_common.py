"""Shared validation helpers for causal temporal model candidates."""

from __future__ import annotations

from typing import Any

from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS, LOCKED_WINDOW_LENGTH

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def validate_temporal_input(x: Any, input_dim: int) -> None:
    """Validate the locked `(batch, 60, 8)` model input contract."""
    if torch is None or not isinstance(x, torch.Tensor):
        type_name = type(x).__name__
        raise TypeError(f"Input must be a torch.Tensor, got {type_name}.")
    if x.dim() != 3:
        raise ValueError(
            "Input tensor must be 3-dimensional with shape "
            f"(batch_size, {LOCKED_WINDOW_LENGTH}, {len(LOCKED_FEATURE_CHANNELS)}), "
            f"got shape {tuple(x.shape)}."
        )
    _, sequence_length, feature_count = x.shape
    if sequence_length != LOCKED_WINDOW_LENGTH:
        raise ValueError(
            f"Input sequence length must match locked window length "
            f"{LOCKED_WINDOW_LENGTH}, got {sequence_length}."
        )
    if feature_count != input_dim:
        raise ValueError(
            f"Input feature dimension must match locked channels {input_dim}, "
            f"got {feature_count}."
        )
    if not torch.isfinite(x).all():
        raise ValueError("Input tensor contains non-finite values (NaN or inf).")


def validate_logits(logits: Any) -> Any:
    """Reject non-finite model outputs before exposing them to callers."""
    if not torch.isfinite(logits).all():
        raise ValueError("Model produced non-finite logits (NaN or inf).")
    return logits
