"""Contract tests for compact Day 2 temporal model candidates."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vigil.ml.dataset.config import (  # noqa: E402
    LOCKED_FEATURE_CHANNELS,
    LOCKED_WINDOW_LENGTH,
)
from vigil.ml.models import (  # noqa: E402
    GRUConfig,
    GRUTemporalClassifier,
    LSTMConfig,
    LSTMTemporalClassifier,
    TCNConfig,
    TCNTemporalClassifier,
)


def test_default_tcn_receptive_field_covers_locked_window() -> None:
    config = TCNConfig()
    receptive_field = 1 + sum(
        (config.kernel_size - 1) * dilation for dilation in config.dilations
    )

    assert receptive_field == 63
    assert receptive_field >= LOCKED_WINDOW_LENGTH


@pytest.mark.parametrize(
    ("model_type", "config"),
    [
        (GRUTemporalClassifier, GRUConfig(hidden_dim=8)),
        (LSTMTemporalClassifier, LSTMConfig(hidden_dim=8)),
        (TCNTemporalClassifier, TCNConfig(hidden_dim=8)),
    ],
)
def test_temporal_candidates_share_locked_contract(model_type, config) -> None:
    torch.manual_seed(7)
    model = model_type(config)
    inputs = torch.randn(2, LOCKED_WINDOW_LENGTH, len(LOCKED_FEATURE_CHANNELS))

    logits = model(inputs)
    probabilities = model.predict_proba(inputs)
    predictions = model.predict(inputs)

    assert logits.shape == (2, 3)
    assert probabilities.shape == (2, 3)
    assert predictions.shape == (2,)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert torch.all((predictions >= 0) & (predictions <= 2))


@pytest.mark.parametrize(
    ("model_type", "config"),
    [
        (LSTMTemporalClassifier, LSTMConfig(hidden_dim=8)),
        (TCNTemporalClassifier, TCNConfig(hidden_dim=8)),
    ],
)
def test_candidate_initialization_is_deterministic(model_type, config) -> None:
    torch.manual_seed(11)
    first = model_type(config)
    torch.manual_seed(11)
    second = model_type(config)
    first_outputs = first(torch.zeros(1, LOCKED_WINDOW_LENGTH, len(LOCKED_FEATURE_CHANNELS)))
    second_outputs = second(torch.zeros(1, LOCKED_WINDOW_LENGTH, len(LOCKED_FEATURE_CHANNELS)))
    assert torch.equal(first_outputs, second_outputs)


@pytest.mark.parametrize(
    "model",
    [
        LSTMTemporalClassifier(LSTMConfig(hidden_dim=8)),
        TCNTemporalClassifier(TCNConfig(hidden_dim=8)),
    ],
)
def test_candidates_reject_invalid_windows(model) -> None:
    valid = torch.zeros(1, LOCKED_WINDOW_LENGTH, len(LOCKED_FEATURE_CHANNELS))
    with pytest.raises(ValueError, match="sequence length"):
        model(valid[:, :-1, :])
    with pytest.raises(ValueError, match="non-finite"):
        invalid = valid.clone()
        invalid[0, 0, 0] = float("nan")
        model(invalid)
