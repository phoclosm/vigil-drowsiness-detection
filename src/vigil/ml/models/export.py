"""Model checkpoint serialization and deserialization for Vigil M2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigil.ml.dataset.config import (
    LOCKED_FEATURE_CHANNELS,
    LOCKED_STRIDE,
    LOCKED_WINDOW_LENGTH,
    WindowConfig,
)
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

BUNDLE_SCHEMA_VERSION = "1.0.0"

CANONICAL_CLASS_LABELS: dict[int, str] = {
    0: "ACTIVE",
    1: "DROWSY",
    2: "SLEEPING",
}


@dataclass(frozen=True)
class LoadedModelBundle:
    """Self-contained bundle containing a loaded model, normalizer, and metadata.

    Attributes:
        model: Restored GRUTemporalClassifier in evaluation mode.
        normalizer: Restored and fitted FeatureNormalizer.
        model_config: Model architectural configuration.
        window_config: Window configuration used for temporal segmentation.
        class_labels: Mapping from canonical class IDs to names.
        feature_channels: Ordered tuple of feature channel names.
        bundle_version: Schema version of the checkpoint bundle.
        metadata: User and environment metadata stored with the bundle.
    """

    model: GRUTemporalClassifier
    normalizer: FeatureNormalizer
    model_config: GRUConfig
    window_config: WindowConfig
    class_labels: dict[int, str]
    feature_channels: tuple[str, ...]
    bundle_version: str
    metadata: dict[str, Any]

    @classmethod
    def from_bundle(cls, filepath: str | Path, map_location: Any = "cpu") -> LoadedModelBundle:
        """Load a bundle from disk."""
        return load_model_bundle(filepath, map_location=map_location)


def export_model_bundle(
    model: GRUTemporalClassifier,
    normalizer: FeatureNormalizer,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a self-contained, versioned model checkpoint bundle.

    The exported bundle includes:
        - Model weights (`state_dict`).
        - Model architecture configuration.
        - Fitted `FeatureNormalizer` parameters (means, stds, counts).
        - Locked 8-channel feature metadata.
        - Window configuration (W=60, S=15).
        - Canonical class mapping.
        - Schema version and creation timestamp.

    Args:
        model: Trained GRUTemporalClassifier instance.
        normalizer: Fitted FeatureNormalizer instance.
        filepath: Filesystem path to save the `.pt` bundle.
        metadata: Optional dictionary with user-defined tracking metadata.

    Returns:
        Path object pointing to the written bundle.

    Raises:
        ImportError: If PyTorch is unavailable.
        TypeError: If model or normalizer have invalid types.
        ValueError: If the normalizer is not fitted.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError(
            "PyTorch is required for model bundle export. "
            "Install PyTorch with 'pip install torch'."
        )

    if not isinstance(model, GRUTemporalClassifier):
        raise TypeError(f"model must be a GRUTemporalClassifier, got {type(model).__name__}.")

    if not isinstance(normalizer, FeatureNormalizer):
        raise TypeError(f"normalizer must be a FeatureNormalizer, got {type(normalizer).__name__}.")

    if not normalizer.is_fitted:
        raise ValueError("Cannot export an unfitted FeatureNormalizer.")

    dest_path = Path(filepath)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    bundle_dict = {
        "bundle_version": BUNDLE_SCHEMA_VERSION,
        "model_type": "GRUTemporalClassifier",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": {
            "input_dim": model.config.input_dim,
            "hidden_dim": model.config.hidden_dim,
            "num_layers": model.config.num_layers,
            "num_classes": model.config.num_classes,
            "dropout": model.config.dropout,
            "bidirectional": model.config.bidirectional,
        },
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "normalizer": {
            "means": dict(normalizer.means),
            "stds": dict(normalizer.stds),
            "observed_counts": dict(normalizer.observed_counts),
            "is_fitted": normalizer.is_fitted,
        },
        "window_config": {
            "window_length": LOCKED_WINDOW_LENGTH,
            "stride": LOCKED_STRIDE,
            "max_frame_gap": 1,
            "max_time_gap_ms": 200.0,
        },
        "feature_channels": list(LOCKED_FEATURE_CHANNELS),
        "class_labels": dict(CANONICAL_CLASS_LABELS),
        "metadata": dict(metadata) if metadata is not None else {},
    }

    torch.save(bundle_dict, dest_path)
    return dest_path


def load_model_bundle(
    filepath: str | Path,
    map_location: str | Any = "cpu",
) -> LoadedModelBundle:
    """Load a versioned model bundle and restore model and normalizer.

    Args:
        filepath: Path to the `.pt` bundle file.
        map_location: Device to load tensors onto (default "cpu").

    Returns:
        LoadedModelBundle with restored components and validated metadata.

    Raises:
        ImportError: If PyTorch is unavailable.
        FileNotFoundError: If the bundle file does not exist.
        ValueError: If the bundle format or version is incompatible or corrupted.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError(
            "PyTorch is required to load a model bundle. "
            "Install PyTorch with 'pip install torch'."
        )

    bundle_path = Path(filepath)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at: {bundle_path}")

    try:
        raw_bundle = torch.load(bundle_path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to deserialize model bundle from {bundle_path}: {e}") from e

    if not isinstance(raw_bundle, dict):
        raise TypeError(f"Invalid model bundle structure: expected dict, got {type(raw_bundle).__name__}.")

    # Validate required keys
    required_keys = (
        "bundle_version",
        "model_config",
        "state_dict",
        "normalizer",
        "window_config",
        "feature_channels",
        "class_labels",
    )
    for key in required_keys:
        if key not in raw_bundle:
            raise ValueError(f"Model bundle is missing required section: '{key}'.")

    # Validate schema version compatibility
    bundle_version = str(raw_bundle["bundle_version"])
    bundle_major = bundle_version.split(".")[0]
    supported_major = BUNDLE_SCHEMA_VERSION.split(".")[0]
    if bundle_major != supported_major:
        raise ValueError(
            f"Incompatible bundle version '{bundle_version}'. "
            f"Supported major version is {supported_major}.x."
        )

    # Validate feature channels
    bundle_channels = tuple(raw_bundle["feature_channels"])
    if bundle_channels != LOCKED_FEATURE_CHANNELS:
        raise ValueError(
            f"Bundle feature channels do not match locked contract: "
            f"expected {LOCKED_FEATURE_CHANNELS}, got {bundle_channels}."
        )

    # Reconstruct model configuration and model
    m_cfg_dict = raw_bundle["model_config"]
    if not isinstance(m_cfg_dict, dict):
        raise TypeError("model_config in bundle must be a dictionary.")

    model_config = GRUConfig(
        input_dim=m_cfg_dict.get("input_dim", len(LOCKED_FEATURE_CHANNELS)),
        hidden_dim=m_cfg_dict.get("hidden_dim", 64),
        num_layers=m_cfg_dict.get("num_layers", 1),
        num_classes=m_cfg_dict.get("num_classes", 3),
        dropout=m_cfg_dict.get("dropout", 0.0),
        bidirectional=m_cfg_dict.get("bidirectional", False),
    )

    model = GRUTemporalClassifier(model_config)
    try:
        model.load_state_dict(raw_bundle["state_dict"])
    except Exception as e:
        raise ValueError(f"Failed to restore model state dictionary: {e}") from e
    model.eval()

    # Reconstruct normalizer
    norm_dict = raw_bundle["normalizer"]
    if not isinstance(norm_dict, dict) or not norm_dict.get("is_fitted", False):
        raise ValueError("normalizer in bundle must be a dictionary with is_fitted=True.")

    normalizer = FeatureNormalizer()
    normalizer.means = {k: float(v) for k, v in norm_dict.get("means", {}).items()}
    normalizer.stds = {k: float(v) for k, v in norm_dict.get("stds", {}).items()}
    normalizer.observed_counts = {k: int(v) for k, v in norm_dict.get("observed_counts", {}).items()}
    normalizer.is_fitted = True

    # Reconstruct window configuration
    w_cfg_dict = raw_bundle["window_config"]
    window_config = WindowConfig(
        window_length=w_cfg_dict.get("window_length", LOCKED_WINDOW_LENGTH),
        stride=w_cfg_dict.get("stride", LOCKED_STRIDE),
        max_frame_gap=w_cfg_dict.get("max_frame_gap", 1),
        max_time_gap_ms=w_cfg_dict.get("max_time_gap_ms", 200.0),
    )

    class_labels = {int(k): str(v) for k, v in raw_bundle["class_labels"].items()}

    return LoadedModelBundle(
        model=model,
        normalizer=normalizer,
        model_config=model_config,
        window_config=window_config,
        class_labels=class_labels,
        feature_channels=bundle_channels,
        bundle_version=bundle_version,
        metadata=raw_bundle.get("metadata", {}),
    )
