"""Model checkpoint serialization and deserialization for Vigil M2."""

from __future__ import annotations

import math
import re
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
from vigil.ml.dataset.normalizer import CONTINUOUS_FEATURE_NAMES, FeatureNormalizer
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier
from vigil.ml.provenance import validate_training_provenance

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


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")
    return value


def _validate_release_metadata(
    metadata: dict[str, Any],
    *,
    model: GRUTemporalClassifier | None = None,
    normalizer: FeatureNormalizer | None = None,
) -> None:
    """Validate optional release evidence without asserting approval."""
    release = metadata.get("release_metadata")
    if release is None:
        return
    if not isinstance(release, dict):
        raise TypeError("release_metadata must be a dictionary.")
    required = (
        "experiment_identity", "dataset_identity", "model_identity", "checkpoint",
        "source_revision", "is_synthetic", "approval_status",
    )
    missing = [key for key in required if key not in release]
    if missing:
        raise ValueError(f"release_metadata is missing required fields: {missing}.")
    if not isinstance(release["source_revision"], str) or not _GIT_SHA_RE.fullmatch(
        release["source_revision"]
    ):
        raise ValueError("release_metadata.source_revision must be a 40-character Git SHA.")
    if not isinstance(release["is_synthetic"], bool):
        raise TypeError("release_metadata.is_synthetic must be boolean.")
    if release["approval_status"] not in {"unapproved", "pending", "approved"}:
        raise ValueError("release_metadata.approval_status is invalid.")
    for key in ("experiment_identity", "dataset_identity", "model_identity", "checkpoint"):
        if not isinstance(release[key], dict):
            raise TypeError(f"release_metadata.{key} must be a dictionary.")
    experiment = release["experiment_identity"]
    for key in ("schema_version", "identity_digest"):
        if key not in experiment:
            raise ValueError(f"release_metadata.experiment_identity missing '{key}'.")
    _require_non_empty_text(experiment["schema_version"], "experiment_identity.schema_version")
    _require_sha256(experiment["identity_digest"], "experiment_identity.identity_digest")

    dataset = release["dataset_identity"]
    for key in (
        "dataset_name", "dataset_version", "manifest_digest",
        "label_mapping_digest", "partition_digest",
    ):
        if key not in dataset:
            raise ValueError(f"release_metadata.dataset_identity missing '{key}'.")
    for key in ("dataset_name", "dataset_version"):
        _require_non_empty_text(dataset[key], f"dataset_identity.{key}")
    for key in ("manifest_digest", "label_mapping_digest", "partition_digest"):
        _require_sha256(dataset[key], f"dataset_identity.{key}")

    model_identity = release["model_identity"]
    for key in ("model_type", "config_digest"):
        if key not in model_identity:
            raise ValueError(f"release_metadata.model_identity missing '{key}'.")
    _require_non_empty_text(model_identity["model_type"], "model_identity.model_type")
    _require_sha256(model_identity["config_digest"], "model_identity.config_digest")

    checkpoint = release["checkpoint"]
    for key in ("checkpoint_sha256", "artifact_sha256", "best_epoch"):
        if key not in checkpoint:
            raise ValueError(f"release_metadata.checkpoint missing '{key}'.")
    _require_sha256(checkpoint["checkpoint_sha256"], "checkpoint.checkpoint_sha256")
    _require_sha256(checkpoint["artifact_sha256"], "checkpoint.artifact_sha256")
    if isinstance(checkpoint["best_epoch"], bool) or not isinstance(
        checkpoint["best_epoch"], int
    ) or checkpoint["best_epoch"] < 0:
        raise ValueError("checkpoint.best_epoch must be a non-negative integer.")

    if release["approval_status"] == "approved":
        if release["is_synthetic"]:
            raise ValueError("Synthetic bundles cannot be marked approved.")
        if not isinstance(release.get("approval_evidence_ref"), str) or not release[
            "approval_evidence_ref"
        ].strip():
            raise ValueError("Approved release metadata requires approval_evidence_ref.")
        if model is None or normalizer is None:
            raise ValueError("Approved release metadata requires model and normalizer context.")
        if not isinstance(getattr(model, "training_provenance", None), dict):
            raise ValueError("Approved release metadata requires verified model-development provenance.")
        provenance = normalizer.provenance
        if provenance is None or provenance.status != "verified":
            raise ValueError("Approved release metadata requires verified normalizer provenance.")
        release_normalizer = release.get("normalizer_provenance")
        if not isinstance(release_normalizer, dict):
            raise ValueError("Approved release metadata requires normalizer_provenance.")
        if release_normalizer.get("status") != "verified":
            raise ValueError("Approved release metadata requires verified normalizer_provenance.")
        for key in ("source_manifest_digest", "training_partition_digest"):
            _require_sha256(release_normalizer.get(key), f"normalizer_provenance.{key}")


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

    if model.config.input_dim != len(LOCKED_FEATURE_CHANNELS):
        raise ValueError(
            "Cannot export a model with an input dimension that does not match the "
            f"locked feature contract ({len(LOCKED_FEATURE_CHANNELS)})."
        )
    if model.config.num_classes != len(CANONICAL_CLASS_LABELS):
        raise ValueError(
            "Cannot export a model with a class count that does not match the "
            f"canonical class contract ({len(CANONICAL_CLASS_LABELS)})."
        )

    if not isinstance(normalizer, FeatureNormalizer):
        raise TypeError(f"normalizer must be a FeatureNormalizer, got {type(normalizer).__name__}.")

    if not normalizer.is_fitted:
        raise ValueError("Cannot export an unfitted FeatureNormalizer.")

    dest_path = Path(filepath)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    bundle_metadata = dict(metadata) if metadata is not None else {}
    training_provenance = getattr(model, "training_provenance", None)
    if training_provenance is not None:
        bundle_metadata["training_provenance"] = validate_training_provenance(
            training_provenance
        )
    _validate_release_metadata(bundle_metadata, model=model, normalizer=normalizer)

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
        "normalizer": normalizer.to_dict(),
        "window_config": {
            "window_length": LOCKED_WINDOW_LENGTH,
            "stride": LOCKED_STRIDE,
            "max_frame_gap": 1,
            "max_time_gap_ms": 200.0,
        },
        "feature_channels": list(LOCKED_FEATURE_CHANNELS),
        "class_labels": dict(CANONICAL_CLASS_LABELS),
        "metadata": bundle_metadata,
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
        "model_type",
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

    if raw_bundle["model_type"] != "GRUTemporalClassifier":
        raise ValueError(
            "Unsupported model_type in bundle: expected 'GRUTemporalClassifier'."
        )

    # Validate feature channels
    if not isinstance(raw_bundle["feature_channels"], (list, tuple)):
        raise TypeError("feature_channels in bundle must be a list or tuple.")
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
    if m_cfg_dict.get("input_dim") != len(LOCKED_FEATURE_CHANNELS):
        raise ValueError("model_config.input_dim must match the locked feature-channel count.")
    if m_cfg_dict.get("num_classes") != len(CANONICAL_CLASS_LABELS):
        raise ValueError("model_config.num_classes must match the canonical class count.")

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
    for section in ("means", "stds", "observed_counts"):
        if not isinstance(norm_dict.get(section), dict):
            raise TypeError(f"normalizer.{section} in bundle must be a dictionary.")
    expected_norm_keys = set(CONTINUOUS_FEATURE_NAMES)
    if (
        set(norm_dict["means"]) != expected_norm_keys
        or set(norm_dict["stds"]) != expected_norm_keys
        or set(norm_dict["observed_counts"]) != expected_norm_keys
    ):
        raise ValueError(
            "normalizer must contain exactly the required continuous feature keys: "
            f"{sorted(expected_norm_keys)}."
        )
    for name in CONTINUOUS_FEATURE_NAMES:
        mean = norm_dict["means"][name]
        std = norm_dict["stds"][name]
        count = norm_dict["observed_counts"][name]
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean):
            raise ValueError(f"normalizer.means['{name}'] must be finite.")
        if isinstance(std, bool) or not isinstance(std, (int, float)) or not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"normalizer.stds['{name}'] must be finite and strictly positive.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"normalizer.observed_counts['{name}'] must be a non-negative integer.")

    normalizer = FeatureNormalizer.from_dict(norm_dict)
    normalizer.means = {k: float(v) for k, v in norm_dict.get("means", {}).items()}
    normalizer.stds = {k: float(v) for k, v in norm_dict.get("stds", {}).items()}
    normalizer.observed_counts = {
        k: int(v) for k, v in norm_dict.get("observed_counts", {}).items()
    }
    normalizer.is_fitted = True

    # Reconstruct window configuration
    w_cfg_dict = raw_bundle["window_config"]
    if not isinstance(w_cfg_dict, dict):
        raise TypeError("window_config in bundle must be a dictionary.")
    for key in ("window_length", "stride", "max_frame_gap", "max_time_gap_ms"):
        if key not in w_cfg_dict:
            raise ValueError(f"window_config is missing required key: '{key}'.")
    if (
        w_cfg_dict["window_length"] != LOCKED_WINDOW_LENGTH
        or w_cfg_dict["stride"] != LOCKED_STRIDE
    ):
        raise ValueError(
            f"window_config must use locked window_length={LOCKED_WINDOW_LENGTH} "
            f"and stride={LOCKED_STRIDE}."
        )
    window_config = WindowConfig(
        window_length=w_cfg_dict["window_length"],
        stride=w_cfg_dict["stride"],
        max_frame_gap=w_cfg_dict["max_frame_gap"],
        max_time_gap_ms=w_cfg_dict["max_time_gap_ms"],
    )

    if not isinstance(raw_bundle["class_labels"], dict):
        raise TypeError("class_labels in bundle must be a dictionary.")
    class_labels = raw_bundle["class_labels"]
    if (
        set(class_labels) != set(CANONICAL_CLASS_LABELS)
        or any(isinstance(key, bool) or not isinstance(key, int) for key in class_labels)
        or any(class_labels[key] != CANONICAL_CLASS_LABELS[key] for key in CANONICAL_CLASS_LABELS)
    ):
        raise ValueError(
            "class_labels must exactly match the canonical mapping "
            "{0: 'ACTIVE', 1: 'DROWSY', 2: 'SLEEPING'}."
        )
    metadata = raw_bundle.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("metadata in bundle must be a dictionary.")
    if "training_provenance" in metadata:
        model.training_provenance = validate_training_provenance(
            metadata["training_provenance"]
        )
    _validate_release_metadata(metadata, model=model, normalizer=normalizer)
    if "training_provenance" in metadata:
        metadata = dict(metadata)
        metadata["training_provenance"] = validate_training_provenance(
            metadata["training_provenance"]
        )

    return LoadedModelBundle(
        model=model,
        normalizer=normalizer,
        model_config=model_config,
        window_config=window_config,
        class_labels=class_labels,
        feature_channels=bundle_channels,
        bundle_version=bundle_version,
        metadata=metadata,
    )
