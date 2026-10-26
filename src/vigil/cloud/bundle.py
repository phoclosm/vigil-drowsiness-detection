"""Swappable stateless adapter around the existing GRU bundle exporter."""

from pathlib import Path
from threading import Lock
from time import perf_counter

from vigil.cloud.contracts import PredictRequest, PredictResponse
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS
from vigil.ml.models.export import load_model_bundle


class BundlePredictor:
    """Validate a trusted bundle once; normalize raw rows exactly once."""

    def __init__(self, path: str | Path):
        path = Path(path)
        # hashlib.file_digest is unavailable on Python 3.10.
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.bundle_id = digest.hexdigest()
        self.bundle = load_model_bundle(path)
        if self.bundle.feature_channels != LOCKED_FEATURE_CHANNELS:
            raise ValueError("bundle feature order mismatch")
        self._lock = Lock()

    def predict(self, request: PredictRequest) -> PredictResponse:
        import torch

        if request.bundle_id != self.bundle_id:
            raise ValueError("bundle identifier mismatch")
        start = perf_counter()
        rows = [list(row) for row in request.features]
        for row in rows:
            for i, mask, name in (
                (0, 4, "ear_avg"),
                (2, 6, "blink_duration_ms"),
                (3, 7, "perclos"),
            ):
                row[i] = self.bundle.normalizer.normalize_continuous(
                    name, row[i], bool(row[mask])
                )
        with self._lock, torch.inference_mode():
            logits = self.bundle.model(torch.tensor([rows], dtype=torch.float32))
            values = torch.softmax(logits, dim=-1)[0].tolist()
        labels = ("ACTIVE", "DROWSY", "SLEEPING")
        index = max(range(3), key=values.__getitem__)
        return PredictResponse(
            predicted_class=labels[index],
            class_id=index,
            confidence=values[index],
            probabilities=dict(zip(labels, values)),
            latency_ms=(perf_counter() - start) * 1000,
            bundle_id=self.bundle_id,
        )
