"""Train-only feature normalizer for continuous ocular measurements.

Integrity rules:
    - Normalization parameters (mean, std) are fitted STRICTLY on training records.
    - Statistics are computed only from valid observed values (ignoring missing frames).
    - Binary indicator (is_eye_closed) and validity masks are NEVER normalized.
    - Handles constant features and all-missing features deterministically.
    - Missing observations on normalized features are imputed as 0.0 (representing the
      training mean), paired with explicit 0.0 validity masks so models distinguish
      true zero from missing data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vigil.ml.data.schema import SampleRecord
from vigil.ml.provenance_types import NormalizerProvenance, record_set_digest, sha256_digest

CONTINUOUS_FEATURE_NAMES: tuple[str, ...] = ("ear_avg", "blink_duration_ms", "perclos")


@dataclass
class FeatureNormalizer:
    """Standard scaler fitted strictly on valid continuous training observations."""

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    observed_counts: dict[str, int] = field(default_factory=dict)
    is_fitted: bool = False
    provenance: NormalizerProvenance | None = None

    @property
    def provenance_status(self) -> str:
        """Return verified lineage status without upgrading legacy bundles."""
        if self.provenance is None:
            return "legacy_unverified"
        return self.provenance.status

    def fit(
        self,
        training_records: list[SampleRecord],
        *,
        manifest_digest: str = "",
    ) -> FeatureNormalizer:
        """Fit means and standard deviations from valid training observations only.

        Args:
            training_records: List of SampleRecords from the training partition.

        Returns:
            self with fitted parameters.
        """
        if not training_records:
            raise ValueError("Cannot fit FeatureNormalizer on an empty training partition.")
        self.provenance = None
        values: dict[str, list[float]] = {name: [] for name in CONTINUOUS_FEATURE_NAMES}

        for r in training_records:
            if r.landmarks_valid and r.ear_avg is not None and math.isfinite(r.ear_avg):
                values["ear_avg"].append(float(r.ear_avg))

            if r.blink_duration_ms is not None and math.isfinite(r.blink_duration_ms):
                values["blink_duration_ms"].append(float(r.blink_duration_ms))

            if r.perclos is not None and math.isfinite(r.perclos):
                values["perclos"].append(float(r.perclos))

        for name in CONTINUOUS_FEATURE_NAMES:
            vals = values[name]
            count = len(vals)
            self.observed_counts[name] = count

            if count == 0:
                # All-missing feature in training split: fallback to identity scaling
                self.means[name] = 0.0
                self.stds[name] = 1.0
            else:
                m = sum(vals) / count
                var = sum((x - m) ** 2 for x in vals) / count
                s = math.sqrt(var)

                # Constant feature check: if variance is 0, use std=1.0 to avoid division by zero
                if s < 1e-6 or not math.isfinite(s):
                    s = 1.0

                self.means[name] = m
                self.stds[name] = s

        self.is_fitted = True
        self.provenance = NormalizerProvenance(
            fit_partition="train",
            fit_record_digest=record_set_digest(training_records),
            fit_record_count=len(training_records),
            fit_subjects=tuple(sorted({record.subject_id for record in training_records})),
            fit_sessions=tuple(sorted({record.session_id for record in training_records})),
            feature_channels=(
                "ear_avg",
                "is_eye_closed",
                "blink_duration_ms",
                "perclos",
                "ear_avg_valid",
                "is_eye_closed_valid",
                "blink_duration_valid",
                "perclos_valid",
            ),
            source_manifest_digest=manifest_digest,
            parameter_digest=sha256_digest({
                "means": self.means,
                "stds": self.stds,
                "observed_counts": self.observed_counts,
                "is_fitted": self.is_fitted,
            }),
        )
        return self

    def normalize_continuous(self, name: str, value: float | None, is_valid: bool) -> float:
        """Normalize a continuous measurement using frozen parameters.

        Args:
            name: Name of the continuous feature ('ear_avg', 'blink_duration_ms', or 'perclos').
            value: Observed value or None if missing.
            is_valid: Boolean indicating whether the observation is valid.

        Returns:
            Normalized float value (imputed as 0.0 if invalid or missing).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot transform before FeatureNormalizer has been fitted on training records.")

        if not is_valid or value is None or not math.isfinite(value):
            # Impute missing observation as 0.0 on the normalized scale (mean)
            return 0.0

        m = self.means[name]
        s = self.stds[name]
        return float((value - m) / s)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted parameters for logging or checkpoint export."""
        result = {
            "means": dict(self.means),
            "stds": dict(self.stds),
            "observed_counts": dict(self.observed_counts),
            "is_fitted": self.is_fitted,
        }
        if self.provenance is not None:
            result["provenance"] = self.provenance.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureNormalizer:
        """Restore normalizer from serialized parameter dictionary."""
        provenance_data = data.get("provenance")
        provenance = None
        if isinstance(provenance_data, dict):
            provenance = NormalizerProvenance(
                fit_partition=provenance_data.get("fit_partition", "train"),
                fit_record_digest=provenance_data.get("fit_record_digest", ""),
                fit_record_count=int(provenance_data.get("fit_record_count", 0)),
                fit_subjects=tuple(provenance_data.get("fit_subjects", [])),
                fit_sessions=tuple(provenance_data.get("fit_sessions", [])),
                feature_channels=tuple(provenance_data.get("feature_channels", ())),
                source_manifest_digest=provenance_data.get("source_manifest_digest", ""),
                parameter_digest=provenance_data.get("parameter_digest", ""),
                status=provenance_data.get("status", "legacy_unverified"),
            )
        return cls(
            means=dict(data.get("means", {})),
            stds=dict(data.get("stds", {})),
            observed_counts=dict(data.get("observed_counts", {})),
            is_fitted=bool(data.get("is_fitted", False)),
            provenance=provenance,
        )
