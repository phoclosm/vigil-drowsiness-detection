"""Strict wire contract for unnormalized temporal features and predictions."""

from math import isclose
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Row = Annotated[list[Number], Field(min_length=8, max_length=8)]


class PredictRequest(BaseModel):
    """60 raw feature rows; only the server applies its bundle normalizer."""

    model_config = ConfigDict(extra="forbid")
    feature_channels: list[str]
    features: Annotated[list[Row], Field(min_length=60, max_length=60)]
    bundle_id: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    @model_validator(mode="after")
    def check_contract(self):
        if tuple(self.feature_channels) != LOCKED_FEATURE_CHANNELS:
            raise ValueError("incorrect feature order")
        for row in self.features:
            if any(row[i] not in (0.0, 1.0) for i in (1, 4, 5, 6, 7)):
                raise ValueError("indicators and masks must be binary")
            for value_index, mask_index in ((0, 4), (1, 5), (2, 6), (3, 7)):
                if row[mask_index] == 0 and row[value_index] != 0:
                    raise ValueError("missing values must use zero and a cleared mask")
            if row[5] != row[4]:
                raise ValueError("eye validity masks must agree")
            if row[4] and not 0 <= row[0] <= 0.60:
                raise ValueError("EAR outside sensor contract")
            if row[2] < 0 or not 0 <= row[3] <= 1:
                raise ValueError("raw duration or PERCLOS outside sensor contract")
        if not any(row[4] or row[5] or row[7] for row in self.features):
            raise ValueError("window contains no valid eye or PERCLOS observation")
        return self


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicted_class: Literal["ACTIVE", "DROWSY", "SLEEPING"]
    class_id: Annotated[int, Field(strict=True, ge=0, le=2)]
    confidence: Annotated[Number, Field(ge=0, le=1)]
    probabilities: dict[str, Number]
    latency_ms: Annotated[Number, Field(ge=0)]
    bundle_id: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    @model_validator(mode="after")
    def check_prediction(self):
        names = ("ACTIVE", "DROWSY", "SLEEPING")
        p = self.probabilities
        if set(p) != set(names) or names[self.class_id] != self.predicted_class:
            raise ValueError("invalid canonical classes")
        if any(not 0 <= x <= 1 for x in p.values()):
            raise ValueError("invalid probability range")
        if not isclose(sum(p.values()), 1, abs_tol=1e-5):
            raise ValueError("probabilities must sum to one")
        if not isclose(p[self.predicted_class], self.confidence, abs_tol=1e-7):
            raise ValueError("confidence differs from predicted probability")
        if not isclose(max(p.values()), self.confidence, abs_tol=1e-7):
            raise ValueError("predicted class must maximize probability")
        return self
