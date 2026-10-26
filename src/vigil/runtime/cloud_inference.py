"""Bounded HTTPS transport and temporal buffering of derived features only."""

from collections import deque
from time import perf_counter, sleep
from urllib.parse import urlsplit

import httpx

from vigil.cloud.contracts import PredictRequest, PredictResponse
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS, WindowConfig
from vigil.ml.dataset.windowing import build_sample_feature_vector
from vigil.ml.inference import (
    InferenceResult,
    STATUS_OK,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_INVALID_INPUT,
)
from vigil.runtime.inference import ModelInferenceError


class CloudTransport:
    """Use TLS verification, finite timeouts, and at most two retries."""

    def __init__(self, endpoint, *, timeout=3.0, retries=1, client=None):
        parts = urlsplit(endpoint)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("cloud endpoint must be an HTTPS URL without credentials")
        import math

        if not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError("cloud timeout must be within (0, 30] seconds")
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError("cloud retries must be 0, 1 or 2")
        self.endpoint = endpoint.rstrip("/")
        self.retries = retries
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self.timeout = timeout
        self.bundle_id = None
        self.last_round_trip_ms = None

    def _request(self, method, path, **kwargs):
        for attempt in range(self.retries + 1):
            try:
                response = self.client.request(
                    method, self.endpoint + path, timeout=self.timeout, **kwargs
                )
                response.raise_for_status()
                if len(response.content) > 65536:
                    raise ValueError("response too large")
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in (429, 502, 503, 504)
                )
                if not retryable or attempt == self.retries:
                    raise ModelInferenceError("cloud inference unavailable") from None
                sleep(0.1 * (2**attempt))

    def connect(self):
        try:
            info = self._request("GET", "/model-info")
            if (
                tuple(info["feature_channels"]) != LOCKED_FEATURE_CHANNELS
                or info["window_length"] != 60
                or info["stride"] != 15
                or info["input_encoding"] != "raw_with_masks_v1"
            ):
                raise ValueError("cloud feature contract mismatch")
            import re

            if not re.fullmatch(r"[a-f0-9]{64}", info["bundle_id"]):
                raise ValueError("invalid bundle identifier")
            self.bundle_id = info["bundle_id"]
        except (KeyError, TypeError, ValueError):
            raise ModelInferenceError("cloud model contract mismatch") from None
        return self

    def predict(self, features):
        payload = PredictRequest(
            feature_channels=list(LOCKED_FEATURE_CHANNELS),
            features=features,
            bundle_id=self.bundle_id,
        )
        start = perf_counter()
        try:
            result = PredictResponse.model_validate(
                self._request("POST", "/predict", json=payload.model_dump())
            )
            if result.bundle_id != self.bundle_id:
                raise ValueError("cloud bundle changed")
            return result
        except ValueError:
            raise ModelInferenceError("cloud prediction contract mismatch") from None
        finally:
            self.last_round_trip_ms = (perf_counter() - start) * 1000

    def close(self):
        self.client.close()


class CloudInferenceEngine:
    """Mirror continuity, warm-up and stride; delegate normalization remotely.

    HTTP errors terminate the demo clearly. There is no implicit local fallback.
    No images, user identifiers, frame timestamps or labels cross the network.
    """

    def __init__(self, transport):
        self.transport = transport
        self.config = WindowConfig()
        self.reset()

    def reset(self):
        self._frames = deque(maxlen=60)
        self._prev = None
        self._since = 0
        self._predicted = False

    def process_sample(self, sample):
        def result(reason, status=STATUS_INSUFFICIENT_HISTORY, prediction=None):
            first = self._frames[0] if self._frames else sample
            return InferenceResult(
                predicted_class=prediction.predicted_class if prediction else None,
                class_id=prediction.class_id if prediction else None,
                probabilities=prediction.probabilities if prediction else None,
                confidence=prediction.confidence if prediction else None,
                latency_ms=prediction.latency_ms if prediction else 0.0,
                status=status,
                reason=reason,
                session_id=sample.session_id,
                buffer_size=len(self._frames),
                start_frame_index=first.frame_index,
                end_frame_index=sample.frame_index,
                start_timestamp_ms=first.timestamp_ms,
                end_timestamp_ms=sample.timestamp_ms,
            )

        try:
            sample.validate()
        except (ValueError, TypeError):
            self.reset()
            return result("invalid_input", STATUS_INVALID_INPUT)
        reason = "buffer_warming_up"
        if self._prev is not None:
            previous = self._prev
            if sample.session_id != previous.session_id:
                self.reset()
                reason = "session_reset"
            elif (
                sample.frame_index <= previous.frame_index
                or sample.timestamp_ms <= previous.timestamp_ms
            ):
                self.reset()
                return result("invalid_input", STATUS_INVALID_INPUT)
            elif sample.frame_index - previous.frame_index > self.config.max_frame_gap:
                self.reset()
                reason = "discontinuity_frame_gap"
            elif (
                sample.timestamp_ms - previous.timestamp_ms
                > self.config.max_time_gap_ms
            ):
                self.reset()
                reason = "discontinuity_timing_gap"
        self._frames.append(sample)
        self._prev = sample
        self._since += 1
        if len(self._frames) < 60:
            return result(reason)
        if self._predicted and self._since < 15:
            return result("stride_skip")
        rows = [build_sample_feature_vector(frame) for frame in self._frames]
        if not any(row[4] or row[5] or row[7] for row in rows):
            return result("invalid_input", STATUS_INVALID_INPUT)
        prediction = self.transport.predict(rows)
        self._predicted = True
        self._since = 0
        return result("predicted", STATUS_OK, prediction)
