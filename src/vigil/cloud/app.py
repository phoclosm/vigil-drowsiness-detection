"""ASGI service factory with safe errors and feature-only requests."""

import json
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from vigil.cloud.bundle import BundlePredictor
from vigil.cloud.assets import resolve_bundle
from vigil.cloud.contracts import PredictRequest, PredictResponse
from vigil.ml.dataset.config import LOCKED_FEATURE_CHANNELS

logger = logging.getLogger("vigil.cloud")


def create_app(bundle_path=None, *, predictor_factory=BundlePredictor):
    """Load a trusted local bundle during startup; expose readiness honestly."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        temporary_path = None
        try:
            path = bundle_path or resolve_bundle()
            if bundle_path is None and not os.environ.get("VIGIL_BUNDLE_PATH"):
                temporary_path = path
            app.state.predictor = predictor_factory(path)
        except Exception:
            logger.error("model_unavailable")
        try:
            yield
        finally:
            app.state.predictor = None
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    app = FastAPI(lifespan=lifespan, debug=False)

    @app.middleware("http")
    async def safe_boundary(request: Request, call_next):
        request_id = uuid4().hex
        # Bound even chunked bodies before Pydantic sees them.
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            if size > 131072:
                return JSONResponse({"detail": "payload too large"}, status_code=413)
            chunks.append(chunk)
        request._body = b"".join(chunks)
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                {"detail": "inference unavailable"}, status_code=503
            )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            json.dumps({"request_id": request_id, "status": response.status_code})
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic's default errors echo inputs; never return client data or paths.
        return JSONResponse(
            {"detail": "invalid temporal feature payload"}, status_code=422
        )

    def predictor():
        value = app.state.predictor
        if value is None:
            raise HTTPException(503, "model unavailable")
        return value

    @app.get("/health")
    def health():
        value = predictor()
        return {"status": "ready", "bundle_id": value.bundle_id}

    @app.get("/model-info")
    def model_info():
        value = predictor()
        return {
            "bundle_id": value.bundle_id,
            "feature_channels": LOCKED_FEATURE_CHANNELS,
            "window_length": 60,
            "stride": 15,
            "input_encoding": "raw_with_masks_v1",
            "max_frame_gap": value.bundle.window_config.max_frame_gap,
            "max_time_gap_ms": value.bundle.window_config.max_time_gap_ms,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest):
        value = predictor()
        if payload.bundle_id != value.bundle_id:
            raise HTTPException(409, "bundle identifier mismatch")
        try:
            return PredictResponse.model_validate(value.predict(payload))
        except Exception:
            raise HTTPException(503, "inference unavailable") from None

    return app
