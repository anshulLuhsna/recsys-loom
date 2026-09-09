"""Recommendation-only FastAPI entrypoint."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from services.recommender.platform import configure_api, readiness_payload
from services.recommender.recommendation_runtime import (
    live_recommendations,
    recommendation_payload,
    recommendation_readiness,
)
from services.recommender.runtime import (
    demo_bundle,
    trending_payload,
)

app = FastAPI(title="RecSys Loom Recommendation API", version="0.1.0")
configure_api(app, "recommendation-api")


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive", "service": "recommendation-api"}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    checks, details = recommendation_readiness()
    payload, status_code = readiness_payload(
        "recommendation-api",
        checks,
        details,
    )
    return JSONResponse(payload, status_code=status_code)


@app.get("/api/health")
def health() -> dict[str, object]:
    store = live_recommendations()
    return {
        "status": "ok",
        "service": "recommendation-api",
        "recommendation_serving": (
            "live_catboost" if store.available else "precomputed_demo"
        ),
        "image_digest": os.environ.get("IMAGE_DIGEST", "local"),
        "model_version": os.environ.get("MODEL_VERSION", "unknown"),
    }


@app.get("/api/demo-customers")
def demo_customers() -> dict[str, object]:
    return demo_bundle()


@app.get("/api/trending")
def trending() -> dict[str, object]:
    return trending_payload()


@app.get("/api/recommendations/{customer_id}")
def recommendations(customer_id: str) -> dict[str, object]:
    payload = recommendation_payload(customer_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="Customer is outside the exported live/demo serving set",
        )
    return payload
