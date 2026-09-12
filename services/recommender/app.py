"""FastAPI service for personalized recommendations and hybrid search."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recsys_loom.search.llm import search_llm_health
from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.service import build_search_engine
from services.recommender.platform import configure_api, readiness_payload
from services.recommender.recommendation_runtime import (
    live_recommendations,
    recommendation_payload,
    recommendation_readiness,
)
from services.recommender.runtime import (
    demo_bundle,
    env_flag,
    images_path,
    lookup_article,
    search_readiness,
    trending_payload,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=256)
    customer_id: str | None = None
    limit: int = Field(default=20, ge=1, le=60)


class RecommendationResponse(BaseModel):
    customer_id: str
    model_version: str
    serving_mode: str
    candidate_count: int
    latency_ms: float
    recommendations: list[dict]


@lru_cache(maxsize=1)
def engine() -> SearchEngine:
    return build_search_engine(
        load_semantic=env_flag("SEARCH_SEMANTIC"),
        load_visual=env_flag("SEARCH_VISUAL"),
        load_llm=env_flag("SEARCH_LLM"),
    )


app = FastAPI(title="RecSys Loom", version="0.1.0")
configure_api(app, "combined-api")
if images_path().exists():
    app.mount("/images", StaticFiles(directory=images_path()), name="images")


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive", "service": "combined-api"}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    recommendation_checks, recommendation_details = recommendation_readiness()
    search_checks, search_details = search_readiness()
    if all(search_checks.values()) and env_flag("SEARCH_WARM_ON_READINESS", True):
        try:
            loaded_engine = engine()
            search_checks["engine"] = bool(loaded_engine.articles)
            search_details["indexed_articles"] = len(loaded_engine.articles)
        except Exception as exc:
            search_checks["engine"] = False
            search_details["engine_error"] = f"{type(exc).__name__}: {exc}"
    checks = {
        **{
            f"recommendation_{name}": passed
            for name, passed in recommendation_checks.items()
        },
        **{f"search_{name}": passed for name, passed in search_checks.items()},
    }
    payload, status_code = readiness_payload(
        "combined-api",
        checks,
        {
            "recommendation": recommendation_details,
            "search": search_details,
        },
    )
    return JSONResponse(payload, status_code=status_code)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "recommendation_serving": (
            "live_catboost"
            if live_recommendations().available
            else "precomputed_demo"
        ),
        "semantic_search": os.environ.get("SEARCH_SEMANTIC", "0"),
        "visual_search": os.environ.get("SEARCH_VISUAL", "0"),
        "llm_search": search_llm_health(),
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


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, object]:
    personalization = {}
    if request.customer_id:
        recs = recommendation_payload(request.customer_id) or {}
        for index, item in enumerate(recs.get("recommendations", []), start=1):
            personalization[item["article_id"]] = 1.0 / index
    return engine().search(
        request.query,
        limit=request.limit,
        personalization=personalization,
    )


@app.get("/api/articles/{article_id}")
def article(article_id: str) -> dict[str, str]:
    found = lookup_article(article_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown article")
    return found.to_card()
