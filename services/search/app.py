"""Search-only FastAPI entrypoint with optional recommendation personalization."""

from __future__ import annotations

from functools import lru_cache
import logging
import os
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field

from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.service import build_search_engine
from services.recommender.platform import configure_api, readiness_payload
from services.recommender.runtime import (
    env_flag,
    lookup_article,
    search_readiness,
)

LOGGER = logging.getLogger("recsys_loom.search_api")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=256)
    customer_id: str | None = None
    limit: int = Field(default=20, ge=1, le=60)


@lru_cache(maxsize=1)
def engine() -> SearchEngine:
    return build_search_engine(
        load_semantic=env_flag("SEARCH_SEMANTIC"),
        load_visual=env_flag("SEARCH_VISUAL"),
        load_llm=env_flag("SEARCH_LLM"),
    )


@lru_cache(maxsize=1)
def recommendation_client() -> httpx.Client:
    try:
        timeout = float(os.environ.get("RECOMMENDATION_TIMEOUT_SECONDS", "0.35"))
    except ValueError:
        timeout = 0.35
    return httpx.Client(timeout=max(0.05, min(timeout, 5.0)))


def personalization_for(customer_id: str | None) -> dict[str, float]:
    base_url = os.environ.get("RECOMMENDATION_SERVICE_URL", "").rstrip("/")
    if not customer_id or not base_url:
        return {}
    try:
        response = recommendation_client().get(
            f"{base_url}/api/recommendations/{quote(customer_id, safe='')}",
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {}
        recommendations = payload.get("recommendations", [])
        if not isinstance(recommendations, list):
            return {}
        return {
            str(item["article_id"]): 1.0 / index
            for index, item in enumerate(recommendations, start=1)
            if isinstance(item, dict) and item.get("article_id")
        }
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        LOGGER.warning(
            "recommendation_personalization_fallback error=%s",
            type(exc).__name__,
        )
        return {}


app = FastAPI(title="RecSys Loom Search API", version="0.1.0")
configure_api(app, "search-api")


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive", "service": "search-api"}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    checks, details = search_readiness()
    if all(checks.values()) and env_flag("SEARCH_WARM_ON_READINESS", True):
        try:
            loaded_engine = engine()
            checks["engine"] = bool(loaded_engine.articles)
            details["indexed_articles"] = len(loaded_engine.articles)
        except Exception as exc:
            checks["engine"] = False
            details["engine_error"] = f"{type(exc).__name__}: {exc}"
    payload, status_code = readiness_payload("search-api", checks, details)
    return JSONResponse(payload, status_code=status_code)


@app.get("/api/search/health")
def health() -> dict[str, object]:
    _, details = search_readiness()
    return {
        "status": "ok",
        "service": "search-api",
        "image_digest": os.environ.get("IMAGE_DIGEST", "local"),
        "model_version": os.environ.get("MODEL_VERSION", "unknown"),
        **details,
    }


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, object]:
    return engine().search(
        request.query,
        limit=request.limit,
        personalization=personalization_for(request.customer_id),
    )


@app.get("/api/articles/{article_id}")
def article(article_id: str) -> dict[str, str]:
    found = lookup_article(article_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown article")
    return found.to_card()
