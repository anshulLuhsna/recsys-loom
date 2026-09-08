"""FastAPI service for personalized recommendations and hybrid search."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recsys_loom.overnight.protocol import OVERNIGHT_DIR, ROOT
from recsys_loom.search.catalog import Article, load_articles
from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.service import build_search_engine

DEMO_PATH = OVERNIGHT_DIR / "demo_customers.json"
REC_PATH = OVERNIGHT_DIR / "demo_recommendations.json"
TRENDING_PATH = ROOT / "artifacts" / "recent_popularity_7d" / "metrics.json"
IMAGES = ROOT / "images"


class SearchRequest(BaseModel):
    query: str
    customer_id: str | None = None
    limit: int = Field(default=20, ge=1, le=60)


class RecommendationResponse(BaseModel):
    customer_id: str
    model_version: str
    candidate_count: int
    latency_ms: float
    recommendations: list[dict]


@lru_cache(maxsize=1)
def engine() -> SearchEngine:
    return build_search_engine(load_semantic=os.environ.get("SEARCH_SEMANTIC", "1") != "0")


@lru_cache(maxsize=1)
def demo_bundle() -> dict[str, object]:
    if DEMO_PATH.exists():
        return json.loads(DEMO_PATH.read_text(encoding="utf-8"))
    return {"customers": []}


@lru_cache(maxsize=1)
def demo_recommendations() -> dict[str, object]:
    if REC_PATH.exists():
        return json.loads(REC_PATH.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=1)
def catalog() -> dict[str, Article]:
    return load_articles()


def _lookup_article(article_id: str) -> Article | None:
    articles = catalog()
    if article_id in articles:
        return articles[article_id]
    padded = article_id.zfill(10)
    if padded in articles:
        return articles[padded]
    stripped = article_id.lstrip("0") or "0"
    for key, article in articles.items():
        if key.lstrip("0") == stripped:
            return article
    return None


app = FastAPI(title="RecSys Loom", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if IMAGES.exists():
    app.mount("/images", StaticFiles(directory=IMAGES), name="images")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo-customers")
def demo_customers() -> dict[str, object]:
    return demo_bundle()


@app.get("/api/trending")
def trending() -> dict[str, object]:
    if not TRENDING_PATH.exists():
        return {"items": [], "source": "missing"}
    metrics = json.loads(TRENDING_PATH.read_text(encoding="utf-8"))
    items = []
    for row in metrics.get("top_articles", [])[:12]:
        article = _lookup_article(str(row["article_id"]))
        if article is None:
            continue
        card = article.to_card()
        card["score"] = float(row.get("purchase_count", 0))
        card["sources"] = ["recent_7d_pop"]
        items.append(card)
    return {
        "source": "recent_7d_popularity",
        "cutoff": "2020-09-15",
        "items": items,
    }


@app.get("/api/recommendations/{customer_id}")
def recommendations(customer_id: str) -> dict[str, object]:
    store = demo_recommendations()
    payload = store.get(customer_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="No precomputed recommendations for this customer yet",
        )
    return payload


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, object]:
    personalization = {}
    if request.customer_id:
        recs = demo_recommendations().get(request.customer_id, {})
        for index, item in enumerate(recs.get("recommendations", []), start=1):
            personalization[item["article_id"]] = 1.0 / index
    return engine().search(
        request.query,
        limit=request.limit,
        personalization=personalization,
    )


@app.get("/api/articles/{article_id}")
def article(article_id: str) -> dict[str, str]:
    found = _lookup_article(article_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown article")
    return found.to_card()
