"""FastAPI service for personalized recommendations and hybrid search."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recsys_loom.overnight.protocol import OVERNIGHT_DIR, ROOT
from recsys_loom.search.catalog import load_articles
from recsys_loom.search.lexical import BM25Index
from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.semantic import SemanticIndex
from recsys_loom.search.structured import StructuredIndex

DEMO_PATH = OVERNIGHT_DIR / "demo_customers.json"
REC_PATH = OVERNIGHT_DIR / "demo_recommendations.json"
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
    articles = load_articles()
    lexical = BM25Index()
    lexical.build(articles)
    return SearchEngine(
        articles,
        lexical,
        SemanticIndex.load(load_encoder=True),
        StructuredIndex(articles),
    )


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
    article = engine().articles.get(article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Unknown article")
    return article.to_card()
