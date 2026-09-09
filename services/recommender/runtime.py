"""Shared serving data access without importing offline training code."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from recsys_loom.overnight.protocol import ARTIFACTS, OVERNIGHT_DIR, ROOT
from recsys_loom.search.catalog import Article, load_articles
from recsys_loom.search.llm import SearchLLMConfig

DEMO_PATH = OVERNIGHT_DIR / "demo_customers.json"
REC_PATH = OVERNIGHT_DIR / "demo_recommendations.json"
TRENDING_PATH = ARTIFACTS / "recent_popularity_7d" / "metrics.json"
TEXT_DIR = ARTIFACTS / "text_retrieval"
VISUAL_DIR = ARTIFACTS / "visual_search"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def catalog_path() -> Path:
    return Path(os.environ.get("CATALOG_PATH", ROOT / "articles.csv"))


def images_path() -> Path:
    return Path(os.environ.get("IMAGES_PATH", ROOT / "images"))


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
    return load_articles(catalog_path())


def lookup_article(article_id: str) -> Article | None:
    articles = catalog()
    if article_id in articles:
        return articles[article_id]
    padded = article_id.zfill(10)
    if padded in articles:
        return articles[padded]
    stripped = article_id.lstrip("0") or "0"
    return next(
        (
            article
            for key, article in articles.items()
            if key.lstrip("0") == stripped
        ),
        None,
    )


def trending_payload() -> dict[str, object]:
    if not TRENDING_PATH.exists():
        return {"items": [], "source": "missing"}
    metrics = json.loads(TRENDING_PATH.read_text(encoding="utf-8"))
    items = []
    for row in metrics.get("top_articles", [])[:12]:
        article = lookup_article(str(row["article_id"]))
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


def search_readiness() -> tuple[dict[str, bool], dict[str, object]]:
    semantic_enabled = env_flag("SEARCH_SEMANTIC")
    visual_enabled = env_flag("SEARCH_VISUAL")
    llm_config = SearchLLMConfig.from_env()
    checks = {
        "catalog": catalog_path().is_file(),
        "semantic_artifacts": (
            not semantic_enabled
            or (
                (TEXT_DIR / "text_embeddings.f32.npy").is_file()
                and (TEXT_DIR / "text_article_ids.npy").is_file()
            )
        ),
        "visual_artifacts": (
            not visual_enabled
            or (
                (VISUAL_DIR / "visual_embedding_metadata.json").is_file()
                and (VISUAL_DIR / "visual_embeddings.f32.npy").is_file()
                and (VISUAL_DIR / "visual_article_ids.npy").is_file()
                and (VISUAL_DIR / "visual_embedding_status.u8.npy").is_file()
            )
        ),
        "llm_configuration": not llm_config.enabled or llm_config.configured,
    }
    return checks, {
        "semantic_search": semantic_enabled,
        "visual_search": visual_enabled,
        "llm_search": llm_config.diagnostics(),
    }
