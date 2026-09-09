"""Recommendation-specific serving state and readiness."""

from __future__ import annotations

from functools import lru_cache

from recsys_loom.live_recommender import LiveRecommendationStore
from services.recommender.runtime import (
    DEMO_PATH,
    REC_PATH,
    TRENDING_PATH,
    catalog,
    catalog_path,
    demo_recommendations,
)


@lru_cache(maxsize=1)
def live_recommendations() -> LiveRecommendationStore:
    return LiveRecommendationStore()


def recommendation_payload(customer_id: str) -> dict[str, object] | None:
    live = live_recommendations().recommend(customer_id, catalog())
    if live is not None:
        return live
    precomputed = demo_recommendations().get(customer_id)
    if not isinstance(precomputed, dict):
        return None
    return {**precomputed, "serving_mode": "precomputed_demo"}


def recommendation_readiness() -> tuple[dict[str, bool], dict[str, object]]:
    store = live_recommendations()
    catalog_error = ""
    catalog_count = 0
    if catalog_path().is_file():
        try:
            catalog_count = len(catalog())
        except (OSError, KeyError, UnicodeError, ValueError) as exc:
            catalog_error = f"{type(exc).__name__}: {exc}"
    checks = {
        "catalog": catalog_count > 0,
        "demo_customers": DEMO_PATH.is_file(),
        "recommendations": store.available or REC_PATH.is_file(),
        "trending": TRENDING_PATH.is_file(),
    }
    if store.configured:
        checks["live_bundle"] = store.available
    return checks, {
        "serving_mode": (
            "live_catboost" if store.available else "precomputed_demo"
        ),
        "live_bundle_errors": store.validation_errors,
        "catalog_articles": catalog_count,
        "catalog_error": catalog_error,
    }
