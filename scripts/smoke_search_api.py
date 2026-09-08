#!/usr/bin/env python3
"""Smoke-test the search API without loading MiniLM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["SEARCH_SEMANTIC"] = "0"

from fastapi.testclient import TestClient

from services.recommender.app import app, engine


def main() -> None:
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200, health.text
    trending = client.get("/api/trending")
    assert trending.status_code == 200, trending.text
    assert trending.json()["items"], "expected recent-popularity cards"
    customers = client.get("/api/demo-customers")
    assert customers.status_code == 200, customers.text
    assert customers.json()["customers"], "expected curated demo personas"
    engine.cache_clear()
    response = client.post(
        "/api/search",
        json={"query": "black trousers", "limit": 8},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["parsed_intent"]["product_type"] == "Trousers"
    assert payload["parsed_intent"]["color"] == "Black"
    assert payload["results"], "expected search hits"
    top = payload["results"][0]
    print(
        {
            "status": "ok",
            "intent": payload["parsed_intent"]["hard_constraints"],
            "top": {
                "name": top["name"],
                "type": top["product_type"],
                "color": top["color"],
            },
            "latency_ms": payload["latency_ms"]["total"],
            "result_count": len(payload["results"]),
        }
    )


if __name__ == "__main__":
    main()
