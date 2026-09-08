#!/usr/bin/env python3
"""Score curated demo customers from the last serving snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.protocol import (
    CONTAMINATED_FINAL_FOLD,
    DEV_CUSTOMERS,
    OVERNIGHT_DIR,
    ensure_directories,
)
from recsys_loom.overnight.ranking import (
    apply_baseline_budget,
    development_specs,
    fit_ranker,
    load_or_build_snapshot,
)
from recsys_loom.search.catalog import load_articles

PERSONAS = [
    ("Demo Customer A", "sparse history", "0"),
    ("Demo Customer B", "medium history", "6-10"),
    ("Demo Customer C", "heavy history", "20+"),
    ("Demo Customer D", "repeat-oriented", "11-20"),
    ("Demo Customer E", "long-tail oriented", "3-5"),
]


def _top_sources(row: np.ndarray, names: list[str]) -> list[str]:
    sources = []
    for source in (
        "recent_7d_pop",
        "repeat_purchase",
        "cooccurrence",
        "als",
        "content",
        "two_tower",
    ):
        column = names.index(f"is_{source}")
        if row[column] == 1.0:
            sources.append(source)
    return sources


def main() -> None:
    ensure_directories()
    connection = connect()
    article_ids = catalog_article_ids(connection)
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    catalog = load_articles()
    snapshots = []
    for specification in development_specs(DEV_CUSTOMERS):
        data, _relevance = load_or_build_snapshot(
            connection, specification, article_to_index
        )
        snapshots.append(apply_baseline_budget(data))
    model = fit_ranker(snapshots[:CONTAMINATED_FINAL_FOLD])
    serving = snapshots[CONTAMINATED_FINAL_FOLD]
    scores = np.asarray(model.predict(serving["features"]), dtype=np.float32)
    names = [str(value) for value in serving["feature_names"]]
    chosen: list[dict[str, object]] = []
    used = set()
    offset = 0
    groups = []
    for customer_index, group_size_value in enumerate(serving["groups"]):
        group_size = int(group_size_value)
        end = offset + group_size
        groups.append((customer_index, offset, end))
        offset = end
    by_bucket: dict[str, list[int]] = {}
    for customer_index, _start, _end in groups:
        bucket = str(serving["history_buckets"][customer_index])
        by_bucket.setdefault(bucket, []).append(customer_index)
    for display_name, label, bucket in PERSONAS:
        candidates = [
            index
            for index in by_bucket.get(bucket, [])
            if index not in used
        ]
        if not candidates:
            candidates = [
                index
                for index, _start, _end in groups
                if index not in used
            ]
        customer_index = candidates[0]
        used.add(customer_index)
        start, end = groups[customer_index][1], groups[customer_index][2]
        take = min(12, end - start)
        local = np.argpartition(scores[start:end], -take)[-take:]
        local = local[np.argsort(-scores[start:end][local])]
        recs = []
        for local_index in local:
            row = serving["features"][start + int(local_index)]
            article_id = article_ids[
                int(serving["pair_article_indices"][start + int(local_index)])
            ]
            card = catalog[article_id].to_card()
            card["score"] = float(scores[start + int(local_index)])
            card["sources"] = _top_sources(row, names)
            recs.append(card)
        customer_id = str(serving["customer_ids"][customer_index])
        chosen.append(
            {
                "customer_id": customer_id,
                "display_name": display_name,
                "history_bucket": label,
                "summary": (
                    f"Real anonymized buyer in the {bucket} history bucket "
                    "at the 2020-09-15 serving cutoff."
                ),
            }
        )
        payload = {
            "customer_id": customer_id,
            "model_version": "baseline_lambdarank_tt50",
            "candidate_count": int(end - start),
            "latency_ms": 0.0,
            "recommendations": recs,
        }
        store_path = OVERNIGHT_DIR / "demo_recommendations.json"
        store = (
            json.loads(store_path.read_text(encoding="utf-8"))
            if store_path.exists()
            else {}
        )
        store[customer_id] = payload
        store_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    (OVERNIGHT_DIR / "demo_customers.json").write_text(
        json.dumps({"customers": chosen}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"customers": chosen}, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
