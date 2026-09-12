#!/usr/bin/env python3
"""Curate real demo personas from the serving-snapshot cache metadata.

Reads only customer_ids / history_buckets from the Sep 15 2K feature
cache (zip members). Does not load ranking features or train a model.
"""

from __future__ import annotations

import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.protocol import OVERNIGHT_DIR, ensure_directories
from recsys_loom.overnight.ranking import feature_cache_path

PERSONAS = [
    ("Demo Customer A", "sparse history", "0"),
    ("Demo Customer B", "medium history", "6-10"),
    ("Demo Customer C", "heavy history", "20+"),
    ("Demo Customer D", "repeat-oriented", "11-20"),
    ("Demo Customer E", "long-tail oriented", "3-5"),
]


def _load_ids_and_buckets(cache_path: Path) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(cache_path) as archive:
        customer_ids = np.load(
            BytesIO(archive.read("customer_ids.npy")),
            allow_pickle=True,
        )
        buckets = np.load(
            BytesIO(archive.read("history_buckets.npy")),
            allow_pickle=True,
        )
    return [str(value) for value in customer_ids], [str(value) for value in buckets]


def main() -> None:
    ensure_directories()
    cache_path = feature_cache_path("2020-09-15", 2000)
    if not cache_path.exists():
        raise SystemExit(f"missing serving cache: {cache_path}")
    customer_ids, buckets = _load_ids_and_buckets(cache_path)
    by_bucket: dict[str, list[str]] = {}
    for customer_id, bucket in zip(customer_ids, buckets):
        by_bucket.setdefault(bucket, []).append(customer_id)
    used: set[str] = set()
    chosen: list[dict[str, str]] = []
    for display_name, label, bucket in PERSONAS:
        candidates = [
            customer_id
            for customer_id in by_bucket.get(bucket, [])
            if customer_id not in used
        ]
        if not candidates:
            candidates = [
                customer_id
                for customer_id in customer_ids
                if customer_id not in used
            ]
        customer_id = candidates[0]
        used.add(customer_id)
        chosen.append(
            {
                "customer_id": customer_id,
                "display_name": display_name,
                "history_bucket": label,
                "summary": (
                    f"Real anonymized buyer in the {bucket} prior-purchase "
                    "bucket at the 2020-09-15 serving cutoff. Top-12 slates "
                    "are exported after BEST_SYSTEM is frozen."
                ),
            }
        )
    output_path = OVERNIGHT_DIR / "demo_customers.json"
    output_path.write_text(
        json.dumps({"customers": chosen}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"path": str(output_path), "customers": chosen}, indent=2))


if __name__ == "__main__":
    main()
