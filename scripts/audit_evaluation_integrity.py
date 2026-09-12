#!/usr/bin/env python3
"""Build the overnight evaluation ledger from caches and the transaction file."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.db import connect
from recsys_loom.overnight.protocol import (
    OVERNIGHT_DIR,
    PROTOCOL,
    SNAPSHOTS,
    ensure_directories,
)
from recsys_loom.two_tower.data import load_target_customers

WEEKLY_WINDOWS = [
    {
        "name": snapshot["cutoff"],
        "target_start": snapshot["target_start"],
        "target_end": snapshot["target_end"],
    }
    for snapshot in SNAPSHOTS
]
FINAL_START = "2020-09-16"
FINAL_END = "2020-09-22"


def unique_customers(path: Path) -> list[str]:
    if not path.exists():
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            customer_id = row["customer_id"]
            if customer_id not in seen:
                seen.add(customer_id)
                ordered.append(customer_id)
    return ordered


def first_n_customers(
    connection,
    target_start: str,
    target_end: str,
    count: int,
) -> list[str]:
    rows = connection.sql(f"""
        SELECT DISTINCT customer_id
        FROM transactions
        WHERE transaction_date BETWEEN DATE '{target_start}' AND DATE '{target_end}'
        ORDER BY customer_id
        LIMIT {int(count)}
    """).fetchall()
    return [row[0] for row in rows]


def collect_cache_customers() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    cache_paths = [
        ROOT / "artifacts" / "ranking" / "candidate_cache" / "existing_2020-08-17_2000.tsv.gz",
        ROOT / "artifacts" / "ranking" / "candidate_cache" / "existing_2020-08-24_2000.tsv.gz",
        ROOT / "artifacts" / "ranking" / "candidate_cache" / "existing_2020-08-31_2000.tsv.gz",
        ROOT / "artifacts" / "ranking" / "candidate_cache" / "existing_2020-09-07_2000.tsv.gz",
        ROOT / "artifacts" / "ranking" / "candidate_cache" / "existing_2020-09-15_2000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "existing_candidates_1000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "existing_candidates_5000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "two_tower_candidates_5000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "ranking_snapshots" / "candidates_2020-08-17_2000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "ranking_snapshots" / "candidates_2020-08-24_2000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "ranking_snapshots" / "candidates_2020-08-31_2000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "ranking_snapshots" / "candidates_2020-09-07_2000.tsv.gz",
        ROOT / "artifacts" / "two_tower" / "ranking_snapshots" / "candidates_2020-09-15_2000.tsv.gz",
    ]
    for path in cache_paths:
        mapping[str(path.relative_to(ROOT))] = unique_customers(path)
    return mapping


def main() -> None:
    ensure_directories()
    connection = connect()
    week_counts = []
    for window in WEEKLY_WINDOWS:
        row = connection.sql(f"""
            SELECT
                COUNT(DISTINCT customer_id) AS customers,
                COUNT(*) AS transactions,
                COUNT(DISTINCT article_id) AS articles
            FROM transactions
            WHERE transaction_date BETWEEN DATE '{window["target_start"]}'
              AND DATE '{window["target_end"]}'
        """).fetchone()
        week_counts.append(
            {
                "cutoff_name": window["name"],
                "target_start": window["target_start"],
                "target_end": window["target_end"],
                "customers": int(row[0]),
                "transactions": int(row[1]),
                "articles": int(row[2]),
            }
        )

    cache_customers = collect_cache_customers()
    cache_sizes = {
        path: len(customer_ids)
        for path, customer_ids in cache_customers.items()
    }

    reconstructed = {
        "first_500": first_n_customers(connection, FINAL_START, FINAL_END, 500),
        "first_1000": first_n_customers(connection, FINAL_START, FINAL_END, 1000),
        "first_2000": first_n_customers(connection, FINAL_START, FINAL_END, 2000),
        "first_5000": first_n_customers(connection, FINAL_START, FINAL_END, 5000),
        "random_100_seed42": load_target_customers(
            connection, FINAL_START, FINAL_END, 100, 42
        ),
        "random_1000_seed42": load_target_customers(
            connection, FINAL_START, FINAL_END, 1000, 42
        ),
        "random_5000_seed42": load_target_customers(
            connection, FINAL_START, FINAL_END, 5000, 42
        ),
    }

    contaminated: set[str] = set()
    final_cache_keys = [
        "artifacts/ranking/candidate_cache/existing_2020-09-15_2000.tsv.gz",
        "artifacts/two_tower/existing_candidates_1000.tsv.gz",
        "artifacts/two_tower/existing_candidates_5000.tsv.gz",
        "artifacts/two_tower/two_tower_candidates_5000.tsv.gz",
        "artifacts/two_tower/ranking_snapshots/candidates_2020-09-15_2000.tsv.gz",
    ]
    for key in final_cache_keys:
        contaminated.update(cache_customers.get(key, []))
    for customer_ids in reconstructed.values():
        contaminated.update(customer_ids)

    all_final = [
        row[0]
        for row in connection.sql(f"""
            SELECT DISTINCT customer_id
            FROM transactions
            WHERE transaction_date BETWEEN DATE '{FINAL_START}' AND DATE '{FINAL_END}'
            ORDER BY customer_id
        """).fetchall()
    ]
    unused = [customer_id for customer_id in all_final if customer_id not in contaminated]
    rng = np.random.default_rng(20260908)
    holdout_size = min(8000, len(unused))
    holdout = sorted(
        unused[int(index)]
        for index in rng.choice(len(unused), size=holdout_size, replace=False)
    ) if unused else []

    holdout_path = OVERNIGHT_DIR / "holdout_customer_ids.txt"
    holdout_path.write_text("\n".join(holdout) + ("\n" if holdout else ""), encoding="utf-8")
    contaminated_path = OVERNIGHT_DIR / "contaminated_final_week_customer_ids.txt"
    contaminated_path.write_text(
        "\n".join(sorted(contaminated)) + "\n",
        encoding="utf-8",
    )

    ledger = {
        "protocol": PROTOCOL,
        "weekly_populations": week_counts,
        "cache_customer_counts": cache_sizes,
        "final_week": {
            "target_start": FINAL_START,
            "target_end": FINAL_END,
            "active_customers": len(all_final),
            "contaminated_customers": len(contaminated),
            "unused_customers": len(unused),
            "reserved_holdout_customers": len(holdout),
            "holdout_path": str(holdout_path.relative_to(ROOT)),
            "contaminated_path": str(contaminated_path.relative_to(ROOT)),
        },
        "previously_used_windows": {
            "training_cutoffs": [
                "2020-08-17",
                "2020-08-24",
                "2020-08-31",
                "2020-09-07",
            ],
            "selection_cutoffs": ["2020-08-31", "2020-09-07"],
            "inspected_final_cutoffs": ["2020-09-15"],
            "customer_subset_sizes": [100, 500, 1000, 2000, 5000],
            "sampling_methods": [
                "ORDER BY customer_id LIMIT N for ranking caches",
                "numpy seed-42 random sample then sort for two-tower 5K",
            ],
        },
        "decision": {
            "untouched_temporal_window": None,
            "final_evaluation_type": "customer_holdout",
            "development_metric": (
                "mean MAP@12 on 2020-08-31 and 2020-09-07, 2K customers"
            ),
            "do_not_use_for_selection": (
                "any 2020-09-16..22 metric, including the historical 5K MAP@12"
            ),
        },
    }
    output_path = OVERNIGHT_DIR / "evaluation_ledger.json"
    output_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(json.dumps(ledger["final_week"], indent=2))
    print(f"Wrote {output_path}")
    connection.close()


if __name__ == "__main__":
    main()
