#!/usr/bin/env python3
"""Phase 1: Evaluate all non-neural retrieval baselines on identical folds.

Runs each candidate source at K=100, 300, 500 and produces:
- standalone recall per source
- marginal recall (unique recoveries)
- pairwise overlap
- eligibility ceiling
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.candidates import CandidateRecord
from recsys_loom.eligibility import (
    EligibilityPolicy,
    eligibility_ceiling,
    eligible_articles,
)
from recsys_loom.retrieval_eval import incremental_union_recall, retrieval_report
from recsys_loom.sources.als import als_candidates
from recsys_loom.sources.cooccurrence import cooccurrence_candidates
from recsys_loom.sources.content import content_candidates
from recsys_loom.sources.popularity import (
    all_history_popularity,
    global_list_to_article_ids,
    recent_popularity,
    time_decayed_popularity,
)
from recsys_loom.sources.repeat_purchase import repeat_purchase_candidates

TRAIN_END = "2020-09-15"
VALIDATION_START = "2020-09-16"
VALIDATION_END = "2020-09-22"
K_VALUES = (100, 300, 500)
MAX_CANDIDATES = 500
OUTPUT_DIR = ROOT / "artifacts" / "retrieval_baselines"


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '6GB'")

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id
        FROM read_csv(
            '{ROOT / "transactions_train.csv"}',
            header = true,
            all_varchar = true
        )
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT *
        FROM read_csv(
            '{ROOT / "articles.csv"}',
            header = true,
            all_varchar = true
        )
    """)


def load_validation(con: duckdb.DuckDBPyConnection) -> dict[str, set[str]]:
    rows = con.sql(f"""
        SELECT customer_id, LIST(DISTINCT article_id) AS articles
        FROM transactions
        WHERE transaction_date BETWEEN
            DATE '{VALIDATION_START}' AND DATE '{VALIDATION_END}'
        GROUP BY customer_id
    """).fetchall()
    return {cid: set(articles) for cid, articles in rows}


def load_observed_articles(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.sql(f"""
        SELECT DISTINCT article_id
        FROM transactions
        WHERE transaction_date <= DATE '{TRAIN_END}'
    """).fetchall()
    return {r[0] for r in rows}


def load_full_catalog(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.sql("SELECT DISTINCT article_id FROM articles").fetchall()
    return {r[0] for r in rows}


def records_to_dict(records: list[CandidateRecord]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for r in records:
        result.setdefault(r.customer_id, []).append(r.article_id)
    return result


def broadcast_global(
    article_ids: list[str],
    customer_ids: Sequence[str],
) -> dict[str, list[str]]:
    """Broadcast a global candidate list to all customers without copying."""
    return {cid: article_ids for cid in customer_ids}


def timed(name: str):
    """Context manager that prints elapsed time."""
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self.start = time.monotonic()
            print(f"  {name}...", end="", flush=True)
            return self
        def __exit__(self, *_):
            self.elapsed = time.monotonic() - self.start
            print(f" {self.elapsed:.1f}s", flush=True)
    return Timer()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.monotonic()

    con = duckdb.connect()
    setup_connection(con)

    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("Loading validation labels...", flush=True)
    relevant_by_customer = load_validation(con)
    customer_ids = sorted(relevant_by_customer)
    if sample_size > 0:
        customer_ids = customer_ids[:sample_size]
        relevant_by_customer = {c: relevant_by_customer[c] for c in customer_ids}
        print(f"  SAMPLED {len(customer_ids):,} of {len(load_validation(con)):,} customers")
    else:
        print(f"  {len(customer_ids):,} validation customers", flush=True)

    print("Loading catalog and eligibility...", flush=True)
    full_catalog = load_full_catalog(con)
    observed = load_observed_articles(con)
    ceilings: dict[str, Any] = {}
    for policy in EligibilityPolicy:
        elig = eligible_articles(policy, full_catalog, observed_before_cutoff=observed)
        ceiling = eligibility_ceiling(relevant_by_customer, elig)
        ceilings[policy.value] = ceiling
        print(f"  {policy.value}: ceiling={ceiling['ceiling']:.4f} "
              f"({ceiling['surviving_relevant_pairs']}/{ceiling['total_relevant_pairs']})")

    candidates_by_source: dict[str, dict[str, list[str]]] = {}
    timings: dict[str, float] = {}

    print("\n=== Global popularity sources ===\n", flush=True)

    with timed("all_history_popularity") as t:
        ids = global_list_to_article_ids(all_history_popularity(con, TRAIN_END, MAX_CANDIDATES))
        candidates_by_source["all_history_pop"] = broadcast_global(ids, customer_ids)
    timings["all_history_pop"] = t.elapsed

    with timed("recent_7d_popularity") as t:
        ids = global_list_to_article_ids(recent_popularity(con, TRAIN_END, 7, MAX_CANDIDATES))
        candidates_by_source["recent_7d_pop"] = broadcast_global(ids, customer_ids)
    timings["recent_7d_pop"] = t.elapsed

    with timed("recent_30d_popularity") as t:
        ids = global_list_to_article_ids(recent_popularity(con, TRAIN_END, 30, MAX_CANDIDATES))
        candidates_by_source["recent_30d_pop"] = broadcast_global(ids, customer_ids)
    timings["recent_30d_pop"] = t.elapsed

    with timed("decayed_14d_popularity") as t:
        ids = global_list_to_article_ids(time_decayed_popularity(con, TRAIN_END, 14.0, MAX_CANDIDATES))
        candidates_by_source["decayed_14d_pop"] = broadcast_global(ids, customer_ids)
    timings["decayed_14d_pop"] = t.elapsed

    print("\n=== Personalized sources ===\n", flush=True)

    with timed("repeat_purchase") as t:
        recs = repeat_purchase_candidates(con, customer_ids, TRAIN_END, 30.0, MAX_CANDIDATES)
        candidates_by_source["repeat_purchase"] = records_to_dict(recs)
        print(f" ({len(recs):,} records)", end="")
    timings["repeat_purchase"] = t.elapsed

    with timed("cooccurrence (90d window, min_co=5)") as t:
        recs = cooccurrence_candidates(con, customer_ids, TRAIN_END, 90, 50, MAX_CANDIDATES, 5)
        candidates_by_source["cooccurrence"] = records_to_dict(recs)
        print(f" ({len(recs):,} records)", end="")
    timings["cooccurrence"] = t.elapsed

    with timed("content (90d history)") as t:
        recs = content_candidates(con, customer_ids, TRAIN_END, 90, MAX_CANDIDATES)
        candidates_by_source["content"] = records_to_dict(recs)
        print(f" ({len(recs):,} records)", end="")
    timings["content"] = t.elapsed

    with timed("ALS (64 factors, 15 iterations)") as t:
        recs = als_candidates(con, customer_ids, TRAIN_END, k=MAX_CANDIDATES)
        candidates_by_source["als"] = records_to_dict(recs)
        print(f" ({len(recs):,} records)", end="")
    timings["als"] = t.elapsed

    print("\n\n=== Evaluating retrieval ===\n", flush=True)
    catalog_size = len(full_catalog)
    report = retrieval_report(
        relevant_by_customer, candidates_by_source, catalog_size, K_VALUES,
    )

    kept_sources = ["recent_7d_pop", "repeat_purchase", "cooccurrence", "als", "content"]
    kept_candidates = {s: candidates_by_source[s] for s in kept_sources if s in candidates_by_source}

    for k_cap in K_VALUES:
        inc = incremental_union_recall(
            relevant_by_customer, kept_candidates, kept_sources, k_cap,
        )
        report[f"union_at_{k_cap}"] = inc

    report["timings"] = timings
    report["ceilings"] = ceilings
    report["contract"] = {
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "k_values": list(K_VALUES),
        "max_candidates": MAX_CANDIDATES,
        "validation_customers": len(customer_ids),
        "catalog_size": catalog_size,
    }

    (OUTPUT_DIR / "retrieval_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )

    print_summary(report)

    total_elapsed = time.monotonic() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s")
    print(f"Report: {OUTPUT_DIR / 'retrieval_report.json'}")
    con.close()


def print_summary(report: dict[str, Any]) -> None:
    print("=== Standalone Recall@K ===\n")
    header = f"{'Source':<22}"
    for k in K_VALUES:
        header += f"  R@{k:<6}"
    header += "  Coverage"
    print(header)
    print("-" * len(header))

    for source_name, metrics in report["standalone"].items():
        line = f"{source_name:<22}"
        for k in K_VALUES:
            r = metrics.get(f"recall_at_{k}", 0)
            line += f"  {r:<.4f} "
        cov = metrics.get("catalog_coverage", 0)
        line += f"  {cov:.4f}"
        print(line)

    for k in K_VALUES:
        if f"at_{k}" in report.get("marginal", {}):
            print(f"\n=== Marginal Recall @ {k} ===\n")
            marg = report["marginal"][f"at_{k}"]
            print(f"{'Source':<22}  {'Hits':>8}  {'Unique':>8}  {'Frac':>8}")
            print("-" * 52)
            for source_name, m in marg.items():
                print(
                    f"{source_name:<22}  {m['total_hits']:>8}  "
                    f"{m['unique_hits']:>8}  {m['unique_fraction']:>8.4f}"
                )

    for k in K_VALUES:
        key = f"union_at_{k}"
        if key in report:
            print(f"\n=== Incremental Union Recall @ {k} per source ===\n")
            print(f"{'Source added':<22}  {'Union R':>10}  {'Avg pool':>10}  {'Hits':>8}")
            print("-" * 56)
            for step in report[key]:
                print(
                    f"+ {step['source_added']:<20}  {step['union_recall']:>10.4f}  "
                    f"{step['avg_pool_size']:>10.1f}  {step['union_hits']:>8}"
                )


if __name__ == "__main__":
    main()
