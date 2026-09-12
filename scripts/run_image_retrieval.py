#!/usr/bin/env python3
"""Build temporal image candidates and measure marginal retrieval value."""

from __future__ import annotations

import csv
import gc
import gzip
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.ranking.cached_data import (
    BASE_SOURCES,
    SnapshotSpec,
    load_candidate_pool,
    load_relevance,
)
from recsys_loom.sources.image import (
    ImageEmbeddingStore,
    build_visual_profiles,
    eligible_image_item_indices,
    filter_history_from_ann,
)

SNAPSHOTS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
    {"cutoff": "2020-09-15", "target_start": "2020-09-16", "target_end": "2020-09-22"},
]
K_VALUES = [50, 100, 300]
MAX_IMAGE_K = 300
TT_BASELINE_K = 50
OUTPUT_DIR = ROOT / "artifacts" / "image_retrieval"


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id
        FROM read_csv('{ROOT / "transactions_train.csv"}', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT * FROM read_csv('{ROOT / "articles.csv"}', header=true, all_varchar=true)
    """)


def make_specs(
    training_customers: int,
    validation_customers: int,
) -> list[SnapshotSpec]:
    specifications: list[SnapshotSpec] = []
    for index, snapshot in enumerate(SNAPSHOTS):
        sample_size = validation_customers if index == 4 else training_customers
        if index == 4:
            existing = (
                ROOT
                / "artifacts"
                / "two_tower"
                / f"existing_candidates_{sample_size}.tsv.gz"
            )
            two_tower = (
                ROOT
                / "artifacts"
                / "two_tower"
                / f"two_tower_candidates_{sample_size}.tsv.gz"
            )
        else:
            existing = (
                ROOT
                / "artifacts"
                / "ranking"
                / "candidate_cache"
                / f"existing_{snapshot['cutoff']}_{sample_size}.tsv.gz"
            )
            two_tower = (
                ROOT
                / "artifacts"
                / "two_tower"
                / "ranking_snapshots"
                / f"candidates_{snapshot['cutoff']}_{sample_size}.tsv.gz"
            )
        specifications.append(
            SnapshotSpec(
                cutoff=snapshot["cutoff"],
                target_start=snapshot["target_start"],
                target_end=snapshot["target_end"],
                customer_count=sample_size,
                existing_candidates_path=existing,
                two_tower_candidates_path=two_tower,
            )
        )
    return specifications


def write_candidates(
    path: Path,
    customer_ids: list[str],
    article_ids: list[str],
    article_indices: np.ndarray,
    scores: np.ndarray,
    cutoff: str,
) -> dict[str, list[str]]:
    by_customer = {customer_id: [] for customer_id in customer_ids}
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["customer_id", "article_id", "rank", "score", "feature_cutoff"])
        for customer_index, customer_id in enumerate(customer_ids):
            for rank, (item_index, score) in enumerate(
                zip(
                    article_indices[customer_index],
                    scores[customer_index],
                ),
                start=1,
            ):
                if int(item_index) < 0:
                    continue
                article_id = article_ids[int(item_index)]
                by_customer[customer_id].append(article_id)
                writer.writerow(
                    [customer_id, article_id, rank, float(score), cutoff]
                )
    return by_customer


def current_union(
    pool: Any,
) -> dict[str, list[str]]:
    union: dict[str, list[str]] = {}
    for customer_id in pool.customer_ids:
        seen: set[str] = set()
        candidates: list[str] = []
        for source_name in BASE_SOURCES:
            for article_id in pool.articles_by_source[source_name].get(
                customer_id, []
            )[:500]:
                if article_id not in seen:
                    seen.add(article_id)
                    candidates.append(article_id)
        for article_id in pool.articles_by_source["two_tower"].get(
            customer_id, []
        )[:TT_BASELINE_K]:
            if article_id not in seen:
                seen.add(article_id)
                candidates.append(article_id)
        union[customer_id] = candidates
    return union


def retrieval_metrics(
    relevance: dict[str, set[str]],
    candidates: dict[str, list[str]],
) -> dict[str, float | int]:
    hits = 0
    customers_with_hit = 0
    candidate_count = 0
    oracle_values: list[float] = []
    for customer_id, relevant in relevance.items():
        proposed = set(candidates.get(customer_id, []))
        customer_hits = len(relevant.intersection(proposed))
        hits += customer_hits
        customers_with_hit += int(customer_hits > 0)
        candidate_count += len(proposed)
        denominator = min(len(relevant), 12)
        oracle_values.append(
            min(customer_hits, 12) / denominator if denominator else 0.0
        )
    total_relevant = sum(len(items) for items in relevance.values())
    customers = len(relevance)
    return {
        "candidate_recall": hits / total_relevant if total_relevant else 0.0,
        "candidate_hit_rate": customers_with_hit / customers if customers else 0.0,
        "average_candidate_count": candidate_count / customers if customers else 0.0,
        "oracle_map_at_12": float(np.mean(oracle_values)),
        "matched_relevant_pairs": hits,
    }


def marginal_report(
    relevance: dict[str, set[str]],
    baseline: dict[str, list[str]],
    image: dict[str, list[str]],
) -> dict[str, Any]:
    baseline_metrics = retrieval_metrics(relevance, baseline)
    results: dict[str, Any] = {"baseline": baseline_metrics}
    for k in K_VALUES:
        combined: dict[str, list[str]] = {}
        for customer_id in relevance:
            seen = set(baseline.get(customer_id, []))
            values = list(baseline.get(customer_id, []))
            for article_id in image.get(customer_id, [])[:k]:
                if article_id not in seen:
                    seen.add(article_id)
                    values.append(article_id)
            combined[customer_id] = values
        image_only = {
            customer_id: values[:k]
            for customer_id, values in image.items()
        }
        combined_metrics = retrieval_metrics(relevance, combined)
        image_metrics = retrieval_metrics(relevance, image_only)
        results[f"at_{k}"] = {
            "image_standalone": image_metrics,
            "baseline_plus_image": combined_metrics,
            "marginal_candidate_recall": (
                float(combined_metrics["candidate_recall"])
                - float(baseline_metrics["candidate_recall"])
            ),
            "marginal_oracle_map_at_12": (
                float(combined_metrics["oracle_map_at_12"])
                - float(baseline_metrics["oracle_map_at_12"])
            ),
        }
    return results


def history_diagnostics(
    con: duckdb.DuckDBPyConnection,
    cutoff: str,
    relevance: dict[str, set[str]],
    baseline: dict[str, list[str]],
    image: dict[str, list[str]],
    k: int,
) -> dict[str, Any]:
    con.execute("CREATE OR REPLACE TEMP TABLE image_eval_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO image_eval_customers VALUES (?)",
        [(customer_id,) for customer_id in relevance],
    )
    rows = con.sql(f"""
        SELECT c.customer_id, COUNT(t.article_id) AS history_count
        FROM image_eval_customers c
        LEFT JOIN transactions t
          ON c.customer_id = t.customer_id
         AND t.transaction_date <= DATE '{cutoff}'
        GROUP BY c.customer_id
    """).fetchall()
    con.execute("DROP TABLE image_eval_customers")
    counts = {customer_id: int(count) for customer_id, count in rows}

    def bucket(count: int) -> str:
        if count == 0:
            return "0"
        if count <= 2:
            return "1-2"
        if count <= 5:
            return "3-5"
        if count <= 10:
            return "6-10"
        if count <= 20:
            return "11-20"
        return "20+"

    result: dict[str, dict[str, int | float]] = {}
    for customer_id, relevant in relevance.items():
        name = bucket(counts.get(customer_id, 0))
        row = result.setdefault(
            name,
            {
                "customers": 0,
                "relevant_pairs": 0,
                "image_hits": 0,
                "unique_image_hits": 0,
            },
        )
        image_hits = relevant.intersection(image.get(customer_id, [])[:k])
        unique_hits = image_hits.difference(baseline.get(customer_id, []))
        row["customers"] = int(row["customers"]) + 1
        row["relevant_pairs"] = int(row["relevant_pairs"]) + len(relevant)
        row["image_hits"] = int(row["image_hits"]) + len(image_hits)
        row["unique_image_hits"] = int(row["unique_image_hits"]) + len(unique_hits)
    for values in result.values():
        total = int(values["relevant_pairs"])
        values["image_recall"] = int(values["image_hits"]) / total if total else 0.0
        values["marginal_recall"] = (
            int(values["unique_image_hits"]) / total if total else 0.0
        )
    return result


def item_diagnostics(
    con: duckdb.DuckDBPyConnection,
    cutoff: str,
    relevance: dict[str, set[str]],
    baseline: dict[str, list[str]],
    image: dict[str, list[str]],
    k: int,
) -> dict[str, Any]:
    rows = con.sql(f"""
        SELECT
            a.article_id,
            COALESCE(stats.purchase_count, 0) AS purchase_count,
            stats.first_purchase,
            a.product_group_name
        FROM articles a
        LEFT JOIN (
            SELECT
                article_id,
                COUNT(*) AS purchase_count,
                MIN(transaction_date) AS first_purchase
            FROM transactions
            WHERE transaction_date <= DATE '{cutoff}'
            GROUP BY article_id
        ) stats USING (article_id)
    """).fetchall()
    threshold = float(np.quantile([int(row[1]) for row in rows], 0.8))
    metadata = {
        article_id: (int(count), first_purchase, product_group or "Unknown")
        for article_id, count, first_purchase, product_group in rows
    }
    totals: dict[str, int] = {}
    unique_hits: dict[str, int] = {}
    group_totals: dict[str, int] = {}
    group_hits: dict[str, int] = {}
    for customer_id, relevant in relevance.items():
        unique = (
            relevant.intersection(image.get(customer_id, [])[:k])
            .difference(baseline.get(customer_id, []))
        )
        for article_id in relevant:
            item = metadata.get(article_id)
            if item is None:
                continue
            count, first_purchase, group = item
            segments = ["popular" if count >= threshold else "long_tail"]
            if first_purchase is not None:
                age = (date.fromisoformat(cutoff) - first_purchase).days
                segments.append("newer_90d" if age <= 90 else "established")
            for segment in segments:
                totals[segment] = totals.get(segment, 0) + 1
                unique_hits[segment] = unique_hits.get(segment, 0) + int(
                    article_id in unique
                )
            group_totals[group] = group_totals.get(group, 0) + 1
            group_hits[group] = group_hits.get(group, 0) + int(article_id in unique)
    top_groups = sorted(group_totals, key=group_totals.get, reverse=True)[:10]
    return {
        "popularity_threshold": threshold,
        "segments": {
            segment: {
                "relevant_pairs": total,
                "unique_image_hits": unique_hits.get(segment, 0),
                "marginal_recall": unique_hits.get(segment, 0) / total,
            }
            for segment, total in totals.items()
        },
        "top_product_groups": {
            group: {
                "relevant_pairs": group_totals[group],
                "unique_image_hits": group_hits.get(group, 0),
                "marginal_recall": group_hits.get(group, 0) / group_totals[group],
            }
            for group in top_groups
        },
    }


def main() -> None:
    training_customers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    validation_customers = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    store = ImageEmbeddingStore.load(OUTPUT_DIR)
    specifications = make_specs(training_customers, validation_customers)
    con = duckdb.connect()
    setup_connection(con)
    reports: list[dict[str, Any]] = []
    final_payload: tuple[Any, ...] | None = None

    for index, specification in enumerate(specifications):
        print(f"Building image candidates for {specification.cutoff}...", flush=True)
        pool = load_candidate_pool(specification)
        relevance = load_relevance(
            con,
            pool.customer_ids,
            specification.target_start,
            specification.target_end,
        )
        profiles = build_visual_profiles(
            con,
            store,
            pool.customer_ids,
            specification.cutoff,
        )
        eligible = eligible_image_item_indices(con, store, specification.cutoff)
        active = np.flatnonzero(profiles.customers_with_profile)
        stem = f"{specification.cutoff}_{specification.customer_count}"
        query_path = OUTPUT_DIR / f"ann_query_{stem}.npz"
        result_path = OUTPUT_DIR / f"ann_result_{stem}.npz"
        metrics_path = OUTPUT_DIR / f"ann_metrics_{stem}.json"
        np.savez_compressed(
            query_path,
            user_embeddings=profiles.embeddings[active],
            eligible_item_indices=eligible,
        )
        ann_started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_image_ann.py"),
                str(OUTPUT_DIR),
                str(query_path),
                str(result_path),
                str(metrics_path),
            ],
            check=True,
        )
        ann_seconds = time.perf_counter() - ann_started
        with np.load(result_path) as arrays:
            filtered_indices, filtered_scores = filter_history_from_ann(
                arrays["article_indices"].astype(np.int64),
                arrays["scores"].astype(np.float32),
                [profiles.history_item_indices[int(value)] for value in active],
                MAX_IMAGE_K,
            )
        full_indices = np.full(
            (len(pool.customer_ids), MAX_IMAGE_K),
            -1,
            dtype=np.int64,
        )
        full_scores = np.full(
            (len(pool.customer_ids), MAX_IMAGE_K),
            -np.inf,
            dtype=np.float32,
        )
        full_indices[active] = filtered_indices
        full_scores[active] = filtered_scores
        candidate_path = OUTPUT_DIR / f"candidates_{stem}.tsv.gz"
        image_candidates = write_candidates(
            candidate_path,
            pool.customer_ids,
            store.article_ids,
            full_indices,
            full_scores,
            specification.cutoff,
        )
        report = {
            "cutoff": specification.cutoff,
            "customers": len(pool.customer_ids),
            "customers_with_visual_profile": len(active),
            "profile_coverage": len(active) / len(pool.customer_ids),
            "eligible_image_articles": len(eligible),
            "ann_seconds": ann_seconds,
            "ann_ms_per_profile": ann_seconds * 1000.0 / max(len(active), 1),
            "ann_diagnostics": json.loads(metrics_path.read_text(encoding="utf-8")),
            "candidate_path": str(candidate_path),
        }
        reports.append(report)
        if index == 4:
            final_payload = (
                pool,
                relevance,
                image_candidates,
                specification.cutoff,
            )
        else:
            del pool, relevance, image_candidates
            gc.collect()

    if final_payload is None:
        raise RuntimeError("Final image retrieval snapshot was not created")
    final_pool, final_relevance, final_image, final_cutoff = final_payload
    baseline = current_union(final_pool)
    final_report = {
        "contract": {
            "encoder": "facebook/dinov2-small",
            "image_only": True,
            "visual_profile": "L2-normalized 45-day recency-weighted mean of last 50 purchases",
            "history_items_excluded_from_image_candidates": True,
            "current_baseline": "five existing sources + two-tower top 50",
            "max_image_candidates": MAX_IMAGE_K,
        },
        "snapshots": reports,
        "final_retrieval": marginal_report(
            final_relevance,
            baseline,
            final_image,
        ),
        "history_segments_at_300": history_diagnostics(
            con,
            final_cutoff,
            final_relevance,
            baseline,
            final_image,
            300,
        ),
        "item_segments_at_300": item_diagnostics(
            con,
            final_cutoff,
            final_relevance,
            baseline,
            final_image,
            300,
        ),
    }
    report_path = OUTPUT_DIR / (
        f"image_retrieval_report_{training_customers}_{validation_customers}.json"
    )
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
