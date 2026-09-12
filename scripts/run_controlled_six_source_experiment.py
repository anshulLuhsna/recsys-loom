#!/usr/bin/env python3
"""Matched cached comparison of five-source and six-source LambdaRank."""

from __future__ import annotations

import gc
import json
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
    SIX_SOURCES,
    CandidatePool,
    SnapshotSpec,
    build_feature_data,
    build_union,
    candidate_pool_metrics,
    load_candidate_pool,
    load_relevance,
    oracle_map_at_12,
    write_json,
)
from recsys_loom.ranking.ranker import train_ranker

TRAIN_SPECS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
]
FINAL_SPEC = {
    "cutoff": "2020-09-15",
    "target_start": "2020-09-16",
    "target_end": "2020-09-22",
}
OUTPUT_DIR = ROOT / "artifacts" / "controlled_six_source"


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '6GB'")
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


def make_specs(train_customers: int, validation_customers: int) -> list[SnapshotSpec]:
    existing_dir = ROOT / "artifacts" / "ranking" / "candidate_cache"
    two_tower_dir = ROOT / "artifacts" / "two_tower" / "ranking_snapshots"
    specs = [
        SnapshotSpec(
            cutoff=values["cutoff"],
            target_start=values["target_start"],
            target_end=values["target_end"],
            customer_count=train_customers,
            existing_candidates_path=(
                existing_dir
                / f"existing_{values['cutoff']}_{train_customers}.tsv.gz"
            ),
            two_tower_candidates_path=(
                two_tower_dir
                / f"candidates_{values['cutoff']}_{train_customers}.tsv.gz"
            ),
        )
        for values in TRAIN_SPECS
    ]
    specs.append(
        SnapshotSpec(
            cutoff=FINAL_SPEC["cutoff"],
            target_start=FINAL_SPEC["target_start"],
            target_end=FINAL_SPEC["target_end"],
            customer_count=validation_customers,
            existing_candidates_path=(
                ROOT
                / "artifacts"
                / "two_tower"
                / f"existing_candidates_{validation_customers}.tsv.gz"
            ),
            two_tower_candidates_path=(
                ROOT
                / "artifacts"
                / "two_tower"
                / f"two_tower_candidates_{validation_customers}.tsv.gz"
            ),
        )
    )
    return specs


def validate_inputs(specifications: list[SnapshotSpec]) -> None:
    missing = [
        str(path)
        for specification in specifications
        for path in (
            specification.existing_candidates_path,
            specification.two_tower_candidates_path,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing candidate caches:\n" + "\n".join(missing))


def evaluate_candidate_arms(
    pool: CandidatePool,
    relevance: dict[str, set[str]],
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    five_union = build_union(pool, BASE_SOURCES, k_per_source=500)
    six_union = build_union(pool, SIX_SOURCES, k_per_source=500)
    five_metrics = candidate_pool_metrics(relevance, five_union)
    six_metrics = candidate_pool_metrics(relevance, six_union)
    result = {
        "five_source": {
            **five_metrics,
            "oracle_map_at_12": oracle_map_at_12(relevance, five_union),
        },
        "six_source": {
            **six_metrics,
            "oracle_map_at_12": oracle_map_at_12(relevance, six_union),
        },
        "delta": {
            "micro_recall": (
                float(six_metrics["micro_recall"])
                - float(five_metrics["micro_recall"])
            ),
            "hit_rate": (
                float(six_metrics["hit_rate"])
                - float(five_metrics["hit_rate"])
            ),
            "oracle_map_at_12": (
                oracle_map_at_12(relevance, six_union)
                - oracle_map_at_12(relevance, five_union)
            ),
            "average_candidate_count": (
                float(six_metrics["average_candidate_count"])
                - float(five_metrics["average_candidate_count"])
            ),
        },
    }
    return result, five_union, six_union


def history_bucket(count: int) -> str:
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


def analyze_history_buckets(
    con: duckdb.DuckDBPyConnection,
    pool: CandidatePool,
    relevance: dict[str, set[str]],
    five_union: dict[str, list[str]],
    six_union: dict[str, list[str]],
    cutoff: str,
) -> dict[str, Any]:
    con.execute("CREATE OR REPLACE TEMP TABLE segment_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO segment_customers VALUES (?)",
        [(customer_id,) for customer_id in pool.customer_ids],
    )
    rows = con.sql(f"""
        SELECT c.customer_id, COUNT(t.article_id) AS history_count
        FROM segment_customers c
        LEFT JOIN transactions t
          ON t.customer_id = c.customer_id
         AND t.transaction_date <= DATE '{cutoff}'
        GROUP BY c.customer_id
    """).fetchall()
    con.execute("DROP TABLE segment_customers")
    counts = {customer_id: int(count) for customer_id, count in rows}

    result: dict[str, Any] = {}
    for bucket_name in ["0", "1-2", "3-5", "6-10", "11-20", "20+"]:
        customers = [
            customer_id
            for customer_id in pool.customer_ids
            if history_bucket(counts.get(customer_id, 0)) == bucket_name
        ]
        if not customers:
            continue
        bucket_relevance = {
            customer_id: relevance[customer_id] for customer_id in customers
        }
        two_tower = {
            customer_id: pool.articles_by_source["two_tower"].get(customer_id, [])[:500]
            for customer_id in customers
        }
        five = {customer_id: five_union[customer_id] for customer_id in customers}
        six = {customer_id: six_union[customer_id] for customer_id in customers}
        tt_metrics = candidate_pool_metrics(bucket_relevance, two_tower)
        five_metrics = candidate_pool_metrics(bucket_relevance, five)
        six_metrics = candidate_pool_metrics(bucket_relevance, six)
        result[bucket_name] = {
            "customers": len(customers),
            "two_tower_recall_at_500": tt_metrics["micro_recall"],
            "five_source_union_recall": five_metrics["micro_recall"],
            "six_source_union_recall": six_metrics["micro_recall"],
            "marginal_recall": (
                float(six_metrics["micro_recall"])
                - float(five_metrics["micro_recall"])
            ),
        }
    return result


def analyze_als_overlap(
    pool: CandidatePool,
    relevance: dict[str, set[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in (100, 300, 500):
        candidate_intersection = 0
        candidate_union = 0
        relevant_intersection = 0
        relevant_union = 0
        for customer_id, relevant in relevance.items():
            als = set(pool.articles_by_source["als"].get(customer_id, [])[:k])
            two_tower = set(
                pool.articles_by_source["two_tower"].get(customer_id, [])[:k]
            )
            candidate_intersection += len(als.intersection(two_tower))
            candidate_union += len(als.union(two_tower))
            als_hits = als.intersection(relevant)
            two_tower_hits = two_tower.intersection(relevant)
            relevant_intersection += len(als_hits.intersection(two_tower_hits))
            relevant_union += len(als_hits.union(two_tower_hits))
        result[f"at_{k}"] = {
            "candidate_jaccard": (
                candidate_intersection / candidate_union
                if candidate_union
                else 0.0
            ),
            "relevant_hit_jaccard": (
                relevant_intersection / relevant_union
                if relevant_union
                else 0.0
            ),
            "shared_relevant_hits": relevant_intersection,
            "combined_relevant_hits": relevant_union,
        }
    return result


def analyze_item_segments(
    con: duckdb.DuckDBPyConnection,
    pool: CandidatePool,
    relevance: dict[str, set[str]],
    five_union: dict[str, list[str]],
    cutoff: str,
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
    purchase_counts = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    popularity_threshold = float(np.quantile(purchase_counts, 0.8))
    metadata = {
        article_id: {
            "purchase_count": int(count),
            "first_purchase": first_purchase,
            "product_group": product_group or "Unknown",
        }
        for article_id, count, first_purchase, product_group in rows
    }

    segment_totals: dict[str, int] = {}
    segment_unique_hits: dict[str, int] = {}
    group_totals: dict[str, int] = {}
    group_unique_hits: dict[str, int] = {}

    for customer_id, relevant_items in relevance.items():
        existing = set(five_union[customer_id])
        two_tower = set(
            pool.articles_by_source["two_tower"].get(customer_id, [])[:500]
        )
        unique_hits = relevant_items.intersection(two_tower).difference(existing)
        for article_id in relevant_items:
            item = metadata.get(article_id)
            if item is None:
                continue
            segments = [
                (
                    "popular"
                    if int(item["purchase_count"]) >= popularity_threshold
                    else "long_tail"
                )
            ]
            first_purchase = item["first_purchase"]
            if first_purchase is not None:
                days_old = (date.fromisoformat(cutoff) - first_purchase).days
                segments.append("newer_90d" if days_old <= 90 else "established")
            for segment in segments:
                segment_totals[segment] = segment_totals.get(segment, 0) + 1
                if article_id in unique_hits:
                    segment_unique_hits[segment] = (
                        segment_unique_hits.get(segment, 0) + 1
                    )
            product_group = str(item["product_group"])
            group_totals[product_group] = group_totals.get(product_group, 0) + 1
            if article_id in unique_hits:
                group_unique_hits[product_group] = (
                    group_unique_hits.get(product_group, 0) + 1
                )

    segments = {
        segment: {
            "relevant_pairs": total,
            "unique_two_tower_hits": segment_unique_hits.get(segment, 0),
            "marginal_recall": segment_unique_hits.get(segment, 0) / total,
        }
        for segment, total in segment_totals.items()
    }
    top_groups = sorted(group_totals, key=group_totals.get, reverse=True)[:10]
    groups = {
        group: {
            "relevant_pairs": group_totals[group],
            "unique_two_tower_hits": group_unique_hits.get(group, 0),
            "marginal_recall": (
                group_unique_hits.get(group, 0) / group_totals[group]
            ),
        }
        for group in top_groups
    }
    return {
        "popularity_threshold": popularity_threshold,
        "segments": segments,
        "top_product_groups": groups,
    }


def evaluate_ranked_scores(
    scores: NDArray[np.float64],
    data: dict[str, NDArray[Any]],
    article_ids: list[str],
    relevance: dict[str, set[str]],
) -> dict[str, float | int]:
    pair_customers = data["pair_customer_indices"]
    pair_articles = data["pair_article_indices"]
    customer_ids = data["customer_ids"].tolist()
    predictions: dict[str, list[str]] = {}
    start = 0
    for customer_index, group_size in enumerate(data["groups"]):
        end = start + int(group_size)
        group_scores = scores[start:end]
        top_count = min(12, len(group_scores))
        top_local = np.argpartition(
            -group_scores,
            kth=top_count - 1,
        )[:top_count]
        top_local = top_local[np.argsort(-group_scores[top_local])]
        predictions[str(customer_ids[customer_index])] = [
            article_ids[int(pair_articles[start + int(local_index)])]
            for local_index in top_local
        ]
        start = end

    average_precisions: list[float] = []
    hits = 0
    customers_with_hit = 0
    relevant_count = 0
    for customer_id, relevant in relevance.items():
        predicted = predictions.get(customer_id, [])
        customer_hits = 0
        precision_sum = 0.0
        for rank, article_id in enumerate(predicted, start=1):
            if article_id in relevant:
                customer_hits += 1
                precision_sum += customer_hits / rank
        average_precisions.append(
            precision_sum / min(len(relevant), 12) if relevant else 0.0
        )
        hits += customer_hits
        customers_with_hit += int(customer_hits > 0)
        relevant_count += len(relevant)
    customers = len(relevance)
    return {
        "map_at_12": float(np.mean(average_precisions)),
        "recall_at_12": hits / relevant_count if relevant_count else 0.0,
        "hit_rate_at_12": customers_with_hit / customers if customers else 0.0,
        "customers": customers,
    }


def run_temporal_ranker_folds(
    system_name: str,
    snapshots: list[dict[str, NDArray[Any]]],
    article_ids: list[str],
    relevance_by_snapshot: list[dict[str, set[str]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for validation_index in (2, 3, 4):
        train_parts = snapshots[:validation_index]
        validation = snapshots[validation_index]
        features = np.concatenate([part["features"] for part in train_parts])
        labels = np.concatenate([part["labels"] for part in train_parts])
        groups = np.concatenate([part["groups"] for part in train_parts])
        feature_names = train_parts[0]["feature_names"].tolist()
        model = train_ranker(
            features,
            labels,
            groups,
            feature_names,
            validation["features"],
            validation["labels"],
            validation["groups"],
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            verbose=0,
        )
        scores = model.predict(validation["features"])
        metrics = evaluate_ranked_scores(
            scores,
            validation,
            article_ids,
            relevance_by_snapshot[validation_index],
        )
        results.append(
            {
                "system": system_name,
                "validation_cutoff": (
                    TRAIN_SPECS[validation_index]["cutoff"]
                    if validation_index < 4
                    else FINAL_SPEC["cutoff"]
                ),
                "training_snapshot_count": validation_index,
                "metrics": metrics,
            }
        )
        del features, labels, groups, model, scores
        gc.collect()
    return results


def build_system_snapshots(
    con: duckdb.DuckDBPyConnection,
    specifications: list[SnapshotSpec],
    relevance_by_snapshot: list[dict[str, set[str]]],
    source_names: list[str],
    article_to_index: dict[str, int],
) -> list[dict[str, NDArray[Any]]]:
    snapshots: list[dict[str, NDArray[Any]]] = []
    for specification, relevance in zip(
        specifications,
        relevance_by_snapshot,
    ):
        print(
            f"  {len(source_names)} sources @ {specification.cutoff}...",
            flush=True,
        )
        pool = load_candidate_pool(specification)
        snapshots.append(
            build_feature_data(
                con,
                pool,
                relevance,
                specification.cutoff,
                source_names,
                article_to_index,
            )
        )
        del pool
        gc.collect()
    return snapshots


def main() -> None:
    train_customers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    validation_customers = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    specifications = make_specs(train_customers, validation_customers)
    validate_inputs(specifications)

    con = duckdb.connect()
    setup_connection(con)
    article_ids = [
        row[0]
        for row in con.sql("SELECT article_id FROM articles ORDER BY article_id").fetchall()
    ]
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }

    candidate_reports: list[dict[str, Any]] = []
    relevance_by_snapshot: list[dict[str, set[str]]] = []
    final_diagnostics: dict[str, Any] = {}

    for specification in specifications:
        print(f"\nLoading cached candidates for {specification.cutoff}...", flush=True)
        pool = load_candidate_pool(specification)
        relevance = load_relevance(
            con,
            pool.customer_ids,
            specification.target_start,
            specification.target_end,
        )
        relevance_by_snapshot.append(relevance)
        candidate_report, five_union, six_union = evaluate_candidate_arms(
            pool,
            relevance,
        )
        candidate_reports.append(
            {
                "cutoff": specification.cutoff,
                "target_start": specification.target_start,
                "target_end": specification.target_end,
                **candidate_report,
            }
        )

        if specification == specifications[-1]:
            final_diagnostics = {
                "history_buckets": analyze_history_buckets(
                    con,
                    pool,
                    relevance,
                    five_union,
                    six_union,
                    specification.cutoff,
                ),
                "als_overlap": analyze_als_overlap(pool, relevance),
                "item_segments": analyze_item_segments(
                    con,
                    pool,
                    relevance,
                    five_union,
                    specification.cutoff,
                ),
            }

        del pool, five_union, six_union
        gc.collect()

    print("\nBuilding in-memory five-source features...", flush=True)
    five_snapshots = build_system_snapshots(
        con,
        specifications,
        relevance_by_snapshot,
        BASE_SOURCES,
        article_to_index,
    )
    five_ranker = run_temporal_ranker_folds(
        "five_source",
        five_snapshots,
        article_ids,
        relevance_by_snapshot,
    )
    del five_snapshots
    gc.collect()

    print("\nBuilding in-memory six-source features...", flush=True)
    six_snapshots = build_system_snapshots(
        con,
        specifications,
        relevance_by_snapshot,
        SIX_SOURCES,
        article_to_index,
    )
    six_ranker = run_temporal_ranker_folds(
        "six_source",
        six_snapshots,
        article_ids,
        relevance_by_snapshot,
    )
    del six_snapshots
    gc.collect()
    temporal_comparison = []
    for five_result, six_result in zip(five_ranker, six_ranker):
        five_metrics = five_result["metrics"]
        six_metrics = six_result["metrics"]
        temporal_comparison.append(
            {
                "validation_cutoff": five_result["validation_cutoff"],
                "five_source": five_metrics,
                "six_source": six_metrics,
                "delta": {
                    metric: float(six_metrics[metric]) - float(five_metrics[metric])
                    for metric in ("map_at_12", "recall_at_12", "hit_rate_at_12")
                },
            }
        )

    final_delta_positive = all(
        comparison["delta"]["map_at_12"] > 0
        for comparison in temporal_comparison
    )
    report = {
        "contract": {
            "train_customers_per_snapshot": train_customers,
            "final_validation_customers": validation_customers,
            "existing_source_budget": 500,
            "popularity_budget": 100,
            "two_tower_budget": 300,
            "only_treatment_change": "two_tower candidates and source features",
        },
        "candidate_comparison_by_snapshot": candidate_reports,
        "ranker_temporal_comparison": temporal_comparison,
        "two_tower_diagnostics": final_diagnostics,
        "six_source_consistently_beats_five_source": final_delta_positive,
        "freeze_six_source_as_baseline": final_delta_positive,
        "feature_storage": "in-memory, one arm at a time",
        "runtime_seconds": time.monotonic() - started,
    }
    report_path = OUTPUT_DIR / "controlled_comparison_report.json"
    write_json(report_path, report)
    print(f"\nReport: {report_path}")
    con.close()


if __name__ == "__main__":
    main()
