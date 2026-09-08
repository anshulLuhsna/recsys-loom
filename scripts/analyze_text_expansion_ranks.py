#!/usr/bin/env python3
"""Diagnose where new relevant TT+text candidates land after LambdaRank."""

from __future__ import annotations

import gc
import json
import sys
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
    build_feature_data,
    load_candidate_pool,
    load_relevance,
)
from recsys_loom.ranking.ranker import train_ranker
from scripts.run_two_tower_budget_sweep import (
    SNAPSHOTS,
    apply_budget,
    combine_training,
    load_history_buckets,
    ranking_metrics,
    setup_connection,
)
from scripts.run_two_tower_text_ranking import FINAL_FOLD, make_specs

TEXT_BUDGET = 200
OUTPUT_DIR = ROOT / "artifacts" / "text_retrieval"


def candidate_union(
    pool: CandidatePool,
    two_tower_budget: int,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for customer_id in pool.customer_ids:
        candidates: set[str] = set()
        for source in BASE_SOURCES:
            candidates.update(
                pool.articles_by_source[source].get(customer_id, [])[:500]
            )
        candidates.update(
            pool.articles_by_source["two_tower"].get(
                customer_id,
                [],
            )[:two_tower_budget]
        )
        result[customer_id] = candidates
    return result


def summarize_values(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.9)),
    }


def feature_profile(
    features: np.ndarray,
    feature_names: list[str],
    row_indices: list[int],
) -> dict[str, dict[str, float]]:
    names = [
        "num_sources",
        "is_recent_7d_pop",
        "is_repeat_purchase",
        "is_cooccurrence",
        "is_als",
        "is_content",
        "is_two_tower",
        "rank_two_tower",
        "score_two_tower",
        "item_purchases_7d",
        "item_purchases_30d",
        "item_purchase_growth",
        "item_unique_buyers_30d",
    ]
    rows = features[np.asarray(row_indices, dtype=np.int64)]
    result: dict[str, dict[str, float]] = {}
    for name in names:
        values = rows[:, feature_names.index(name)]
        finite = values[np.isfinite(values)]
        result[name] = {
            "mean": float(finite.mean()) if len(finite) else 0.0,
            "median": float(np.median(finite)) if len(finite) else 0.0,
            "coverage": float(len(finite) / len(values)),
        }
    return result


def rank_diagnostics(
    scores: np.ndarray,
    validation: dict[str, np.ndarray],
    article_ids: list[str],
    relevance: dict[str, set[str]],
    baseline_union: dict[str, set[str]],
    text_union: dict[str, set[str]],
) -> dict[str, Any]:
    ranks: list[float] = []
    rank_percentiles: list[float] = []
    new_rows: list[int] = []
    baseline_rows: list[int] = []
    top_counts = {12: 0, 50: 0, 100: 0, 300: 0}
    offset = 0
    for customer_id_value, group_size_value in zip(
        validation["customer_ids"],
        validation["groups"],
    ):
        customer_id = str(customer_id_value)
        group_size = int(group_size_value)
        end = offset + group_size
        order = np.argsort(-scores[offset:end], kind="stable")
        local_ranks = np.empty(group_size, dtype=np.int32)
        local_ranks[order] = np.arange(1, group_size + 1, dtype=np.int32)
        new_relevant = (
            relevance[customer_id]
            .intersection(text_union[customer_id])
            .difference(baseline_union[customer_id])
        )
        baseline_relevant = relevance[customer_id].intersection(
            baseline_union[customer_id]
        )
        for local_index in range(group_size):
            row_index = offset + local_index
            article_id = article_ids[
                int(validation["pair_article_indices"][row_index])
            ]
            if article_id in new_relevant:
                rank = int(local_ranks[local_index])
                ranks.append(float(rank))
                rank_percentiles.append(rank / group_size)
                new_rows.append(row_index)
                for threshold in top_counts:
                    top_counts[threshold] += int(rank <= threshold)
            elif article_id in baseline_relevant:
                baseline_rows.append(row_index)
        offset = end

    feature_names = [
        str(value) for value in validation["feature_names"]
    ]
    return {
        "new_relevant_text_pairs": len(ranks),
        "rank": summarize_values(ranks),
        "rank_percentile": summarize_values(rank_percentiles),
        "top_k_rate": {
            f"top_{threshold}": top_counts[threshold] / len(ranks)
            for threshold in top_counts
        },
        "new_relevant_feature_profile": feature_profile(
            validation["features"],
            feature_names,
            new_rows,
        ),
        "already_retrieved_relevant_feature_profile": feature_profile(
            validation["features"],
            feature_names,
            baseline_rows,
        ),
    }


def main() -> None:
    training_customers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    final_customers = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    con = duckdb.connect()
    setup_connection(con)
    article_ids = [
        row[0]
        for row in con.sql(
            "SELECT article_id FROM articles ORDER BY article_id"
        ).fetchall()
    ]
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    text_specs = make_specs(
        "metadata_text",
        training_customers,
        final_customers,
    )
    metadata_specs = make_specs(
        "metadata",
        training_customers,
        final_customers,
    )
    max_data: list[dict[str, np.ndarray]] = []
    relevance: list[dict[str, set[str]]] = []
    text_union: dict[str, set[str]] = {}
    for index, specification in enumerate(text_specs):
        print(f"Building text-expanded pool {specification.cutoff}...", flush=True)
        pool = load_candidate_pool(specification)
        snapshot_relevance = load_relevance(
            con,
            pool.customer_ids,
            specification.target_start,
            specification.target_end,
        )
        data = build_feature_data(
            con,
            pool,
            snapshot_relevance,
            specification.cutoff,
            SIX_SOURCES,
            article_to_index,
        )
        data["history_buckets"] = np.asarray(
            load_history_buckets(
                con,
                pool.customer_ids,
                specification.cutoff,
            )
        )
        max_data.append(data)
        relevance.append(snapshot_relevance)
        if index == FINAL_FOLD:
            text_union = candidate_union(pool, TEXT_BUDGET)
        del pool
        gc.collect()

    baseline_pool = load_candidate_pool(metadata_specs[FINAL_FOLD])
    baseline_union = candidate_union(baseline_pool, 50)
    del baseline_pool
    training_snapshots = [
        apply_budget(snapshot, TEXT_BUDGET)
        for snapshot in max_data[:FINAL_FOLD]
    ]
    validation = apply_budget(max_data[FINAL_FOLD], TEXT_BUDGET)
    train = combine_training(training_snapshots)
    del training_snapshots
    gc.collect()
    feature_names = [str(value) for value in train["feature_names"]]
    model = train_ranker(
        train["features"],
        train["labels"],
        train["groups"],
        feature_names,
        validation["features"],
        validation["labels"],
        validation["groups"],
        verbose=0,
    )
    scores = model.predict(validation["features"])
    metrics, _ = ranking_metrics(
        scores,
        validation,
        article_ids,
        relevance[FINAL_FOLD],
    )
    diagnostics = rank_diagnostics(
        scores,
        validation,
        article_ids,
        relevance[FINAL_FOLD],
        baseline_union,
        text_union,
    )
    report = {
        "contract": {
            "baseline": "five sources plus metadata TT K=50",
            "treatment": "five sources plus text TT K=200",
            "ranker": "LambdaRank retrained on text-expanded temporal pools",
            "final_week": SNAPSHOTS[FINAL_FOLD],
        },
        "treatment_ranker_metrics": metrics,
        "diagnostics": diagnostics,
    }
    path = OUTPUT_DIR / (
        f"text_expansion_rank_diagnostic_{training_customers}_{final_customers}.json"
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
