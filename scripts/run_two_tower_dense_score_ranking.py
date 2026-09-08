#!/usr/bin/env python3
"""Rank an unchanged baseline pool with dense two-tower compatibility features."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.ranking.cached_data import (
    SIX_SOURCES,
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

OUTPUT_DIR = ROOT / "artifacts" / "text_retrieval"
DENSE_DIR = OUTPUT_DIR / "dense_scores"
TT_BUDGET = 50
VALIDATION_FOLDS = [2, 3, 4]
ARMS = [
    "baseline",
    "dense_metadata_score_rank",
    "dense_text_score",
    "dense_text_score_rank",
]


def dense_scores(
    data: dict[str, np.ndarray],
    cutoff: str,
) -> tuple[np.ndarray, np.ndarray]:
    saved_customers = np.load(
        DENSE_DIR / cutoff / "customer_ids.npy"
    ).astype(str)
    if saved_customers.tolist() != data["customer_ids"].astype(str).tolist():
        raise RuntimeError(f"Dense-score customer alignment failed at {cutoff}")
    pair_customers = data["pair_customer_indices"]
    pair_articles = data["pair_article_indices"]
    result: list[np.ndarray] = []
    for arm in ("metadata", "metadata_text"):
        users = np.load(
            DENSE_DIR / cutoff / f"{arm}_user_embeddings.npy",
            mmap_mode="r",
        )
        items = np.load(
            DENSE_DIR / cutoff / f"{arm}_item_embeddings.npy",
            mmap_mode="r",
        )
        scores = np.empty(len(pair_customers), dtype=np.float32)
        for start in range(0, len(scores), 200_000):
            end = min(start + 200_000, len(scores))
            scores[start:end] = np.einsum(
                "ij,ij->i",
                users[pair_customers[start:end]],
                items[pair_articles[start:end]],
                optimize=True,
            )
        result.append(scores)
    return result[0], result[1]


def group_ranks(
    scores: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.empty(len(scores), dtype=np.float32)
    percentiles = np.empty(len(scores), dtype=np.float32)
    offset = 0
    for group_size_value in groups:
        group_size = int(group_size_value)
        end = offset + group_size
        order = np.argsort(-scores[offset:end], kind="stable")
        local_ranks = np.empty(group_size, dtype=np.float32)
        local_ranks[order] = np.arange(1, group_size + 1, dtype=np.float32)
        ranks[offset:end] = local_ranks
        percentiles[offset:end] = local_ranks / group_size
        offset = end
    return ranks, percentiles


def build_arm_snapshot(
    max_data: dict[str, np.ndarray],
    arm: str,
) -> dict[str, np.ndarray]:
    filtered = apply_budget(max_data, TT_BUDGET)
    feature_names = [str(value) for value in filtered["feature_names"]]
    metadata_score_index = feature_names.index("dense_metadata_tt_score")
    text_score_index = feature_names.index("dense_text_tt_score")
    base_feature_count = len(feature_names) - 2
    base_features = filtered["features"][:, :base_feature_count]
    base_names = feature_names[:base_feature_count]
    if arm == "baseline":
        features = base_features
        names = base_names
    elif arm == "dense_metadata_score_rank":
        score = filtered["features"][:, metadata_score_index]
        rank, rank_percentile = group_ranks(score, filtered["groups"])
        features = np.column_stack(
            [base_features, score, rank, rank_percentile]
        ).astype(np.float32)
        names = [
            *base_names,
            "dense_metadata_tt_score",
            "dense_metadata_tt_rank",
            "dense_metadata_tt_rank_pct",
        ]
    elif arm == "dense_text_score":
        score = filtered["features"][:, text_score_index]
        features = np.column_stack([base_features, score]).astype(np.float32)
        names = [*base_names, "dense_text_tt_score"]
    elif arm == "dense_text_score_rank":
        score = filtered["features"][:, text_score_index]
        rank, rank_percentile = group_ranks(score, filtered["groups"])
        features = np.column_stack(
            [base_features, score, rank, rank_percentile]
        ).astype(np.float32)
        names = [
            *base_names,
            "dense_text_tt_score",
            "dense_text_tt_rank",
            "dense_text_tt_rank_pct",
        ]
    else:
        raise ValueError(f"Unknown arm: {arm}")
    filtered["features"] = features
    filtered["feature_names"] = np.asarray(names)
    return filtered


def run_fold(
    max_data: list[dict[str, np.ndarray]],
    relevance: list[dict[str, set[str]]],
    validation_index: int,
    arm: str,
    article_ids: list[str],
) -> dict[str, Any]:
    train_snapshots = [
        build_arm_snapshot(snapshot, arm)
        for snapshot in max_data[:validation_index]
    ]
    validation = build_arm_snapshot(max_data[validation_index], arm)
    train = combine_training(train_snapshots)
    del train_snapshots
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
    started = time.perf_counter()
    scores = model.predict(validation["features"])
    scoring_seconds = time.perf_counter() - started
    metrics, _ = ranking_metrics(
        scores,
        validation,
        article_ids,
        relevance[validation_index],
    )
    metrics["ranker_scoring_ms_per_customer"] = (
        scoring_seconds * 1000.0 / len(validation["groups"])
    )
    gains = model.feature_importance(importance_type="gain")
    importance = {
        name: float(gain)
        for name, gain in zip(feature_names, gains)
        if name.startswith("dense_")
    }
    result = {
        "ranker": metrics,
        "dense_feature_gain": importance,
    }
    del train, validation, model, scores
    gc.collect()
    return result


def score_diagnostics(data: dict[str, np.ndarray]) -> dict[str, float]:
    names = [str(value) for value in data["feature_names"]]
    metadata_dense = data["features"][:, names.index("dense_metadata_tt_score")]
    text_dense = data["features"][:, names.index("dense_text_tt_score")]
    labels = data["labels"].astype(bool)
    source_flag = data["features"][:, names.index("is_two_tower")] == 1.0
    source_score = data["features"][:, names.index("score_two_tower")]
    differences = np.abs(metadata_dense[source_flag] - source_score[source_flag])
    return {
        "metadata_dense_source_score_mean_absolute_difference": float(
            differences.mean()
        ),
        "metadata_dense_source_score_p99_absolute_difference": float(
            np.quantile(differences, 0.99)
        ),
        "metadata_positive_mean": float(metadata_dense[labels].mean()),
        "metadata_negative_mean": float(metadata_dense[~labels].mean()),
        "text_positive_mean": float(text_dense[labels].mean()),
        "text_negative_mean": float(text_dense[~labels].mean()),
        "metadata_text_score_correlation": float(
            np.corrcoef(metadata_dense, text_dense)[0, 1]
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
    saved_article_ids = np.load(DENSE_DIR / "article_ids.npy").astype(str)
    if saved_article_ids.tolist() != article_ids:
        raise RuntimeError("Dense-score article alignment failed")
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    specifications = make_specs(
        "metadata",
        training_customers,
        final_customers,
    )
    max_data: list[dict[str, np.ndarray]] = []
    relevance: list[dict[str, set[str]]] = []
    diagnostics: dict[str, dict[str, float]] = {}
    for specification in specifications:
        print(f"Building unchanged pool {specification.cutoff}...", flush=True)
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
        metadata_score, text_score = dense_scores(data, specification.cutoff)
        data["features"] = np.column_stack(
            [data["features"], metadata_score, text_score]
        ).astype(np.float32)
        data["feature_names"] = np.asarray(
            [
                *[str(value) for value in data["feature_names"]],
                "dense_metadata_tt_score",
                "dense_text_tt_score",
            ]
        )
        diagnostics[specification.cutoff] = score_diagnostics(data)
        max_data.append(data)
        relevance.append(snapshot_relevance)
        del pool, metadata_score, text_score
        gc.collect()

    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        print(f"\nLambdaRank arm: {arm}", flush=True)
        results[arm] = {
            str(index): run_fold(
                max_data,
                relevance,
                index,
                arm,
                article_ids,
            )
            for index in VALIDATION_FOLDS
        }
        results[arm]["earlier_fold_mean_map_at_12"] = float(
            np.mean(
                [
                    results[arm][str(index)]["ranker"]["map_at_12"]
                    for index in (2, 3)
                ]
            )
        )

    baseline_final = results["baseline"]["4"]["ranker"]
    final_deltas = {
        arm: {
            metric: (
                float(results[arm]["4"]["ranker"][metric])
                - float(baseline_final[metric])
            )
            for metric in ("map_at_12", "recall_at_12", "hit_rate_at_12")
        }
        for arm in ARMS
        if arm != "baseline"
    }
    report = {
        "contract": {
            "candidate_pool": "unchanged five sources plus metadata TT K=50",
            "added_candidates": 0,
            "validation_folds": VALIDATION_FOLDS,
            "arms": ARMS,
        },
        "score_diagnostics": diagnostics,
        "results": results,
        "final_deltas_vs_baseline": final_deltas,
    }
    path = OUTPUT_DIR / (
        f"two_tower_dense_score_ranking_{training_customers}_{final_customers}.json"
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
