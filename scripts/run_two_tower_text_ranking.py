#!/usr/bin/env python3
"""Controlled LambdaRank comparison replacing metadata TT with text TT."""

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
    SnapshotSpec,
    build_feature_data,
    load_candidate_pool,
    load_relevance,
)
from scripts.run_two_tower_budget_sweep import (
    SNAPSHOTS,
    apply_budget,
    combine_training,
    load_history_buckets,
    ranking_metrics,
    setup_connection,
)
from recsys_loom.ranking.ranker import train_ranker

SELECTION_FOLDS = [2, 3]
FINAL_FOLD = 4
TT_BUDGET = 50
OUTPUT_DIR = ROOT / "artifacts" / "text_retrieval"
ABLATION_DIR = OUTPUT_DIR / "two_tower_ablation"


def make_specs(
    arm: str,
    training_customers: int,
    final_customers: int,
) -> list[SnapshotSpec]:
    result: list[SnapshotSpec] = []
    for index, snapshot in enumerate(SNAPSHOTS):
        sample_size = final_customers if index == FINAL_FOLD else training_customers
        existing_path = (
            ROOT
            / "artifacts"
            / "two_tower"
            / f"existing_candidates_{sample_size}.tsv.gz"
            if index == FINAL_FOLD
            else ROOT
            / "artifacts"
            / "ranking"
            / "candidate_cache"
            / f"existing_{snapshot['cutoff']}_{sample_size}.tsv.gz"
        )
        result.append(
            SnapshotSpec(
                cutoff=snapshot["cutoff"],
                target_start=snapshot["target_start"],
                target_end=snapshot["target_end"],
                customer_count=sample_size,
                existing_candidates_path=existing_path,
                two_tower_candidates_path=(
                    ABLATION_DIR
                    / arm
                    / f"candidates_{snapshot['cutoff']}_{sample_size}.tsv.gz"
                ),
            )
        )
    return result


def candidate_metrics(
    data: dict[str, np.ndarray],
    relevance: dict[str, set[str]],
) -> dict[str, float | int]:
    offset = 0
    matched = 0
    customers_with_hit = 0
    oracle_values: list[float] = []
    for customer_id, group_size_value in zip(data["customer_ids"], data["groups"]):
        group_size = int(group_size_value)
        end = offset + group_size
        hits = int(data["labels"][offset:end].sum())
        relevant_count = len(relevance.get(str(customer_id), set()))
        matched += hits
        customers_with_hit += int(hits > 0)
        denominator = min(relevant_count, 12)
        oracle_values.append(
            min(hits, 12) / denominator if denominator else 0.0
        )
        offset = end
    total_relevant = sum(len(items) for items in relevance.values())
    customers = len(data["groups"])
    return {
        "candidate_recall": matched / total_relevant if total_relevant else 0.0,
        "candidate_hit_rate": customers_with_hit / customers,
        "oracle_map_at_12": float(np.mean(oracle_values)),
        "average_candidate_count": len(data["labels"]) / customers,
        "matched_relevant_pairs": matched,
    }


def run_fold(
    snapshots: list[dict[str, np.ndarray]],
    relevance: list[dict[str, set[str]]],
    validation_index: int,
    article_ids: list[str],
) -> dict[str, Any]:
    train_snapshots = [
        apply_budget(snapshot, TT_BUDGET)
        for snapshot in snapshots[:validation_index]
    ]
    validation = apply_budget(snapshots[validation_index], TT_BUDGET)
    train = combine_training(train_snapshots)
    del train_snapshots
    gc.collect()
    model = train_ranker(
        train["features"],
        train["labels"],
        train["groups"],
        [str(value) for value in train["feature_names"]],
        validation["features"],
        validation["labels"],
        validation["groups"],
        verbose=0,
    )
    scoring_started = time.perf_counter()
    scores = model.predict(validation["features"])
    scoring_seconds = time.perf_counter() - scoring_started
    ranker, _ = ranking_metrics(
        scores,
        validation,
        article_ids,
        relevance[validation_index],
    )
    ranker["ranker_scoring_ms_per_customer"] = (
        scoring_seconds * 1000.0 / len(validation["groups"])
    )
    result = {
        "candidate": candidate_metrics(
            validation,
            relevance[validation_index],
        ),
        "ranker": ranker,
    }
    del train, validation, model, scores
    gc.collect()
    return result


def build_arm_data(
    con: duckdb.DuckDBPyConnection,
    specifications: list[SnapshotSpec],
    article_to_index: dict[str, int],
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, set[str]]]]:
    snapshots: list[dict[str, np.ndarray]] = []
    relevance_by_snapshot: list[dict[str, set[str]]] = []
    for specification in specifications:
        print(
            f"  building {specification.cutoff} feature data...",
            flush=True,
        )
        pool = load_candidate_pool(specification)
        relevance = load_relevance(
            con,
            pool.customer_ids,
            specification.target_start,
            specification.target_end,
        )
        data = build_feature_data(
            con,
            pool,
            relevance,
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
        snapshots.append(data)
        relevance_by_snapshot.append(relevance)
        del pool
        gc.collect()
    return snapshots, relevance_by_snapshot


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
    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ("metadata", "metadata_text"):
        print(f"\nLambdaRank arm: {arm}", flush=True)
        specifications = make_specs(arm, training_customers, final_customers)
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
            raise FileNotFoundError(
                "Missing candidate artifacts:\n" + "\n".join(missing)
            )
        snapshots, relevance = build_arm_data(
            con,
            specifications,
            article_to_index,
        )
        fold_results = {
            str(index): run_fold(
                snapshots,
                relevance,
                index,
                article_ids,
            )
            for index in [*SELECTION_FOLDS, FINAL_FOLD]
        }
        arm_results[arm] = {
            "folds": fold_results,
            "selection_mean_map_at_12": float(
                np.mean(
                    [
                        fold_results[str(index)]["ranker"]["map_at_12"]
                        for index in SELECTION_FOLDS
                    ]
                )
            ),
            "final": fold_results[str(FINAL_FOLD)],
        }
        del snapshots, relevance
        gc.collect()

    delta: dict[str, float] = {}
    for section, names in {
        "candidate": [
            "candidate_recall",
            "candidate_hit_rate",
            "oracle_map_at_12",
            "average_candidate_count",
        ],
        "ranker": [
            "map_at_12",
            "recall_at_12",
            "hit_rate_at_12",
            "average_candidate_count",
            "ranker_scoring_ms_per_customer",
        ],
    }.items():
        for name in names:
            delta[f"{section}.{name}"] = (
                float(arm_results["metadata_text"]["final"][section][name])
                - float(arm_results["metadata"]["final"][section][name])
            )
    report = {
        "contract": {
            "two_tower_budget": TT_BUDGET,
            "candidate_sources": SIX_SOURCES,
            "training_customers_per_historical_snapshot": training_customers,
            "final_customers": final_customers,
            "selection_fold_indices": SELECTION_FOLDS,
            "final_fold_index": FINAL_FOLD,
            "only_change": "metadata-only TT candidates replaced by metadata+text TT candidates",
        },
        "arms": arm_results,
        "final_delta_metadata_text_minus_metadata": delta,
        "selection_mean_map_delta": (
            arm_results["metadata_text"]["selection_mean_map_at_12"]
            - arm_results["metadata"]["selection_mean_map_at_12"]
        ),
    }
    path = OUTPUT_DIR / (
        f"two_tower_text_ranking_{training_customers}_{final_customers}.json"
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
