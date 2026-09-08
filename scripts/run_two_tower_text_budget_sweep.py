#!/usr/bin/env python3
"""Select TT+text candidate depth on earlier folds, then evaluate final once."""

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
    SIX_SOURCES,
    build_feature_data,
    load_candidate_pool,
    load_relevance,
)
from scripts.run_two_tower_budget_sweep import (
    SNAPSHOTS,
    apply_budget,
    load_history_buckets,
    run_arm,
    setup_connection,
)
from scripts.run_two_tower_text_ranking import make_specs

TT_BUDGETS = [0, 25, 50, 100, 200, 300]
SELECTION_FOLDS = [2, 3]
FINAL_FOLD = 4
MAP_TOLERANCE = 0.0002
OUTPUT_DIR = ROOT / "artifacts" / "text_retrieval"


def build_data(
    con: duckdb.DuckDBPyConnection,
    training_customers: int,
    final_customers: int,
    article_to_index: dict[str, int],
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, set[str]]]]:
    specifications = make_specs(
        "metadata_text",
        training_customers,
        final_customers,
    )
    snapshots: list[dict[str, np.ndarray]] = []
    relevance: list[dict[str, set[str]]] = []
    for specification in specifications:
        print(f"Building {specification.cutoff}...", flush=True)
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
        snapshots.append(data)
        relevance.append(snapshot_relevance)
        del pool
        gc.collect()
    return snapshots, relevance


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
    snapshots, relevance = build_data(
        con,
        training_customers,
        final_customers,
        article_to_index,
    )
    selection_results: dict[int, list[dict[str, Any]]] = {
        budget: [] for budget in TT_BUDGETS
    }
    for validation_index in SELECTION_FOLDS:
        print(f"\nSelection fold {SNAPSHOTS[validation_index]['cutoff']}", flush=True)
        for budget in TT_BUDGETS:
            print(f"  TT+text K={budget}", flush=True)
            selection_results[budget].append(
                run_arm(
                    snapshots,
                    relevance,
                    validation_index,
                    budget,
                    article_ids,
                )
            )
    mean_map_by_budget = {
        budget: float(
            np.mean(
                [
                    result["ranker"]["map_at_12"]
                    for result in results
                ]
            )
        )
        for budget, results in selection_results.items()
    }
    best_mean = max(mean_map_by_budget.values())
    selected_budget = min(
        budget
        for budget, mean_map in mean_map_by_budget.items()
        if mean_map >= best_mean - MAP_TOLERANCE
    )
    print(f"\nFrozen TT+text K={selected_budget}", flush=True)
    final_result = run_arm(
        snapshots,
        relevance,
        FINAL_FOLD,
        selected_budget,
        article_ids,
    )
    fixed_k_report = json.loads(
        (
            OUTPUT_DIR
            / f"two_tower_text_ranking_{training_customers}_{final_customers}.json"
        ).read_text(encoding="utf-8")
    )
    metadata_baseline = fixed_k_report["arms"]["metadata"]["final"]
    report = {
        "contract": {
            "budgets": TT_BUDGETS,
            "selection_folds": SELECTION_FOLDS,
            "final_fold": FINAL_FOLD,
            "selection_rule": (
                "smallest K within 0.0002 MAP@12 of the best earlier-fold mean"
            ),
            "final_week_evaluations_after_selection": 1,
        },
        "selection_results": {
            str(budget): results
            for budget, results in selection_results.items()
        },
        "mean_map_at_12_by_budget": {
            str(budget): value
            for budget, value in mean_map_by_budget.items()
        },
        "selected_budget": selected_budget,
        "final_selected_result": final_result,
        "metadata_tt_k50_baseline": metadata_baseline,
        "final_delta_vs_metadata_tt_k50": {
            "candidate_recall": (
                final_result["candidate"]["candidate_recall"]
                - metadata_baseline["candidate"]["candidate_recall"]
            ),
            "oracle_map_at_12": (
                final_result["candidate"]["oracle_map_at_12"]
                - metadata_baseline["candidate"]["oracle_map_at_12"]
            ),
            "lambda_rank_map_at_12": (
                final_result["ranker"]["map_at_12"]
                - metadata_baseline["ranker"]["map_at_12"]
            ),
        },
    }
    path = OUTPUT_DIR / (
        f"two_tower_text_budget_selection_{training_customers}_{final_customers}.json"
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
