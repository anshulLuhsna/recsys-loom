#!/usr/bin/env python3
"""Select an image candidate budget against the frozen TT-K50 baseline."""

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
    SEVEN_SOURCES,
    SnapshotSpec,
    build_feature_data,
    load_candidate_pool,
    load_relevance,
)
from scripts.run_two_tower_budget_sweep import (
    add_efficiency,
    choose_budgets,
    load_history_buckets,
    run_arm,
    setup_connection,
)

IMAGE_BUDGETS = [0, 25, 50, 100, 200, 300]
SNAPSHOTS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
    {"cutoff": "2020-09-15", "target_start": "2020-09-16", "target_end": "2020-09-22"},
]
SELECTION_FOLDS = [2, 3]
FINAL_FOLD = 4
OUTPUT_DIR = ROOT / "artifacts" / "image_retrieval"


def make_specs(
    training_customers: int,
    validation_customers: int,
) -> list[SnapshotSpec]:
    specifications: list[SnapshotSpec] = []
    for index, snapshot in enumerate(SNAPSHOTS):
        sample_size = validation_customers if index == FINAL_FOLD else training_customers
        if index == FINAL_FOLD:
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
                image_candidates_path=(
                    OUTPUT_DIR / f"candidates_{snapshot['cutoff']}_{sample_size}.tsv.gz"
                ),
            )
        )
    return specifications


def main() -> None:
    started = time.monotonic()
    training_customers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    validation_customers = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    specifications = make_specs(training_customers, validation_customers)
    missing = [
        str(path)
        for specification in specifications
        for path in (
            specification.existing_candidates_path,
            specification.two_tower_candidates_path,
            specification.image_candidates_path,
        )
        if path is not None and not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing candidate artifacts:\n" + "\n".join(missing))

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
    max_data: list[dict[str, np.ndarray]] = []
    relevance: list[dict[str, set[str]]] = []
    for specification in specifications:
        print(f"Building image max-K snapshot {specification.cutoff}...", flush=True)
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
            SEVEN_SOURCES,
            article_to_index,
        )
        data["history_buckets"] = np.asarray(
            load_history_buckets(con, pool.customer_ids, specification.cutoff)
        )
        max_data.append(data)
        relevance.append(snapshot_relevance)
        del pool
        gc.collect()

    selection_results: dict[int, list[dict[str, Any]]] = {
        budget: [] for budget in IMAGE_BUDGETS
    }
    for validation_index in SELECTION_FOLDS:
        print(
            f"\nImage selection fold {specifications[validation_index].cutoff}",
            flush=True,
        )
        for budget in IMAGE_BUDGETS:
            print(f"  Image K={budget}...", flush=True)
            selection_results[budget].append(
                run_arm(
                    max_data,
                    relevance,
                    validation_index,
                    budget,
                    article_ids,
                    variable_source="image",
                    source_names=SEVEN_SOURCES,
                    fixed_source_budgets={"two_tower": 50},
                )
            )

    global_budget, history_policy, selection = choose_budgets(selection_results)
    print(f"\nFrozen global image K: {global_budget}", flush=True)
    print(f"Frozen image history policy: {history_policy}", flush=True)

    final_results: dict[int, dict[str, Any]] = {}
    print("\nFinal image budget sweep...", flush=True)
    for budget in IMAGE_BUDGETS:
        print(f"  Image K={budget}...", flush=True)
        final_results[budget] = run_arm(
            max_data,
            relevance,
            FINAL_FOLD,
            budget,
            article_ids,
            variable_source="image",
            source_names=SEVEN_SOURCES,
            fixed_source_budgets={"two_tower": 50},
        )
    add_efficiency(final_results)

    print("\nFinal frozen image history-policy evaluation...", flush=True)
    final_policy_result = run_arm(
        max_data,
        relevance,
        FINAL_FOLD,
        history_policy,
        article_ids,
        variable_source="image",
        source_names=SEVEN_SOURCES,
        fixed_source_budgets={"two_tower": 50},
    )
    report = {
        "contract": {
            "training_customers_per_snapshot": training_customers,
            "final_validation_customers": validation_customers,
            "image_budgets": IMAGE_BUDGETS,
            "frozen_two_tower_budget": 50,
            "selection_fold_cutoffs": [
                specifications[index].cutoff for index in SELECTION_FOLDS
            ],
            "final_cutoff": specifications[FINAL_FOLD].cutoff,
            "latency_definition": "LambdaRank scoring only; milliseconds per customer",
            "selection_rule": (
                "smallest K within 0.0002 MAP@12 of the best earlier-fold mean"
            ),
        },
        "selection_results": {
            str(budget): folds for budget, folds in selection_results.items()
        },
        "selection": selection,
        "frozen_history_policy": history_policy,
        "final_budget_sweep": {
            str(budget): result for budget, result in final_results.items()
        },
        "final_frozen_global_budget": {
            "budget": global_budget,
            "result": final_results[global_budget],
        },
        "final_frozen_history_policy": final_policy_result,
        "runtime_seconds": time.monotonic() - started,
    }
    report_path = OUTPUT_DIR / (
        f"image_budget_sweep_{training_customers}_{validation_customers}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
