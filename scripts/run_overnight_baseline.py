#!/usr/bin/env python3
"""Freeze BASELINE_RANKER under the overnight development protocol."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.protocol import (
    BASELINE_RANKER,
    CONTAMINATED_FINAL_FOLD,
    DEV_CUSTOMERS,
    OVERNIGHT_DIR,
    PROTOCOL,
    SELECTION_FOLDS,
    SNAPSHOTS,
    TT_BUDGET,
    ensure_directories,
)
from recsys_loom.overnight.ranking import (
    apply_baseline_budget,
    development_specs,
    evaluate_model,
    fit_ranker,
    load_or_build_snapshot,
)


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    connection = connect()
    article_ids = catalog_article_ids(connection)
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    specifications = development_specs(DEV_CUSTOMERS)
    snapshots = []
    relevance = []
    for specification in specifications:
        print(f"Loading snapshot {specification.cutoff}...", flush=True)
        data, snapshot_relevance = load_or_build_snapshot(
            connection,
            specification,
            article_to_index,
        )
        snapshots.append(apply_baseline_budget(data))
        relevance.append(snapshot_relevance)

    fold_results = []
    for validation_index in SELECTION_FOLDS:
        cutoff = SNAPSHOTS[validation_index]["cutoff"]
        print(f"Evaluating development fold {cutoff}...", flush=True)
        model = fit_ranker(snapshots[:validation_index])
        result = evaluate_model(
            model,
            snapshots[validation_index],
            relevance[validation_index],
            article_ids,
        )
        result.pop("scores", None)
        train_positives = int(
            sum(int(snapshot["labels"].sum()) for snapshot in snapshots[:validation_index])
        )
        fold_results.append(
            {
                "cutoff": cutoff,
                "target_start": SNAPSHOTS[validation_index]["target_start"],
                "target_end": SNAPSHOTS[validation_index]["target_end"],
                "training_positives": train_positives,
                "training_groups": int(
                    sum(len(snapshot["groups"]) for snapshot in snapshots[:validation_index])
                ),
                "candidate": result["candidate"],
                "ranker": result["ranker"],
                "ranker_by_history_bucket": result["ranker_by_history_bucket"],
            }
        )

    maps = [fold["ranker"]["map_at_12"] for fold in fold_results]
    baseline = {
        "name": "BASELINE_RANKER",
        "protocol": PROTOCOL,
        "architecture": BASELINE_RANKER,
        "two_tower_k": TT_BUDGET,
        "development_folds": fold_results,
        "development_mean_map_at_12": float(sum(maps) / len(maps)),
        "development_std_map_at_12": float(
            (sum((value - sum(maps) / len(maps)) ** 2 for value in maps) / len(maps))
            ** 0.5
        ),
        "contaminated_final_fold": {
            "cutoff": SNAPSHOTS[CONTAMINATED_FINAL_FOLD]["cutoff"],
            "status": "inspected_reference_only_not_for_selection",
            "customers": int(len(snapshots[CONTAMINATED_FINAL_FOLD]["groups"])),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "baseline_ranker.json"
    output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(json.dumps({
        "development_mean_map_at_12": baseline["development_mean_map_at_12"],
        "folds": [
            {
                "cutoff": fold["cutoff"],
                "map_at_12": fold["ranker"]["map_at_12"],
                "recall_at_12": fold["ranker"]["recall_at_12"],
                "hit_rate_at_12": fold["ranker"]["hit_rate_at_12"],
                "training_positives": fold["training_positives"],
            }
            for fold in fold_results
        ],
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
