#!/usr/bin/env python3
"""Disciplined LightGBM ranking search on development folds only."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.protocol import (
    DEV_CUSTOMERS,
    MAP_TOLERANCE,
    OVERNIGHT_DIR,
    SELECTION_FOLDS,
    SNAPSHOTS,
    ensure_directories,
)
from recsys_loom.overnight.ranking import (
    apply_baseline_budget,
    development_specs,
    evaluate_model,
    fit_ranker,
    load_or_build_snapshot,
)

TRIALS = [
    {"name": "baseline", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "params_update": {}},
    {"name": "trunc12", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "params_update": {"lambdarank_truncation_level": 12}},
    {"name": "trunc20", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "params_update": {"lambdarank_truncation_level": 20}},
    {"name": "trunc30", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "params_update": {"lambdarank_truncation_level": 30}},
    {"name": "leaves31", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50, "params_update": {}},
    {"name": "leaves127", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 50, "params_update": {}},
    {"name": "min20", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20, "params_update": {}},
    {"name": "min100", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100, "params_update": {}},
    {"name": "lr03_n500", "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 50, "params_update": {}},
    {"name": "lr10_n200", "n_estimators": 200, "learning_rate": 0.10, "num_leaves": 63, "min_child_samples": 50, "params_update": {}},
    {"name": "xendcg", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "params_update": {"objective": "rank_xendcg"}},
    {"name": "reg_l2", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "params_update": {"lambda_l2": 1.0}},
]


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    connection = connect()
    article_ids = catalog_article_ids(connection)
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    snapshots = []
    relevance = []
    for specification in development_specs(DEV_CUSTOMERS):
        print(f"Loading snapshot {specification.cutoff}...", flush=True)
        data, snapshot_relevance = load_or_build_snapshot(
            connection, specification, article_to_index
        )
        snapshots.append(apply_baseline_budget(data))
        relevance.append(snapshot_relevance)

    trial_rows = []
    for trial in TRIALS:
        print(f"\nTrial {trial['name']}...", flush=True)
        fold_maps = []
        fold_details = []
        trial_started = time.monotonic()
        try:
            for validation_index in SELECTION_FOLDS:
                model = fit_ranker(
                    snapshots[:validation_index],
                    n_estimators=trial["n_estimators"],
                    learning_rate=trial["learning_rate"],
                    num_leaves=trial["num_leaves"],
                    min_child_samples=trial["min_child_samples"],
                    params_update=trial["params_update"] or None,
                )
                result = evaluate_model(
                    model,
                    snapshots[validation_index],
                    relevance[validation_index],
                    article_ids,
                )
                result.pop("scores", None)
                fold_maps.append(result["ranker"]["map_at_12"])
                fold_details.append(
                    {
                        "cutoff": SNAPSHOTS[validation_index]["cutoff"],
                        "ranker": result["ranker"],
                    }
                )
        except Exception as exc:  # pragma: no cover - trial-level prune
            trial_rows.append(
                {
                    "name": trial["name"],
                    "params": trial,
                    "error": str(exc),
                    "decision": "reject",
                }
            )
            print(f"  failed: {exc}", flush=True)
            continue
        mean_map = float(sum(fold_maps) / len(fold_maps))
        std_map = float(
            (sum((value - mean_map) ** 2 for value in fold_maps) / len(fold_maps))
            ** 0.5
        )
        trial_rows.append(
            {
                "name": trial["name"],
                "params": trial,
                "fold_maps": fold_maps,
                "mean_map_at_12": mean_map,
                "std_map_at_12": std_map,
                "folds": fold_details,
                "runtime_seconds": time.monotonic() - trial_started,
            }
        )
        print(f"  mean MAP@12={mean_map:.5f} std={std_map:.5f}", flush=True)

    successful = [row for row in trial_rows if "mean_map_at_12" in row]
    best_mean = max(row["mean_map_at_12"] for row in successful)
    chosen = next(
        row
        for row in successful
        if row["mean_map_at_12"] >= best_mean - MAP_TOLERANCE
        and row["name"] == "baseline"
    ) if any(
        row["name"] == "baseline" and row["mean_map_at_12"] >= best_mean - MAP_TOLERANCE
        for row in successful
    ) else min(
        (row for row in successful if row["mean_map_at_12"] >= best_mean - MAP_TOLERANCE),
        key=lambda row: (
            0 if row["name"] == "baseline" else 1,
            -row["mean_map_at_12"],
            row["params"]["num_leaves"],
            row["params"]["n_estimators"],
        ),
    )
    report = {
        "selection_rule": (
            "simplest trial within 0.0002 MAP of the best development mean"
        ),
        "trials": trial_rows,
        "best_mean_map_at_12": best_mean,
        "selected": chosen["name"],
        "selected_mean_map_at_12": chosen["mean_map_at_12"],
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "lgbm_tune.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": report["selected"],
        "selected_mean_map_at_12": report["selected_mean_map_at_12"],
        "best_mean_map_at_12": best_mean,
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
