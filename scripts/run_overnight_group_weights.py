#!/usr/bin/env python3
"""Test whether large groups or active users dominate LambdaRank."""

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
    combine_training,
    development_specs,
    evaluate_model,
    load_or_build_snapshot,
)
from recsys_loom.overnight.selection import selected_lightgbm_kwargs
from recsys_loom.overnight.transforms import row_weights_for_scheme
from recsys_loom.ranking.ranker import train_ranker


def run_arm(train_snapshots, validation, relevance, article_ids, scheme: str):
    train = combine_training(train_snapshots)
    model = train_ranker(
        train["features"],
        train["labels"],
        train["groups"],
        [str(value) for value in train["feature_names"]],
        verbose=0,
        weight=row_weights_for_scheme(train, scheme),
        **selected_lightgbm_kwargs(),
    )
    result = evaluate_model(model, validation, relevance, article_ids)
    result.pop("scores", None)
    return result


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
        data, snapshot_relevance = load_or_build_snapshot(
            connection, specification, article_to_index
        )
        snapshots.append(apply_baseline_budget(data))
        relevance.append(snapshot_relevance)

    arms = ["unweighted", "inv_sqrt_group_size", "inv_sqrt_history"]
    summary = {}
    for name in arms:
        folds = []
        for validation_index in SELECTION_FOLDS:
            result = run_arm(
                snapshots[:validation_index],
                snapshots[validation_index],
                relevance[validation_index],
                article_ids,
                name,
            )
            folds.append(
                {
                    "cutoff": SNAPSHOTS[validation_index]["cutoff"],
                    "ranker": result["ranker"],
                }
            )
        maps = [fold["ranker"]["map_at_12"] for fold in folds]
        summary[name] = {
            "mean_map_at_12": float(sum(maps) / len(maps)),
            "fold_maps": maps,
            "folds": folds,
        }
    best_name = max(summary, key=lambda name: summary[name]["mean_map_at_12"])
    best_mean = summary[best_name]["mean_map_at_12"]
    selected = "unweighted"
    if best_mean > summary["unweighted"]["mean_map_at_12"] + MAP_TOLERANCE:
        selected = best_name
    report = {
        "hypothesis": (
            "Down-weighting large candidate groups or historically active "
            "users stops them from dominating LambdaRank."
        ),
        "arms": summary,
        "selected": selected,
        "runtime_seconds": time.monotonic() - started,
    }
    path = OVERNIGHT_DIR / "group_weights.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "means": {name: arm["mean_map_at_12"] for name, arm in summary.items()},
        "path": str(path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
