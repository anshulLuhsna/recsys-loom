#!/usr/bin/env python3
"""Weight rare positives only if failure analysis justifies it."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.protocol import (
    DEV_CUSTOMERS,
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
from recsys_loom.ranking.ranker import train_ranker


def weighted_labels(data: dict[str, np.ndarray]) -> np.ndarray:
    names = [str(value) for value in data["feature_names"]]
    popularity = np.nan_to_num(
        data["features"][:, names.index("item_purchases_30d")],
        nan=0.0,
    )
    weights = np.ones(len(data["labels"]), dtype=np.float32)
    positive = data["labels"] > 0
    weights[positive] = 1.0 / np.sqrt(np.maximum(popularity[positive], 1.0))
    weights = np.clip(weights, 0.25, 4.0)
    return data["labels"].astype(np.float32) * weights


def run_arm(train_snapshots, validation, relevance, article_ids, weighted: bool):
    train = combine_training(train_snapshots)
    labels = weighted_labels(train) if weighted else train["labels"]
    model = train_ranker(
        train["features"],
        labels,
        train["groups"],
        [str(value) for value in train["feature_names"]],
        verbose=0,
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

    arms = {"unweighted": False, "inv_sqrt_popularity": True}
    summary = {}
    for name, weighted in arms.items():
        folds = []
        for validation_index in SELECTION_FOLDS:
            result = run_arm(
                snapshots[:validation_index],
                snapshots[validation_index],
                relevance[validation_index],
                article_ids,
                weighted,
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
    selected = "unweighted"
    if (
        summary["inv_sqrt_popularity"]["mean_map_at_12"]
        > summary["unweighted"]["mean_map_at_12"] + 0.0002
    ):
        selected = "inv_sqrt_popularity"
    report = {
        "hypothesis": (
            "Down-weighting head positives helps long-tail ranking without "
            "hurting overall MAP@12."
        ),
        "arms": summary,
        "selected": selected,
        "runtime_seconds": time.monotonic() - started,
    }
    path = OVERNIGHT_DIR / "longtail_weights.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "means": {name: arm["mean_map_at_12"] for name, arm in summary.items()},
        "path": str(path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
