#!/usr/bin/env python3
"""Compare full-candidate LambdaRank with hard-negative downsampling."""

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
    development_specs,
    evaluate_model,
    fit_ranker,
    load_or_build_snapshot,
)
from recsys_loom.overnight.transforms import NEGATIVE_SCHEMES, downsample_snapshot


def run_arm(
    train_snapshots: list[dict[str, np.ndarray]],
    validation: dict[str, np.ndarray],
    relevance: dict[str, set[str]],
    article_ids: list[str],
) -> dict[str, object]:
    model = fit_ranker(train_snapshots)
    result = evaluate_model(model, validation, relevance, article_ids)
    result.pop("scores", None)
    result["training_positives"] = int(
        sum(int(snapshot["labels"].sum()) for snapshot in train_snapshots)
    )
    result["training_rows"] = int(
        sum(len(snapshot["labels"]) for snapshot in train_snapshots)
    )
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
        print(f"Loading snapshot {specification.cutoff}...", flush=True)
        data, snapshot_relevance = load_or_build_snapshot(
            connection, specification, article_to_index
        )
        snapshots.append(apply_baseline_budget(data))
        relevance.append(snapshot_relevance)

    schemes = NEGATIVE_SCHEMES
    report_folds = {name: [] for name in schemes}
    for validation_index in SELECTION_FOLDS:
        cutoff = SNAPSHOTS[validation_index]["cutoff"]
        print(f"\nFold {cutoff}", flush=True)
        full_train = snapshots[:validation_index]
        validation = snapshots[validation_index]
        for name, spec in schemes.items():
            print(f"  {name}...", flush=True)
            if spec is None:
                train = full_train
            else:
                hard, random = spec
                train = [
                    downsample_snapshot(snapshot, hard, random, seed=17 + index)
                    for index, snapshot in enumerate(full_train)
                ]
            result = run_arm(
                train,
                validation,
                relevance[validation_index],
                article_ids,
            )
            report_folds[name].append(
                {
                    "cutoff": cutoff,
                    "ranker": result["ranker"],
                    "training_positives": result["training_positives"],
                    "training_rows": result["training_rows"],
                }
            )

    summary = {}
    for name, folds in report_folds.items():
        maps = [fold["ranker"]["map_at_12"] for fold in folds]
        summary[name] = {
            "mean_map_at_12": float(sum(maps) / len(maps)),
            "fold_maps": maps,
            "folds": folds,
        }
    best = max(summary.values(), key=lambda row: row["mean_map_at_12"])
    selected = "all_candidates"
    if (
        summary["all_candidates"]["mean_map_at_12"]
        >= best["mean_map_at_12"] - 0.0002
    ):
        selected = "all_candidates"
    else:
        selected = max(summary, key=lambda name: summary[name]["mean_map_at_12"])
    report = {
        "hypothesis": (
            "Hard negatives teach finer distinctions than using every "
            "unpurchased candidate in a 1K+ group."
        ),
        "schemes": summary,
        "selected": selected,
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "hard_negatives.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "mean_maps": {
            name: row["mean_map_at_12"] for name, row in summary.items()
        },
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
