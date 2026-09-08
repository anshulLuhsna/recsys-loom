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


def downsample_group(
    features: np.ndarray,
    labels: np.ndarray,
    article_indices: np.ndarray,
    feature_index: dict[str, int],
    hard: int,
    random: int,
    rng: np.random.Generator,
) -> np.ndarray:
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels == 0)
    if len(negative) == 0:
        return np.ones(len(labels), dtype=np.bool_)
    hardness = (
        features[negative, feature_index["num_sources"]]
        + np.nan_to_num(features[negative, feature_index["score_als"]], nan=0.0)
        + np.nan_to_num(
            features[negative, feature_index["score_two_tower"]], nan=0.0
        )
    )
    hard_take = min(hard, len(negative))
    hard_local = np.argpartition(-hardness, hard_take - 1)[:hard_take]
    remaining = np.setdiff1d(np.arange(len(negative)), hard_local, assume_unique=False)
    random_take = min(random, len(remaining))
    random_local = (
        rng.choice(remaining, size=random_take, replace=False)
        if random_take
        else np.asarray([], dtype=np.int64)
    )
    keep = np.zeros(len(labels), dtype=np.bool_)
    keep[positive] = True
    keep[negative[hard_local]] = True
    keep[negative[random_local]] = True
    return keep


def downsample_snapshot(
    data: dict[str, np.ndarray],
    hard: int,
    random: int,
    seed: int,
) -> dict[str, np.ndarray]:
    feature_index = {
        str(name): index for index, name in enumerate(data["feature_names"])
    }
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(data["labels"]), dtype=np.bool_)
    group_sizes: list[int] = []
    offset = 0
    for group_index, group_size_value in enumerate(data["groups"]):
        group_size = int(group_size_value)
        end = offset + group_size
        group_keep = downsample_group(
            data["features"][offset:end],
            data["labels"][offset:end],
            data["pair_article_indices"][offset:end],
            feature_index,
            hard,
            random,
            rng,
        )
        keep[offset:end] = group_keep
        group_sizes.append(int(group_keep.sum()))
        offset = end
        _ = group_index
    return {
        "features": data["features"][keep],
        "labels": data["labels"][keep],
        "groups": np.asarray(group_sizes, dtype=np.int32),
        "pair_article_indices": data["pair_article_indices"][keep],
        "customer_ids": data["customer_ids"],
        "history_buckets": data["history_buckets"],
        "feature_names": data["feature_names"],
    }


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

    schemes = {
        "all_candidates": None,
        "hard50_rand50": (50, 50),
        "hard100_rand50": (100, 50),
        "hard50_rand150": (50, 150),
    }
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
