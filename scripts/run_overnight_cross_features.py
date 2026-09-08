#!/usr/bin/env python3
"""Test a small set of explicit ranking interaction features."""

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


def add_crosses(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = [str(value) for value in data["feature_names"]]
    index = {name: position for position, name in enumerate(names)}
    features = data["features"]
    als = np.nan_to_num(features[:, index["score_als"]], nan=0.0)
    tt = np.nan_to_num(features[:, index["score_two_tower"]], nan=0.0)
    sources = features[:, index["num_sources"]]
    popularity = np.nan_to_num(features[:, index["item_purchases_30d"]], nan=0.0)
    repeat = np.nan_to_num(features[:, index["score_repeat_purchase"]], nan=0.0)
    recency = np.nan_to_num(features[:, index["days_since_bought_item"]], nan=365.0)
    als_rank = np.nan_to_num(features[:, index["rank_als"]], nan=500.0)
    tt_rank = np.nan_to_num(features[:, index["rank_two_tower"]], nan=500.0)
    extras = np.column_stack(
        [
            als * tt,
            tt * sources,
            sources * np.log1p(popularity),
            repeat / (1.0 + np.maximum(recency, 0.0)),
            als_rank - tt_rank,
            np.minimum(als_rank, tt_rank),
        ]
    ).astype(np.float32)
    extra_names = np.asarray(
        [
            "cross_als_tt",
            "cross_tt_sources",
            "cross_sources_logpop",
            "cross_repeat_recency",
            "als_rank_minus_tt_rank",
            "min_als_tt_rank",
        ]
    )
    updated = dict(data)
    updated["features"] = np.concatenate([features, extras], axis=1)
    updated["feature_names"] = np.concatenate([data["feature_names"], extra_names])
    return updated


def mean_map(snapshots, relevance, article_ids, transform) -> dict[str, object]:
    folds = []
    for validation_index in SELECTION_FOLDS:
        train = [transform(snapshot) for snapshot in snapshots[:validation_index]]
        validation = transform(snapshots[validation_index])
        model = fit_ranker(train)
        result = evaluate_model(
            model,
            validation,
            relevance[validation_index],
            article_ids,
        )
        result.pop("scores", None)
        folds.append(
            {
                "cutoff": SNAPSHOTS[validation_index]["cutoff"],
                "ranker": result["ranker"],
            }
        )
    maps = [fold["ranker"]["map_at_12"] for fold in folds]
    return {
        "mean_map_at_12": float(sum(maps) / len(maps)),
        "fold_maps": maps,
        "folds": folds,
    }


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
    baseline = mean_map(snapshots, relevance, article_ids, lambda data: data)
    crossed = mean_map(snapshots, relevance, article_ids, add_crosses)
    selected = "baseline_features"
    if crossed["mean_map_at_12"] > baseline["mean_map_at_12"] + 0.0002:
        selected = "targeted_crosses"
    report = {
        "hypothesis": (
            "A few explicit source/popularity crosses help LambdaRank "
            "beyond trees discovering the same interactions."
        ),
        "baseline_features": baseline,
        "targeted_crosses": crossed,
        "selected": selected,
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "cross_features.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "baseline": baseline["mean_map_at_12"],
        "crosses": crossed["mean_map_at_12"],
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
