#!/usr/bin/env python3
"""Compare CatBoost ranking with the frozen LightGBM LambdaRank baseline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from catboost import CatBoostRanker, Pool
except ImportError:
    CatBoostRanker = None
    Pool = None

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
    fit_ranker,
    load_or_build_snapshot,
    ranking_metrics,
)


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    if CatBoostRanker is None or Pool is None:
        report = {
            "decision": "skipped",
            "reason": "catboost is not installed",
        }
        path = OVERNIGHT_DIR / "catboost_ranker.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

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

    lgbm_folds = []
    cat_folds = []
    for validation_index in SELECTION_FOLDS:
        cutoff = SNAPSHOTS[validation_index]["cutoff"]
        print(f"Fold {cutoff}", flush=True)
        lgbm = fit_ranker(snapshots[:validation_index])
        lgbm_result = evaluate_model(
            lgbm,
            snapshots[validation_index],
            relevance[validation_index],
            article_ids,
        )
        lgbm_result.pop("scores", None)
        lgbm_folds.append({"cutoff": cutoff, "ranker": lgbm_result["ranker"]})

        train = combine_training(snapshots[:validation_index])
        validation = snapshots[validation_index]
        group_id = np.repeat(
            np.arange(len(train["groups"])),
            train["groups"],
        )
        train_pool = Pool(
            data=np.nan_to_num(train["features"], nan=0.0),
            label=train["labels"],
            group_id=group_id,
        )
        valid_features = np.nan_to_num(validation["features"], nan=0.0)
        model = CatBoostRanker(
            loss_function="YetiRank",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False,
        )
        model.fit(train_pool)
        scores = model.predict(valid_features)
        ranker, _ = ranking_metrics(
            np.asarray(scores),
            validation,
            article_ids,
            relevance[validation_index],
        )
        cat_folds.append({"cutoff": cutoff, "ranker": ranker})

    lgbm_maps = [fold["ranker"]["map_at_12"] for fold in lgbm_folds]
    cat_maps = [fold["ranker"]["map_at_12"] for fold in cat_folds]
    lgbm_mean = float(sum(lgbm_maps) / len(lgbm_maps))
    cat_mean = float(sum(cat_maps) / len(cat_maps))
    selected = "lightgbm_lambdarank"
    if cat_mean > lgbm_mean + 0.0002:
        selected = "catboost_yetirank"
    report = {
        "hypothesis": (
            "CatBoost YetiRank generalizes better than LightGBM LambdaRank "
            "on the same candidates and features."
        ),
        "lightgbm": {"mean_map_at_12": lgbm_mean, "folds": lgbm_folds},
        "catboost": {"mean_map_at_12": cat_mean, "folds": cat_folds},
        "selected": selected,
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "catboost_ranker.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "lightgbm": lgbm_mean,
        "catboost": cat_mean,
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
