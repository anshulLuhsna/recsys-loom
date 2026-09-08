#!/usr/bin/env python3
"""LambdaRank shortlist followed by a small set-aware reranker."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.listwise import (
    fit_reranker,
    predict_shortlist,
    shortlist_groups,
    slate_metrics,
    tiny_overfit_sanity,
)
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
    fit_ranker,
    load_or_build_snapshot,
    ranking_metrics,
)

SHORTLISTS = [25, 50, 100]


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    sanity = tiny_overfit_sanity()
    print(f"tiny overfit: {sanity}", flush=True)
    if not sanity["overfit_ok"]:
        report = {
            "decision": "reject",
            "reason": "tiny-overfit sanity failed",
            "sanity": sanity,
        }
        path = OVERNIGHT_DIR / "listwise_reranker.json"
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

    results = {str(shortlist): [] for shortlist in SHORTLISTS}
    baseline_folds = []
    for validation_index in SELECTION_FOLDS:
        cutoff = SNAPSHOTS[validation_index]["cutoff"]
        print(f"Fold {cutoff}", flush=True)
        model = fit_ranker(snapshots[:validation_index])
        validation = snapshots[validation_index]
        base_scores = np.asarray(
            model.predict(validation["features"]), dtype=np.float32
        )
        base_metrics, _ = ranking_metrics(
            base_scores, validation, article_ids, relevance[validation_index]
        )
        baseline_folds.append({"cutoff": cutoff, "ranker": base_metrics})
        train_groups_by_n: dict[int, list] = {}
        for train_snapshot in snapshots[:validation_index]:
            train_scores = np.asarray(
                model.predict(train_snapshot["features"]), dtype=np.float32
            )
            for shortlist in SHORTLISTS:
                train_groups_by_n.setdefault(shortlist, []).extend(
                    shortlist_groups(train_scores, train_snapshot, shortlist)
                )
        for shortlist in SHORTLISTS:
            reranker, history = fit_reranker(train_groups_by_n[shortlist])
            val_groups = shortlist_groups(base_scores, validation, shortlist)
            predictions = predict_shortlist(reranker, val_groups, article_ids)
            metrics = slate_metrics(predictions, validation, relevance[validation_index])
            results[str(shortlist)].append(
                {
                    "cutoff": cutoff,
                    "ranker": metrics,
                    "final_train_loss": history["loss"][-1],
                    "loss_curve": history["loss"],
                }
            )

    baseline_maps = [fold["ranker"]["map_at_12"] for fold in baseline_folds]
    summary = {
        "lambda_only": {
            "mean_map_at_12": float(sum(baseline_maps) / len(baseline_maps)),
            "folds": baseline_folds,
        }
    }
    for shortlist, folds in results.items():
        maps = [fold["ranker"]["map_at_12"] for fold in folds]
        summary[f"shortlist_{shortlist}"] = {
            "mean_map_at_12": float(sum(maps) / len(maps)),
            "folds": folds,
        }
    best_name = max(summary, key=lambda name: summary[name]["mean_map_at_12"])
    selected = "lambda_only"
    if (
        summary[best_name]["mean_map_at_12"]
        > summary["lambda_only"]["mean_map_at_12"] + 0.0002
    ):
        selected = best_name
    report = {
        "hypothesis": (
            "Relative candidate context on a shortlist improves Top-12 "
            "over pointwise LambdaRank scores."
        ),
        "sanity": sanity,
        "arms": summary,
        "selected": selected,
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "listwise_reranker.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "means": {
            name: arm["mean_map_at_12"] for name, arm in summary.items()
        },
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
