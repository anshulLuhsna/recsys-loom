#!/usr/bin/env python3
"""Compare current supervision with extra earlier ranking weeks."""

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
    OVERNIGHT_DIR,
    SNAPSHOTS,
    ensure_directories,
    selection_fold_indices,
)
from recsys_loom.overnight.ranking import (
    apply_baseline_budget,
    development_specs,
    evaluate_model,
    fit_ranker,
    load_or_build_snapshot,
)


def available(include_extra: bool) -> bool:
    return all(
        specification.existing_candidates_path.exists()
        and specification.two_tower_candidates_path.exists()
        for specification in development_specs(
            DEV_CUSTOMERS,
            include_extra_training=include_extra,
        )
    )


def run_protocol(include_extra: bool, connection, article_ids, article_to_index):
    specifications = development_specs(
        DEV_CUSTOMERS,
        include_extra_training=include_extra,
    )
    snapshots = []
    relevance = []
    for specification in specifications:
        print(f"Loading {specification.cutoff} extra={include_extra}...", flush=True)
        data, snapshot_relevance = load_or_build_snapshot(
            connection, specification, article_to_index
        )
        snapshots.append(apply_baseline_budget(data))
        relevance.append(snapshot_relevance)
    folds = []
    for validation_index in selection_fold_indices(include_extra):
        model = fit_ranker(snapshots[:validation_index])
        result = evaluate_model(
            model,
            snapshots[validation_index],
            relevance[validation_index],
            article_ids,
        )
        result.pop("scores", None)
        folds.append(
            {
                "cutoff": specifications[validation_index].cutoff,
                "training_positives": int(
                    sum(
                        int(snapshot["labels"].sum())
                        for snapshot in snapshots[:validation_index]
                    )
                ),
                "training_groups": int(
                    sum(len(snapshot["groups"]) for snapshot in snapshots[:validation_index])
                ),
                "ranker": result["ranker"],
                "candidate": result["candidate"],
            }
        )
    maps = [fold["ranker"]["map_at_12"] for fold in folds]
    return {
        "include_extra_training": include_extra,
        "mean_map_at_12": float(sum(maps) / len(maps)),
        "fold_maps": maps,
        "folds": folds,
    }


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    if not available(False):
        raise FileNotFoundError("Core 2K candidate caches are required")
    connection = connect()
    article_ids = catalog_article_ids(connection)
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    current = run_protocol(False, connection, article_ids, article_to_index)
    scaled = None
    if available(True):
        scaled = run_protocol(True, connection, article_ids, article_to_index)
    else:
        print("Extra-week caches missing; scaled arm skipped.", flush=True)
    selected = "current"
    if scaled is not None and scaled["mean_map_at_12"] > current["mean_map_at_12"] + 0.0002:
        selected = "scaled"
    report = {
        "hypothesis": (
            "More temporally safe ranking groups improve LambdaRank more "
            "than changing the ranker architecture."
        ),
        "current": current,
        "scaled": scaled,
        "selected": selected,
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "scaled_supervision.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "current_mean": current["mean_map_at_12"],
        "scaled_mean": None if scaled is None else scaled["mean_map_at_12"],
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
