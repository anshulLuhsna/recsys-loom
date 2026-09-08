#!/usr/bin/env python3
"""Evaluate the frozen BEST_SYSTEM once on reserved unused customers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.best_system import load_spec
from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.protocol import (
    DEV_CUSTOMERS,
    OVERNIGHT_DIR,
    ensure_directories,
)
from recsys_loom.overnight.ranking import (
    apply_baseline_budget,
    development_specs,
    load_or_build_snapshot,
    segment_report,
)
from recsys_loom.overnight.serve import (
    fit_from_spec,
    predict_scores,
    transform_snapshot,
    uses_extra_training,
)
from recsys_loom.ranking.cached_data import SnapshotSpec

HOLDOUT_DIR = OVERNIGHT_DIR / "holdout"


def holdout_specs() -> list[SnapshotSpec]:
    specifications = []
    for batch in range(4):
        existing = HOLDOUT_DIR / f"existing_2020-09-15_batch{batch}.tsv.gz"
        two_tower = HOLDOUT_DIR / f"two_tower_2020-09-15_batch{batch}.tsv.gz"
        if not existing.exists() or not two_tower.exists():
            continue
        specifications.append(
            SnapshotSpec(
                cutoff="2020-09-15",
                target_start="2020-09-16",
                target_end="2020-09-22",
                customer_count=2000,
                existing_candidates_path=existing,
                two_tower_candidates_path=two_tower,
            )
        )
    return specifications


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    specs = holdout_specs()
    if not specs:
        report = {
            "status": "blocked",
            "reason": "holdout candidate caches are not built yet",
            "command": "python scripts/export_holdout_candidates.py 0",
        }
        path = OVERNIGHT_DIR / "holdout_evaluation.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    spec = load_spec()
    if spec.get("status") != "frozen_for_holdout":
        raise SystemExit("Freeze BEST_SYSTEM from development reports before holdout")
    connection = connect()
    article_ids = catalog_article_ids(connection)
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    train_snapshots = []
    for specification in development_specs(
        DEV_CUSTOMERS,
        include_extra_training=uses_extra_training(spec),
    ):
        if specification.cutoff == "2020-09-15":
            continue
        data, _relevance = load_or_build_snapshot(
            connection, specification, article_to_index
        )
        train_snapshots.append(apply_baseline_budget(data))
    model = fit_from_spec(train_snapshots, spec)
    batch_results = []
    for specification in specs:
        data, relevance = load_or_build_snapshot(
            connection,
            specification,
            article_to_index,
            cache_tag=specification.existing_candidates_path.stem,
        )
        validation = transform_snapshot(
            apply_baseline_budget(data),
            spec,
            training=False,
        )
        scores = predict_scores(model, validation)
        segments = segment_report(scores, validation, article_ids, relevance)
        batch_results.append(
            {
                "path": str(specification.existing_candidates_path.name),
                "ranker": segments["overall"],
                "candidate": segments["candidate"],
                "segments": {
                    "by_history_bucket": segments["by_history_bucket"],
                    "retrieved_positives": segments["retrieved_positives"],
                },
            }
        )
    maps = [batch["ranker"]["map_at_12"] for batch in batch_results]
    customers = sum(int(batch["ranker"]["customers"]) for batch in batch_results)
    recalls = [batch["ranker"]["recall_at_12"] for batch in batch_results]
    hits = [batch["ranker"]["hit_rate_at_12"] for batch in batch_results]
    report = {
        "evaluation_type": "customer_holdout_final_evaluation",
        "best_system": spec,
        "customers": customers,
        "mean_map_at_12": float(sum(maps) / len(maps)),
        "mean_recall_at_12": float(sum(recalls) / len(recalls)),
        "mean_hit_rate_at_12": float(sum(hits) / len(hits)),
        "batch_maps": maps,
        "batches": batch_results,
        "runtime_seconds": time.monotonic() - started,
        "warning": (
            "This is a customer holdout on 2020-09-16..22, not a pristine "
            "unseen temporal window."
        ),
    }
    path = OVERNIGHT_DIR / "holdout_evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "customers": customers,
        "mean_map_at_12": report["mean_map_at_12"],
        "path": str(path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
