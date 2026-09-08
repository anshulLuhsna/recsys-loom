#!/usr/bin/env python3
"""Export two extra earlier weeks for scaled ranking supervision."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.protocol import DEV_CUSTOMERS, EXTRA_TRAINING_SNAPSHOTS
from recsys_loom.two_tower.data import (
    Catalog,
    UserNumericalNormalizer,
    build_catalog,
    build_snapshot,
    load_target_customers,
)
from recsys_loom.two_tower.training import (
    encode_all_items,
    encode_snapshot_users,
    fit_for_fixed_epochs,
    train_two_tower,
)


def _load_script(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_existing_export = _load_script(
    "export_existing_ranking_candidates.py",
    "export_existing_ranking_candidates",
)
_tt_export = _load_script(
    "export_two_tower_ranking_candidates.py",
    "export_two_tower_ranking_candidates",
)
TT_OUTPUT = _tt_export.OUTPUT_DIR
make_model = _tt_export.make_model
setup_tt_connection = _tt_export.setup_connection
sorted_target_customers = _tt_export.sorted_target_customers
write_candidates = _tt_export.write_candidates
export_existing_snapshot = _existing_export.export_snapshot
setup_existing_connection = _existing_export.setup_connection

TT_TRAIN_WEEKS = {
    "2020-08-03": [
        {"cutoff": "2020-07-06", "target_start": "2020-07-07", "target_end": "2020-07-13"},
        {"cutoff": "2020-07-13", "target_start": "2020-07-14", "target_end": "2020-07-20"},
        {"cutoff": "2020-07-20", "target_start": "2020-07-21", "target_end": "2020-07-27"},
        {"cutoff": "2020-07-27", "target_start": "2020-07-28", "target_end": "2020-08-03"},
    ],
    "2020-08-10": [
        {"cutoff": "2020-07-13", "target_start": "2020-07-14", "target_end": "2020-07-20"},
        {"cutoff": "2020-07-20", "target_start": "2020-07-21", "target_end": "2020-07-27"},
        {"cutoff": "2020-07-27", "target_start": "2020-07-28", "target_end": "2020-08-03"},
        {"cutoff": "2020-08-03", "target_start": "2020-08-04", "target_end": "2020-08-10"},
    ],
}


def export_two_tower(
    connection: duckdb.DuckDBPyConnection,
    catalog: Catalog,
    ranking_snapshot: dict[str, str],
    sample_size: int,
) -> dict[str, object]:
    weeks = TT_TRAIN_WEEKS[ranking_snapshot["cutoff"]]
    weekly = []
    for index, specification in enumerate(weeks):
        customer_ids = load_target_customers(
            connection,
            specification["target_start"],
            specification["target_end"],
            sample_size=sample_size,
            random_seed=180 + index,
        )
        weekly.append(
            build_snapshot(
                connection,
                catalog,
                specification["cutoff"],
                specification["target_start"],
                specification["target_end"],
                customer_ids,
                observed_items_only=True,
            )
        )
        print(
            f"  tt-train {specification['cutoff']}: "
            f"{weekly[-1].positive_count} positives",
            flush=True,
        )
    tuning_normalizer = UserNumericalNormalizer.fit(weekly[:3])
    _, log = train_two_tower(
        make_model(catalog),
        weekly[:3],
        weekly[3],
        catalog,
        tuning_normalizer,
        epochs=20,
        patience=4,
        verbose=False,
    )
    selected_epoch = max(log.selected_epoch, 1)
    final_normalizer = UserNumericalNormalizer.fit(weekly)
    model, refit_losses = fit_for_fixed_epochs(
        make_model(catalog),
        weekly,
        catalog,
        final_normalizer,
        epochs=selected_epoch,
    )
    target_ids = sorted_target_customers(
        connection,
        ranking_snapshot["target_start"],
        ranking_snapshot["target_end"],
        sample_size,
    )
    inference = build_snapshot(
        connection,
        catalog,
        ranking_snapshot["cutoff"],
        ranking_snapshot["target_start"],
        ranking_snapshot["target_end"],
        target_ids,
        observed_items_only=True,
    )
    stem = f"{ranking_snapshot['cutoff']}_{sample_size}"
    embeddings_path = TT_OUTPUT / f"embeddings_{stem}.npz"
    ann_results_path = TT_OUTPUT / f"ann_results_{stem}.npz"
    ann_metrics_path = TT_OUTPUT / f"ann_metrics_{stem}.json"
    index_path = TT_OUTPUT / f"index_{stem}.faiss"
    np.savez_compressed(
        embeddings_path,
        item_embeddings=encode_all_items(model, catalog),
        user_embeddings=encode_snapshot_users(
            model, inference, final_normalizer
        ),
        eligible_item_indices=inference.eligible_item_indices,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_two_tower_ann.py"),
            str(embeddings_path),
            str(ann_results_path),
            str(index_path),
            str(ann_metrics_path),
        ],
        check=True,
    )
    ann_results = np.load(ann_results_path)
    candidate_path = TT_OUTPUT / f"candidates_{stem}.tsv.gz"
    write_candidates(
        candidate_path,
        target_ids,
        catalog.article_ids,
        ann_results["scores"],
        ann_results["article_indices"],
        ranking_snapshot["cutoff"],
    )
    return {
        "cutoff": ranking_snapshot["cutoff"],
        "selected_epoch": selected_epoch,
        "refit_losses": refit_losses,
        "candidates": str(candidate_path),
    }


def main() -> None:
    started = time.monotonic()
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEV_CUSTOMERS
    existing_con = duckdb.connect()
    setup_existing_connection(existing_con)
    existing_reports = []
    for snapshot in EXTRA_TRAINING_SNAPSHOTS:
        path = (
            ROOT
            / "artifacts"
            / "ranking"
            / "candidate_cache"
            / f"existing_{snapshot['cutoff']}_{sample_size}.tsv.gz"
        )
        if path.exists():
            print(f"Existing cache present for {snapshot['cutoff']}", flush=True)
            existing_reports.append({"cutoff": snapshot["cutoff"], "cached": True})
            continue
        print(f"Exporting existing sources {snapshot['cutoff']}...", flush=True)
        existing_reports.append(
            export_existing_snapshot(existing_con, snapshot, sample_size)
        )
    existing_con.close()

    tt_con = duckdb.connect()
    setup_tt_connection(tt_con)
    catalog = build_catalog(tt_con)
    TT_OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    tt_reports = []
    for snapshot in EXTRA_TRAINING_SNAPSHOTS:
        candidate_path = TT_OUTPUT / f"candidates_{snapshot['cutoff']}_{sample_size}.tsv.gz"
        if candidate_path.exists():
            print(f"TT cache present for {snapshot['cutoff']}", flush=True)
            tt_reports.append({"cutoff": snapshot["cutoff"], "cached": True})
            continue
        print(f"Exporting two-tower {snapshot['cutoff']}...", flush=True)
        tt_reports.append(export_two_tower(tt_con, catalog, snapshot, sample_size))
    tt_con.close()
    report = {
        "existing": existing_reports,
        "two_tower": tt_reports,
        "runtime_seconds": time.monotonic() - started,
    }
    path = (
        ROOT
        / "artifacts"
        / "overnight"
        / f"extra_snapshot_export_{sample_size}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
