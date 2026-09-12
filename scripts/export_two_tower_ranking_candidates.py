#!/usr/bin/env python3
"""Export leakage-safe two-tower candidates for every LambdaRank snapshot."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.two_tower.data import (
    USER_NUMERICAL_FIELDS,
    Catalog,
    SnapshotData,
    UserNumericalNormalizer,
    build_catalog,
    build_snapshot,
    load_target_customers,
)
from recsys_loom.two_tower.model import TwoTowerModel
from recsys_loom.two_tower.training import (
    encode_all_items,
    encode_snapshot_users,
    fit_for_fixed_epochs,
    train_two_tower,
)

WEEKLY_SNAPSHOTS = [
    {"cutoff": "2020-07-20", "target_start": "2020-07-21", "target_end": "2020-07-27"},
    {"cutoff": "2020-07-27", "target_start": "2020-07-28", "target_end": "2020-08-03"},
    {"cutoff": "2020-08-03", "target_start": "2020-08-04", "target_end": "2020-08-10"},
    {"cutoff": "2020-08-10", "target_start": "2020-08-11", "target_end": "2020-08-17"},
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
]
RANKING_SNAPSHOTS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
    {"cutoff": "2020-09-15", "target_start": "2020-09-16", "target_end": "2020-09-22"},
]
OUTPUT_DIR = ROOT / "artifacts" / "two_tower" / "ranking_snapshots"
CANDIDATE_K = 300


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '6GB'")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id,
            TRY_CAST(price AS DOUBLE) AS price,
            TRY_CAST(sales_channel_id AS INTEGER) AS sales_channel_id
        FROM read_csv('{ROOT / "transactions_train.csv"}', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT * FROM read_csv('{ROOT / "articles.csv"}', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW customers AS
        SELECT * FROM read_csv('{ROOT / "customers.csv"}', header=true, all_varchar=true)
    """)


def make_model(catalog: Catalog) -> TwoTowerModel:
    return TwoTowerModel(
        article_cardinalities=catalog.article_cardinalities,
        customer_cardinalities=catalog.customer_cardinalities,
        numerical_dim=len(USER_NUMERICAL_FIELDS),
        embedding_dim=64,
        categorical_embedding_dim=8,
        customer_embedding_dim=4,
        hidden_dim=128,
        dropout=0.1,
    )


def sorted_target_customers(
    con: duckdb.DuckDBPyConnection,
    target_start: str,
    target_end: str,
    sample_size: int,
) -> list[str]:
    rows = con.sql(f"""
        SELECT DISTINCT customer_id
        FROM transactions
        WHERE transaction_date BETWEEN DATE '{target_start}' AND DATE '{target_end}'
        ORDER BY customer_id
        LIMIT {int(sample_size)}
    """).fetchall()
    return [row[0] for row in rows]


def write_candidates(
    path: Path,
    customer_ids: list[str],
    article_ids: list[str],
    scores: np.ndarray,
    indices: np.ndarray,
    cutoff: str,
) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["customer_id", "article_id", "rank", "score", "feature_cutoff"])
        for customer_index, customer_id in enumerate(customer_ids):
            for rank, (score, item_index) in enumerate(
                zip(scores[customer_index, :CANDIDATE_K], indices[customer_index, :CANDIDATE_K]),
                start=1,
            ):
                if int(item_index) < 0:
                    continue
                writer.writerow(
                    [
                        customer_id,
                        article_ids[int(item_index)],
                        rank,
                        float(score),
                        cutoff,
                    ]
                )


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    con = duckdb.connect()
    setup_connection(con)
    catalog = build_catalog(con)

    print("Building reusable two-tower training snapshots...", flush=True)
    weekly_data: list[SnapshotData] = []
    for index, specification in enumerate(WEEKLY_SNAPSHOTS):
        customer_ids = load_target_customers(
            con,
            specification["target_start"],
            specification["target_end"],
            sample_size=sample_size,
            random_seed=100 + index,
        )
        snapshot = build_snapshot(
            con,
            catalog,
            specification["cutoff"],
            specification["target_start"],
            specification["target_end"],
            customer_ids,
            observed_items_only=True,
        )
        weekly_data.append(snapshot)
        print(
            f"  {snapshot.cutoff}: {snapshot.positive_count:,} positives",
            flush=True,
        )

    reports: list[dict[str, object]] = []
    for ranking_index, ranking_specification in enumerate(RANKING_SNAPSHOTS):
        print(
            f"\nTraining source for ranking cutoff {ranking_specification['cutoff']}...",
            flush=True,
        )
        training_window = weekly_data[ranking_index : ranking_index + 4]
        tuning_train = training_window[:3]
        tuning_validation = training_window[3]
        tuning_normalizer = UserNumericalNormalizer.fit(tuning_train)
        _, log = train_two_tower(
            make_model(catalog),
            tuning_train,
            tuning_validation,
            catalog,
            tuning_normalizer,
            epochs=20,
            patience=4,
            verbose=False,
        )
        selected_epoch = max(log.selected_epoch, 1)
        final_normalizer = UserNumericalNormalizer.fit(training_window)
        model, refit_losses = fit_for_fixed_epochs(
            make_model(catalog),
            training_window,
            catalog,
            final_normalizer,
            epochs=selected_epoch,
        )

        target_customer_ids = sorted_target_customers(
            con,
            ranking_specification["target_start"],
            ranking_specification["target_end"],
            sample_size,
        )
        inference_snapshot = build_snapshot(
            con,
            catalog,
            ranking_specification["cutoff"],
            ranking_specification["target_start"],
            ranking_specification["target_end"],
            target_customer_ids,
            observed_items_only=True,
        )
        item_embeddings = encode_all_items(model, catalog)
        user_embeddings = encode_snapshot_users(
            model,
            inference_snapshot,
            final_normalizer,
        )

        stem = f"{ranking_specification['cutoff']}_{sample_size}"
        embeddings_path = OUTPUT_DIR / f"embeddings_{stem}.npz"
        ann_results_path = OUTPUT_DIR / f"ann_results_{stem}.npz"
        ann_metrics_path = OUTPUT_DIR / f"ann_metrics_{stem}.json"
        index_path = OUTPUT_DIR / f"index_{stem}.faiss"
        np.savez_compressed(
            embeddings_path,
            item_embeddings=item_embeddings,
            user_embeddings=user_embeddings,
            eligible_item_indices=inference_snapshot.eligible_item_indices,
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
        candidate_path = OUTPUT_DIR / f"candidates_{stem}.tsv.gz"
        write_candidates(
            candidate_path,
            target_customer_ids,
            catalog.article_ids,
            ann_results["scores"],
            ann_results["article_indices"],
            ranking_specification["cutoff"],
        )
        model_path = OUTPUT_DIR / f"model_{stem}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "normalizer_mean": final_normalizer.mean,
                "normalizer_std": final_normalizer.std,
                "selected_epoch": selected_epoch,
            },
            model_path,
        )
        reports.append(
            {
                "ranking_snapshot": ranking_specification,
                "two_tower_training_snapshots": [
                    snapshot.cutoff for snapshot in training_window
                ],
                "selected_epoch": selected_epoch,
                "refit_losses": refit_losses,
                "candidates": str(candidate_path),
                "model": str(model_path),
            }
        )
        print(f"  candidates: {candidate_path}", flush=True)

    report_path = OUTPUT_DIR / f"manifest_{sample_size}.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nManifest: {report_path}")
    con.close()


if __name__ == "__main__":
    main()
