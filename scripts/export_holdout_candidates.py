#!/usr/bin/env python3
"""Export reserved holdout customers in batches for the final evaluation."""

from __future__ import annotations

import csv
import gzip
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

from recsys_loom.overnight.protocol import OVERNIGHT_DIR, ensure_directories
from recsys_loom.sources.als import als_candidates
from recsys_loom.sources.cooccurrence import cooccurrence_candidates
from recsys_loom.sources.content import content_candidates
from recsys_loom.sources.popularity import recent_popularity
from recsys_loom.sources.repeat_purchase import repeat_purchase_candidates
from recsys_loom.two_tower.data import (
    UserNumericalNormalizer,
    build_catalog,
    build_snapshot,
)
from recsys_loom.two_tower.model import TwoTowerModel
from recsys_loom.two_tower.training import encode_all_items, encode_snapshot_users

CUTOFF = "2020-09-15"
TARGET_START = "2020-09-16"
TARGET_END = "2020-09-22"
BATCH_SIZE = 2000
OUTPUT_DIR = OVERNIGHT_DIR / "holdout"
MODEL_PATH = (
    ROOT / "artifacts" / "two_tower" / "ranking_snapshots" / "model_2020-09-15_2000.pt"
)
MAX_CANDIDATES = 500
POP_K = 100
TT_K = 50


def setup_connection(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '6GB'")
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT TRY_CAST(t_dat AS DATE) AS transaction_date, customer_id, article_id,
               TRY_CAST(price AS DOUBLE) AS price,
               TRY_CAST(sales_channel_id AS INTEGER) AS sales_channel_id
        FROM read_csv('{ROOT / "transactions_train.csv"}', header=true, all_varchar=true)
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT * FROM read_csv('{ROOT / "articles.csv"}', header=true, all_varchar=true)
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW customers AS
        SELECT * FROM read_csv('{ROOT / "customers.csv"}', header=true, all_varchar=true)
    """)


def load_holdout_ids() -> list[str]:
    return [
        line.strip()
        for line in (OVERNIGHT_DIR / "holdout_customer_ids.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def write_existing(path: Path, customer_ids: list[str], connection) -> dict[str, int]:
    popularity = recent_popularity(connection, CUTOFF, 7, POP_K)
    counts = {"recent_7d_pop": len(customer_ids) * len(popularity)}
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["source_name", "customer_id", "article_id", "rank", "score", "feature_cutoff"]
        )
        for customer_id in customer_ids:
            for rank, (article_id, score) in enumerate(popularity, start=1):
                writer.writerow(
                    ["recent_7d_pop", customer_id, article_id, rank, score, CUTOFF]
                )
        generators = [
            ("repeat_purchase", lambda: repeat_purchase_candidates(
                connection, customer_ids, CUTOFF, 30.0, MAX_CANDIDATES
            )),
            ("cooccurrence", lambda: cooccurrence_candidates(
                connection, customer_ids, CUTOFF, 90, 50, MAX_CANDIDATES, 5
            )),
            ("content", lambda: content_candidates(
                connection, customer_ids, CUTOFF, 90, MAX_CANDIDATES
            )),
            ("als", lambda: als_candidates(
                connection, customer_ids, CUTOFF, k=MAX_CANDIDATES
            )),
        ]
        for source_name, generator in generators:
            started = time.monotonic()
            records = generator()
            for record in records:
                writer.writerow(
                    [
                        record.source_name,
                        record.customer_id,
                        record.article_id,
                        record.source_rank,
                        record.source_score,
                        CUTOFF,
                    ]
                )
            counts[source_name] = len(records)
            print(f"  {source_name}: {len(records):,} in {time.monotonic()-started:.1f}s", flush=True)
    return counts


def write_two_tower(path: Path, customer_ids: list[str], connection) -> None:
    catalog = build_catalog(connection)
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = TwoTowerModel(
        article_cardinalities=catalog.article_cardinalities,
        customer_cardinalities=catalog.customer_cardinalities,
        numerical_dim=len(payload["normalizer_mean"]),
        embedding_dim=64,
        categorical_embedding_dim=8,
        customer_embedding_dim=4,
        hidden_dim=128,
        dropout=0.1,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalizer = UserNumericalNormalizer(
        mean=np.asarray(payload["normalizer_mean"], dtype=np.float32),
        std=np.asarray(payload["normalizer_std"], dtype=np.float32),
    )
    snapshot = build_snapshot(
        connection,
        catalog,
        CUTOFF,
        TARGET_START,
        TARGET_END,
        customer_ids,
        observed_items_only=True,
    )
    item_embeddings = encode_all_items(model, catalog)
    user_embeddings = encode_snapshot_users(model, snapshot, normalizer)
    stem = OUTPUT_DIR / "tmp_holdout_tt"
    embeddings_path = Path(str(stem) + "_emb.npz")
    ann_results_path = Path(str(stem) + "_ann.npz")
    index_path = Path(str(stem) + ".faiss")
    metrics_path = Path(str(stem) + "_ann.json")
    np.savez_compressed(
        embeddings_path,
        item_embeddings=item_embeddings,
        user_embeddings=user_embeddings,
        eligible_item_indices=snapshot.eligible_item_indices,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_two_tower_ann.py"),
            str(embeddings_path),
            str(ann_results_path),
            str(index_path),
            str(metrics_path),
        ],
        check=True,
    )
    ann = np.load(ann_results_path)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["customer_id", "article_id", "rank", "score", "feature_cutoff"])
        for customer_index, customer_id in enumerate(customer_ids):
            for rank, (score, item_index) in enumerate(
                zip(ann["scores"][customer_index, :TT_K], ann["article_indices"][customer_index, :TT_K]),
                start=1,
            ):
                if int(item_index) < 0:
                    continue
                writer.writerow(
                    [
                        customer_id,
                        catalog.article_ids[int(item_index)],
                        rank,
                        float(score),
                        CUTOFF,
                    ]
                )


def main() -> None:
    ensure_directories()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    holdout = load_holdout_ids()
    start = batch_index * BATCH_SIZE
    customer_ids = holdout[start : start + BATCH_SIZE]
    if not customer_ids:
        raise SystemExit(f"No holdout customers for batch {batch_index}")
    existing_path = OUTPUT_DIR / f"existing_{CUTOFF}_batch{batch_index}.tsv.gz"
    tt_path = OUTPUT_DIR / f"two_tower_{CUTOFF}_batch{batch_index}.tsv.gz"
    connection = duckdb.connect()
    setup_connection(connection)
    print(f"Holdout batch {batch_index}: {len(customer_ids)} customers", flush=True)
    counts = {}
    if not existing_path.exists():
        counts = write_existing(existing_path, customer_ids, connection)
    if not tt_path.exists():
        write_two_tower(tt_path, customer_ids, connection)
    report = {
        "batch": batch_index,
        "customers": len(customer_ids),
        "existing": str(existing_path),
        "two_tower": str(tt_path),
        "counts": counts,
    }
    (OUTPUT_DIR / f"manifest_batch{batch_index}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
