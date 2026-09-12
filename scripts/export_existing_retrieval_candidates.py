#!/usr/bin/env python3
"""Export the frozen five-source validation candidates for union comparison.

This runs separately from FAISS because ``implicit`` and FAISS have previously
loaded conflicting BLAS runtimes in the same macOS Python process.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.candidates import CandidateRecord
from recsys_loom.sources.als import als_candidates
from recsys_loom.sources.cooccurrence import cooccurrence_candidates
from recsys_loom.sources.content import content_candidates
from recsys_loom.sources.popularity import recent_popularity
from recsys_loom.sources.repeat_purchase import repeat_purchase_candidates
from recsys_loom.two_tower.data import load_target_customers

TRAIN_END = "2020-09-15"
VALIDATION_START = "2020-09-16"
VALIDATION_END = "2020-09-22"
MAX_CANDIDATES = 500
OUTPUT_DIR = ROOT / "artifacts" / "two_tower"


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
        FROM read_csv(
            '{ROOT / "transactions_train.csv"}',
            header = true,
            all_varchar = true
        )
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT *
        FROM read_csv(
            '{ROOT / "articles.csv"}',
            header = true,
            all_varchar = true
        )
    """)


def write_records(
    writer: csv.writer,
    records: Iterable[CandidateRecord],
) -> int:
    count = 0
    for record in records:
        writer.writerow(
            [
                record.source_name,
                record.customer_id,
                record.article_id,
                record.source_rank,
                record.source_score,
            ]
        )
        count += 1
    return count


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"existing_candidates_{sample_size}.tsv.gz"
    manifest_path = OUTPUT_DIR / f"existing_candidates_{sample_size}.json"

    con = duckdb.connect()
    setup_connection(con)
    customer_ids = load_target_customers(
        con,
        VALIDATION_START,
        VALIDATION_END,
        sample_size=sample_size,
    )

    counts: dict[str, int] = {}
    timings: dict[str, float] = {}
    with gzip.open(output_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source_name", "customer_id", "article_id", "rank", "score"])

        started = time.monotonic()
        popularity = recent_popularity(con, TRAIN_END, 7, MAX_CANDIDATES)
        for customer_id in customer_ids:
            for rank, (article_id, score) in enumerate(popularity, start=1):
                writer.writerow(
                    ["recent_7d_pop", customer_id, article_id, rank, score]
                )
        counts["recent_7d_pop"] = len(customer_ids) * len(popularity)
        timings["recent_7d_pop"] = time.monotonic() - started

        generators = [
            (
                "repeat_purchase",
                lambda: repeat_purchase_candidates(
                    con,
                    customer_ids,
                    TRAIN_END,
                    30.0,
                    MAX_CANDIDATES,
                ),
            ),
            (
                "cooccurrence",
                lambda: cooccurrence_candidates(
                    con,
                    customer_ids,
                    TRAIN_END,
                    90,
                    50,
                    MAX_CANDIDATES,
                    5,
                ),
            ),
            (
                "content",
                lambda: content_candidates(
                    con,
                    customer_ids,
                    TRAIN_END,
                    90,
                    MAX_CANDIDATES,
                ),
            ),
            (
                "als",
                lambda: als_candidates(
                    con,
                    customer_ids,
                    TRAIN_END,
                    k=MAX_CANDIDATES,
                ),
            ),
        ]

        for source_name, generate in generators:
            started = time.monotonic()
            records = generate()
            counts[source_name] = write_records(writer, records)
            timings[source_name] = time.monotonic() - started
            print(
                f"{source_name}: {counts[source_name]:,} records "
                f"in {timings[source_name]:.1f}s",
                flush=True,
            )

    manifest = {
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "sample_size": len(customer_ids),
        "max_candidates": MAX_CANDIDATES,
        "counts": counts,
        "timings_seconds": timings,
        "output": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Candidates: {output_path}")
    print(f"Manifest: {manifest_path}")
    con.close()


if __name__ == "__main__":
    main()
