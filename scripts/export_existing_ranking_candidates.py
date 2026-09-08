#!/usr/bin/env python3
"""Cache the five existing candidate sources once for controlled ranker A/Bs."""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time
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

SNAPSHOTS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
    {"cutoff": "2020-09-15", "target_start": "2020-09-16", "target_end": "2020-09-22"},
]
MAX_CANDIDATES = 500
POPULARITY_CANDIDATES = 100
OUTPUT_DIR = ROOT / "artifacts" / "ranking" / "candidate_cache"


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '6GB'")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id
        FROM read_csv('{ROOT / "transactions_train.csv"}', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT * FROM read_csv('{ROOT / "articles.csv"}', header=true, all_varchar=true)
    """)


def load_customers(
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


def write_records(
    writer: csv.writer,
    records: list[CandidateRecord],
) -> int:
    for record in records:
        writer.writerow(
            [
                record.source_name,
                record.customer_id,
                record.article_id,
                record.source_rank,
                record.source_score,
                record.feature_cutoff,
            ]
        )
    return len(records)


def export_snapshot(
    con: duckdb.DuckDBPyConnection,
    snapshot: dict[str, str],
    sample_size: int,
) -> dict[str, object]:
    cutoff = snapshot["cutoff"]
    customer_ids = load_customers(
        con,
        snapshot["target_start"],
        snapshot["target_end"],
        sample_size,
    )
    path = OUTPUT_DIR / f"existing_{cutoff}_{sample_size}.tsv.gz"
    counts: dict[str, int] = {}
    timings: dict[str, float] = {}

    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "source_name",
                "customer_id",
                "article_id",
                "rank",
                "score",
                "feature_cutoff",
            ]
        )

        started = time.monotonic()
        popularity = recent_popularity(
            con,
            cutoff,
            7,
            POPULARITY_CANDIDATES,
        )
        for customer_id in customer_ids:
            for rank, (article_id, score) in enumerate(popularity, start=1):
                writer.writerow(
                    [
                        "recent_7d_pop",
                        customer_id,
                        article_id,
                        rank,
                        score,
                        cutoff,
                    ]
                )
        counts["recent_7d_pop"] = len(customer_ids) * len(popularity)
        timings["recent_7d_pop"] = time.monotonic() - started

        generators = [
            (
                "repeat_purchase",
                lambda: repeat_purchase_candidates(
                    con, customer_ids, cutoff, 30.0, MAX_CANDIDATES
                ),
            ),
            (
                "cooccurrence",
                lambda: cooccurrence_candidates(
                    con,
                    customer_ids,
                    cutoff,
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
                    cutoff,
                    90,
                    MAX_CANDIDATES,
                ),
            ),
            (
                "als",
                lambda: als_candidates(
                    con,
                    customer_ids,
                    cutoff,
                    k=MAX_CANDIDATES,
                ),
            ),
        ]
        for source_name, generator in generators:
            started = time.monotonic()
            records = generator()
            counts[source_name] = write_records(writer, records)
            timings[source_name] = time.monotonic() - started
            print(
                f"  {source_name}: {counts[source_name]:,} rows "
                f"in {timings[source_name]:.1f}s",
                flush=True,
            )

    return {
        "snapshot": snapshot,
        "customers": len(customer_ids),
        "path": str(path),
        "counts": counts,
        "timings_seconds": timings,
    }


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    requested_cutoff = sys.argv[2] if len(sys.argv) > 2 else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    setup_connection(con)

    manifests: list[dict[str, object]] = []
    for snapshot in SNAPSHOTS:
        if requested_cutoff and snapshot["cutoff"] != requested_cutoff:
            continue
        print(f"\nExporting cutoff {snapshot['cutoff']}...", flush=True)
        manifests.append(export_snapshot(con, snapshot, sample_size))

    manifest_path = OUTPUT_DIR / f"manifest_{sample_size}.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    con.close()


if __name__ == "__main__":
    main()
