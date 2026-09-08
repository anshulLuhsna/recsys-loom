#!/usr/bin/env python3
"""Train and evaluate a LambdaRank ranker on rolling temporal snapshots.

Temporal design:
    Each training snapshot recreates the entire system as it would
    have existed at that cutoff: separate ALS model, popularity counts,
    co-occurrence pairs, content vectors, repeat-purchase histories,
    and all temporal features.

    Training snapshots (concatenated):
        history <= Aug 17  -> predict Aug 18-24
        history <= Aug 24  -> predict Aug 25-31
        history <= Aug 31  -> predict Sep 1-7
        history <= Sep  7  -> predict Sep 8-14

    Validation (held out):
        history <= Sep 15  -> predict Sep 16-22

    LambdaRank groups are (snapshot_date, customer_id) so the same
    customer at two different dates is treated as two separate ranking
    problems. This lets the model see preferences evolving over time.
"""

from __future__ import annotations

import faulthandler
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

faulthandler.enable()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.candidates import CandidateRecord
from recsys_loom.metrics import average_precision_at_k
from recsys_loom.ranking.features import SOURCE_NAMES, build_features
from recsys_loom.ranking.ranker import (
    build_labels,
    evaluate_ranking,
    predict_and_rank,
    train_ranker,
)
from recsys_loom.sources.als import als_candidates
from recsys_loom.sources.cooccurrence import cooccurrence_candidates
from recsys_loom.sources.content import content_candidates
from recsys_loom.sources.popularity import global_list_to_article_ids, recent_popularity
from recsys_loom.sources.repeat_purchase import repeat_purchase_candidates

TRAIN_SNAPSHOTS = [
    {"train_end": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"train_end": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"train_end": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"train_end": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
]

VALID_SNAPSHOT = {
    "train_end": "2020-09-15",
    "target_start": "2020-09-16",
    "target_end": "2020-09-22",
}

MAX_CANDIDATES = 500
POP_CANDIDATES = 100
OUTPUT_DIR = ROOT / "artifacts" / "ranking"
TWO_TOWER_SOURCE_NAMES = [*SOURCE_NAMES, "two_tower"]


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '6GB'")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id
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


def load_target_labels(
    con: duckdb.DuckDBPyConnection,
    target_start: str,
    target_end: str,
) -> dict[str, set[str]]:
    rows = con.sql(f"""
        SELECT customer_id, LIST(DISTINCT article_id) AS articles
        FROM transactions
        WHERE transaction_date BETWEEN DATE '{target_start}' AND DATE '{target_end}'
        GROUP BY customer_id
    """).fetchall()
    return {cid: set(articles) for cid, articles in rows}


def generate_candidates(
    con: duckdb.DuckDBPyConnection,
    customer_ids: list[str],
    train_end: str,
    two_tower_records: list[CandidateRecord] | None = None,
) -> dict[str, list[CandidateRecord]]:
    """Run all retrieval sources for one temporal snapshot."""
    records: dict[str, list[CandidateRecord]] = {}

    t = time.monotonic()
    pop_list = recent_popularity(con, train_end, 7, POP_CANDIDATES)
    pop_ids = global_list_to_article_ids(pop_list)
    pop_scores = {aid: score for aid, score in pop_list}
    pop_records = []
    for rank, aid in enumerate(pop_ids, 1):
        for cid in customer_ids:
            pop_records.append(CandidateRecord(
                customer_id=cid, article_id=aid,
                source_name="recent_7d_pop", source_rank=rank,
                source_score=pop_scores[aid], feature_cutoff=train_end,
            ))
    records["recent_7d_pop"] = pop_records
    print(f"    pop {time.monotonic()-t:.0f}s", end="", flush=True)

    t = time.monotonic()
    records["repeat_purchase"] = repeat_purchase_candidates(
        con, customer_ids, train_end, 30.0, MAX_CANDIDATES,
    )
    print(f" | rpt {time.monotonic()-t:.0f}s", end="", flush=True)

    t = time.monotonic()
    records["cooccurrence"] = cooccurrence_candidates(
        con, customer_ids, train_end, 90, 50, MAX_CANDIDATES, 5,
    )
    print(f" | cooc {time.monotonic()-t:.0f}s", end="", flush=True)

    t = time.monotonic()
    records["content"] = content_candidates(
        con, customer_ids, train_end, 90, MAX_CANDIDATES,
    )
    print(f" | cont {time.monotonic()-t:.0f}s", end="", flush=True)

    t = time.monotonic()
    records["als"] = als_candidates(
        con, customer_ids, train_end, k=MAX_CANDIDATES,
    )
    print(f" | als {time.monotonic()-t:.0f}s", flush=True)
    if two_tower_records is not None:
        records["two_tower"] = two_tower_records
        print(f"    two_tower {len(two_tower_records):,} cached records", flush=True)

    return records


def load_two_tower_records(
    candidate_dir: Path,
    train_end: str,
    customer_sample: int,
) -> list[CandidateRecord]:
    path = candidate_dir / f"candidates_{train_end}_{customer_sample}.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing two-tower snapshot candidates: {path}")

    records: list[CandidateRecord] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            records.append(
                CandidateRecord(
                    customer_id=row["customer_id"],
                    article_id=row["article_id"],
                    source_name="two_tower",
                    source_rank=int(row["rank"]),
                    source_score=float(row["score"]),
                    model_version="two_tower_metadata_history_64d_v1",
                    feature_cutoff=row["feature_cutoff"],
                )
            )
    return records


def build_snapshot_groups(
    pairs: list[tuple[str, str]],
) -> tuple[list[int], list[str]]:
    """Build group sizes from ordered pairs where each unique customer is one group."""
    groups: list[int] = []
    group_keys: list[str] = []
    current = None
    count = 0
    for cid, _ in pairs:
        if cid != current:
            if current is not None:
                groups.append(count)
                group_keys.append(current)
            current = cid
            count = 0
        count += 1
    if current is not None:
        groups.append(count)
        group_keys.append(current)
    return groups, group_keys


def process_snapshot(
    con: duckdb.DuckDBPyConnection,
    snap: dict[str, str],
    customer_sample: int,
    label: str,
    two_tower_candidate_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Generate candidates, build features, and label for one snapshot."""
    train_end = snap["train_end"]
    target_start = snap["target_start"]
    target_end = snap["target_end"]

    print(f"\n  [{label}] cutoff={train_end}, predict={target_start} to {target_end}",
          flush=True)

    relevant = load_target_labels(con, target_start, target_end)
    customer_ids = sorted(relevant)

    if customer_sample > 0:
        customer_ids = customer_ids[:customer_sample]
        relevant = {c: relevant[c] for c in customer_ids}

    total_purchases = sum(len(v) for v in relevant.values())
    print(f"    {len(customer_ids):,} customers, {total_purchases:,} purchases", flush=True)

    if not customer_ids:
        return None

    two_tower_records = None
    if two_tower_candidate_dir is not None:
        two_tower_records = load_two_tower_records(
            two_tower_candidate_dir,
            train_end,
            customer_sample,
        )

    print(f"    Candidates: ", end="", flush=True)
    candidate_records = generate_candidates(
        con,
        customer_ids,
        train_end,
        two_tower_records=two_tower_records,
    )
    total_recs = sum(len(r) for r in candidate_records.values())
    print(f"    Total: {total_recs:,} records", flush=True)

    print(f"    Features: ", end="", flush=True)
    t0 = time.monotonic()
    feature_names, X, pairs = build_features(
        con,
        customer_ids,
        candidate_records,
        train_end,
        MAX_CANDIDATES,
        source_names=(
            TWO_TOWER_SOURCE_NAMES
            if two_tower_candidate_dir is not None
            else SOURCE_NAMES
        ),
    )
    print(f"{len(pairs):,} pairs, {time.monotonic()-t0:.1f}s", flush=True)

    y = build_labels(pairs, relevant)
    groups, group_keys = build_snapshot_groups(pairs)

    pos = int(y.sum())
    print(f"    Labels: {pos:,} pos / {len(y)-pos:,} neg", flush=True)

    return {
        "train_end": train_end,
        "target_start": target_start,
        "target_end": target_end,
        "feature_names": feature_names,
        "X": X,
        "y": y,
        "pairs": pairs,
        "groups": np.array(groups, dtype=np.int32),
        "group_keys": group_keys,
        "relevant": relevant,
        "customers": len(customer_ids),
        "positives": pos,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.monotonic()

    customer_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    two_tower_candidate_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    con = duckdb.connect()
    setup_connection(con)

    print("=" * 60, flush=True)
    print("Building training snapshots (4 rolling weeks)", flush=True)
    print("=" * 60, flush=True)

    train_results: list[dict[str, Any]] = []
    for i, snap in enumerate(TRAIN_SNAPSHOTS):
        result = process_snapshot(
            con,
            snap,
            customer_sample,
            f"train-{i+1}",
            two_tower_candidate_dir,
        )
        if result:
            train_results.append(result)

    X_train = np.concatenate([r["X"] for r in train_results])
    y_train = np.concatenate([r["y"] for r in train_results])
    groups_train = np.concatenate([r["groups"] for r in train_results])
    feature_names = train_results[0]["feature_names"]
    total_train_pos = sum(r["positives"] for r in train_results)
    total_train_pairs = sum(len(r["pairs"]) for r in train_results)
    total_train_groups = sum(len(r["groups"]) for r in train_results)

    print(f"\n  Combined training: {total_train_pairs:,} pairs, "
          f"{total_train_pos:,} positives, {total_train_groups:,} groups", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Building validation snapshot", flush=True)
    print("=" * 60, flush=True)

    valid = process_snapshot(
        con,
        VALID_SNAPSHOT,
        customer_sample,
        "valid",
        two_tower_candidate_dir,
    )

    print(f"\n{'='*60}", flush=True)
    print("Training LambdaRank ranker", flush=True)
    print("=" * 60, flush=True)

    t0 = time.monotonic()
    model = train_ranker(
        X_train, y_train, groups_train,
        feature_names,
        valid["X"], valid["y"], valid["groups"],
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        verbose=50,
    )
    train_time = time.monotonic() - t0
    print(f"\nTraining: {train_time:.1f}s\n", flush=True)

    print("=" * 60, flush=True)
    print("Evaluation", flush=True)
    print("=" * 60, flush=True)

    predictions = predict_and_rank(model, valid["X"], valid["pairs"], top_k=12)
    metrics = evaluate_ranking(predictions, valid["relevant"], k=12)

    print(f"\n  LambdaRank (multi-week):")
    print(f"    MAP@12:      {metrics['map_at_12']:.4f}")
    print(f"    Recall@12:   {metrics['recall_at_12']:.4f}")
    print(f"    Hit Rate@12: {metrics['hit_rate_at_12']:.4f}")

    pop_list = recent_popularity(con, valid["train_end"], 7, 12)
    pop_baseline = {cid: global_list_to_article_ids(pop_list) for cid in valid["relevant"]}
    pop_metrics = evaluate_ranking(pop_baseline, valid["relevant"], k=12)

    print(f"\n  Popularity baseline:")
    print(f"    MAP@12:      {pop_metrics['map_at_12']:.4f}")
    print(f"    Recall@12:   {pop_metrics['recall_at_12']:.4f}")
    print(f"    Hit Rate@12: {pop_metrics['hit_rate_at_12']:.4f}")

    valid_pairs_by_customer: dict[str, list[str]] = {}
    for cid, aid in valid["pairs"]:
        valid_pairs_by_customer.setdefault(cid, []).append(aid)
    oracle_predictions: dict[str, list[str]] = {}
    for cid, candidates in valid_pairs_by_customer.items():
        relevant_set = valid["relevant"].get(cid, set())
        hits = [a for a in candidates if a in relevant_set]
        non_hits = [a for a in candidates if a not in relevant_set]
        oracle_predictions[cid] = hits + non_hits
    oracle_metrics = evaluate_ranking(oracle_predictions, valid["relevant"], k=12)

    print(f"\n  Oracle (same candidate pool):")
    print(f"    MAP@12:      {oracle_metrics['map_at_12']:.4f}")
    print(f"    Recall@12:   {oracle_metrics['recall_at_12']:.4f}")
    print(f"    Hit Rate@12: {oracle_metrics['hit_rate_at_12']:.4f}")

    ranker_map = metrics["map_at_12"]
    oracle_map = oracle_metrics["map_at_12"]
    pop_map = pop_metrics["map_at_12"]

    print(f"\n{'='*60}")
    print("Diagnostic summary")
    print(f"{'='*60}\n")
    print(f"  Training: {len(TRAIN_SNAPSHOTS)} snapshots, "
          f"{total_train_pos:,} positives, {total_train_groups:,} groups")
    print(f"  Validation: 1 snapshot, {valid['positives']:,} positives\n")
    print(f"  Popularity baseline:   {pop_map:.4f} MAP@12")
    print(f"  LambdaRank multi-week: {ranker_map:.4f} MAP@12")
    print(f"  Oracle (same pool):    {oracle_map:.4f} MAP@12")
    if pop_map > 0:
        print(f"\n  Lift over popularity:  {(ranker_map - pop_map) / pop_map * 100:+.1f}%")
    if oracle_map > 0:
        efficiency = ranker_map / oracle_map
        print(f"  Ranking efficiency:    {efficiency:.1%} (actual / oracle)")
        print(f"  Ranking headroom:      {oracle_map - ranker_map:.4f}")

    customers_with_pos = sum(
        1 for cid in valid["relevant"]
        if any(aid in valid["relevant"][cid] for aid in valid_pairs_by_customer.get(cid, []))
    )
    customers_no_pos = valid["customers"] - customers_with_pos
    print(f"\n  Customers with 0 positives in pool: {customers_no_pos} / {valid['customers']}"
          f" ({100*customers_no_pos/valid['customers']:.1f}%)")
    print(f"  (These inflate LightGBM NDCG by getting score=1.0 for free)")

    print(f"\nFeature importance (top 15 by gain):\n")
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    print(f"  {'Feature':<35}  {'Gain':>10}")
    print(f"  {'-'*35}  {'-'*10}")
    for fname, gain in feat_imp[:15]:
        print(f"  {fname:<35}  {gain:>10.1f}")

    report: dict[str, Any] = {
        "training_snapshots": [
            {
                "train_end": r["train_end"],
                "target": f"{r['target_start']} to {r['target_end']}",
                "customers": r["customers"],
                "positives": r["positives"],
            }
            for r in train_results
        ],
        "validation_snapshot": {
            "train_end": valid["train_end"],
            "target": f"{valid['target_start']} to {valid['target_end']}",
            "customers": valid["customers"],
            "positives": valid["positives"],
        },
        "total_train_positives": total_train_pos,
        "total_train_groups": total_train_groups,
        "ranker_metrics": metrics,
        "oracle_metrics": oracle_metrics,
        "ranking_efficiency": ranker_map / oracle_map if oracle_map > 0 else None,
        "popularity_baseline": pop_metrics,
        "feature_importance": [
            {"feature": f, "gain": float(g)} for f, g in feat_imp
        ],
        "training_time_s": train_time,
        "total_time_s": time.monotonic() - total_start,
        "two_tower_source": two_tower_candidate_dir is not None,
    }

    source_label = "6source" if two_tower_candidate_dir is not None else "5source"
    sample_label = str(customer_sample) if customer_sample > 0 else "all"
    report_name = f"ranking_report_{source_label}_{sample_label}.json"
    model_name = f"lambdarank_model_{source_label}_{sample_label}.txt"
    (OUTPUT_DIR / report_name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    model.save_model(str(OUTPUT_DIR / model_name))

    total_elapsed = time.monotonic() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s")
    print(f"Report: {OUTPUT_DIR / report_name}")
    print(f"Model: {OUTPUT_DIR / model_name}")

    con.close()


if __name__ == "__main__":
    main()
