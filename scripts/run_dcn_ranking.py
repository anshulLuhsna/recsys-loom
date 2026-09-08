#!/usr/bin/env python3
"""R2 experiment: DCN V2 vs MLP baseline, with proper normalization and debugging.

Systematic debugging protocol:
    1. Normalize features (log1p + standardize, fit on train only)
    2. Tiny overfit sanity test (can the model memorize 100 customers?)
    3. Train both MLP and DCN V2 on identical inputs
    4. Save epoch-level loss + val MAP@12 for diagnosis
    5. Hard negative sampling (mix random + high-scoring unpurchased)
    6. Compare against frozen LambdaRank R1 = 0.0256

Same candidate pools, temporal snapshots, and labels as LambdaRank.
"""

from __future__ import annotations

import faulthandler
import gc
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
from recsys_loom.ranking.dcn_v2 import (
    DCNV2,
    FeatureNormalizer,
    SimpleMLPRanker,
    overfit_sanity_test,
    predict_neural,
    train_neural_ranker,
)
from recsys_loom.ranking.features import CATEGORY_FEATURES, build_features
from recsys_loom.ranking.ranker import build_labels, evaluate_ranking
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

MAX_CANDIDATES = 200
POP_CANDIDATES = 100
OUTPUT_DIR = ROOT / "artifacts" / "ranking"


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '6GB'")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT TRY_CAST(t_dat AS DATE) AS transaction_date, customer_id, article_id
        FROM read_csv('{ROOT / "transactions_train.csv"}', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT * FROM read_csv('{ROOT / "articles.csv"}', header=true, all_varchar=true)
    """)


def load_target_labels(con, target_start, target_end):
    rows = con.sql(f"""
        SELECT customer_id, LIST(DISTINCT article_id) AS articles
        FROM transactions
        WHERE transaction_date BETWEEN DATE '{target_start}' AND DATE '{target_end}'
        GROUP BY customer_id
    """).fetchall()
    return {cid: set(articles) for cid, articles in rows}


def generate_candidates(con, customer_ids, train_end):
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
    print(f"pop {time.monotonic()-t:.0f}s", end="", flush=True)

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
    records["content"] = content_candidates(con, customer_ids, train_end, 90, MAX_CANDIDATES)
    print(f" | cont {time.monotonic()-t:.0f}s", end="", flush=True)

    t = time.monotonic()
    records["als"] = als_candidates(con, customer_ids, train_end, k=MAX_CANDIDATES)
    print(f" | als {time.monotonic()-t:.0f}s", flush=True)

    return records


def build_categorical_indices(con, pairs):
    cols = ", ".join(CATEGORY_FEATURES)
    rows = con.sql(f"SELECT article_id, {cols} FROM articles").fetchall()

    vocabs: list[dict[str, int]] = [{} for _ in CATEGORY_FEATURES]
    article_cats: dict[str, list[int]] = {}

    for row in rows:
        aid = row[0]
        indices = []
        for j, _ in enumerate(CATEGORY_FEATURES):
            val = row[j + 1] or ""
            if val and val not in vocabs[j]:
                vocabs[j][val] = len(vocabs[j]) + 1
            indices.append(vocabs[j].get(val, 0))
        article_cats[aid] = indices

    cardinalities = [len(v) for v in vocabs]
    cat_matrix = np.zeros((len(pairs), len(CATEGORY_FEATURES)), dtype=np.int64)
    for i, (_, aid) in enumerate(pairs):
        if aid in article_cats:
            cat_matrix[i] = article_cats[aid]

    return cat_matrix, cardinalities


def process_snapshot(con, snap, customer_sample, label):
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

    print(f"    {len(customer_ids):,} customers, "
          f"{sum(len(v) for v in relevant.values()):,} purchases", flush=True)

    if not customer_ids:
        return None

    print(f"    Candidates: ", end="", flush=True)
    candidate_records = generate_candidates(con, customer_ids, train_end)

    t0 = time.monotonic()
    feature_names, X_num, pairs = build_features(
        con, customer_ids, candidate_records, train_end, MAX_CANDIDATES,
    )
    X_cat, cardinalities = build_categorical_indices(con, pairs)
    print(f"    Features: {len(pairs):,} pairs, {time.monotonic()-t0:.1f}s", flush=True)

    y = build_labels(pairs, relevant)
    pos = int(y.sum())
    print(f"    Labels: {pos:,} pos / {len(y)-pos:,} neg", flush=True)

    return {
        "train_end": train_end,
        "feature_names": feature_names,
        "X_num": X_num,
        "X_cat": X_cat,
        "cardinalities": cardinalities,
        "y": y,
        "pairs": pairs,
        "relevant": relevant,
        "customers": len(customer_ids),
        "positives": pos,
    }


def run_model(
    model_name: str,
    model: Any,
    X_num_train: np.ndarray,
    X_cat_train: np.ndarray,
    y_train: np.ndarray,
    train_pairs: list,
    feature_names: list[str],
    X_num_val: np.ndarray,
    X_cat_val: np.ndarray,
    val_pairs: list,
    val_relevant: dict,
) -> tuple[dict, Any, Any]:
    """Train and evaluate one model. Returns (metrics, log, model)."""
    print(f"\n  Training {model_name}...", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}", flush=True)

    t0 = time.monotonic()
    model, log = train_neural_ranker(
        model,
        X_num_train, X_cat_train, y_train, train_pairs, feature_names,
        X_val_num=X_num_val, X_val_cat=X_cat_val,
        val_pairs=val_pairs, val_relevant=val_relevant,
        epochs=40, batch_size=4096, lr=1e-3, weight_decay=1e-4,
        neg_per_pos=20, hard_neg_fraction=0.5, patience=6,
    )
    train_time = time.monotonic() - t0
    print(f"    Training time: {train_time:.1f}s", flush=True)

    predictions = predict_neural(model, X_num_val, X_cat_val, val_pairs, top_k=12)
    metrics = evaluate_ranking(predictions, val_relevant, k=12)
    print(f"    MAP@12={metrics['map_at_12']:.4f}  "
          f"R@12={metrics['recall_at_12']:.4f}  "
          f"HR@12={metrics['hit_rate_at_12']:.4f}", flush=True)

    return metrics, log, model


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.monotonic()

    customer_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    con = duckdb.connect()
    setup_connection(con)

    print("=" * 60, flush=True)
    print("R2 experiment: DCN V2 vs MLP (debugged)", flush=True)
    print("=" * 60, flush=True)

    print("\nBuilding training snapshots (4 rolling weeks)", flush=True)
    X_num_parts: list[np.ndarray] = []
    X_cat_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    all_train_pairs: list[tuple[str, str]] = []
    feature_names: list[str] = []
    total_pos = 0
    cardinalities: list[int] = []

    for i, snap in enumerate(TRAIN_SNAPSHOTS):
        result = process_snapshot(con, snap, customer_sample, f"train-{i+1}")
        if result:
            X_num_parts.append(result["X_num"])
            X_cat_parts.append(result["X_cat"])
            y_parts.append(result["y"])
            prefix = result["train_end"]
            all_train_pairs.extend(
                (f"{prefix}_{cid}", aid) for cid, aid in result["pairs"]
            )
            if not feature_names:
                feature_names = result["feature_names"]
                cardinalities = result["cardinalities"]
            total_pos += result["positives"]
            del result
            gc.collect()

    X_num_train_raw = np.concatenate(X_num_parts)
    X_cat_train = np.concatenate(X_cat_parts)
    y_train = np.concatenate(y_parts)
    del X_num_parts, X_cat_parts, y_parts
    gc.collect()

    print(f"\n  Combined: {len(all_train_pairs):,} pairs, {total_pos:,} positives", flush=True)

    print("\nBuilding validation snapshot", flush=True)
    valid = process_snapshot(con, VALID_SNAPSHOT, customer_sample, "valid")

    if not cardinalities:
        cardinalities = valid["cardinalities"]
    num_numerical = X_num_train_raw.shape[1]

    print(f"\n{'='*60}", flush=True)
    print("Step 1: Normalizing features", flush=True)
    print(f"{'='*60}\n", flush=True)

    normalizer = FeatureNormalizer()
    normalizer.fit(X_num_train_raw, feature_names)
    X_num_train = normalizer.transform(X_num_train_raw)
    X_num_val = normalizer.transform(valid["X_num"])

    print(f"  Train feature stats after normalization:", flush=True)
    print(f"    mean range: [{X_num_train.mean(axis=0).min():.3f}, "
          f"{X_num_train.mean(axis=0).max():.3f}]", flush=True)
    print(f"    std range:  [{X_num_train.std(axis=0).min():.3f}, "
          f"{X_num_train.std(axis=0).max():.3f}]", flush=True)
    print(f"    NaN count:  {np.isnan(X_num_train).sum()}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Step 2: Overfit sanity test (tiny subset)", flush=True)
    print(f"{'='*60}\n", flush=True)

    test_size = min(50000, len(all_train_pairs))
    test_model = DCNV2(
        num_numerical=num_numerical,
        categorical_cardinalities=cardinalities,
        embedding_dim=8, cross_layers=2, mlp_dims=(64, 32), dropout=0.0,
    )
    can_overfit = overfit_sanity_test(
        test_model,
        X_num_train[:test_size],
        X_cat_train[:test_size],
        y_train[:test_size],
        all_train_pairs[:test_size],
        feature_names,
        n_epochs=50,
        lr=1e-2,
    )
    del test_model

    if not can_overfit:
        print("\n  WARNING: Model cannot overfit tiny data. Check implementation.\n", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Step 3: Train MLP baseline + DCN V2", flush=True)
    print(f"  {num_numerical} numerical features (normalized)", flush=True)
    print(f"  {len(cardinalities)} categorical features, "
          f"embedding_dim=8", flush=True)
    print(f"{'='*60}", flush=True)

    mlp_model = SimpleMLPRanker(
        num_numerical=num_numerical,
        categorical_cardinalities=cardinalities,
        embedding_dim=8, mlp_dims=(64, 32), dropout=0.2,
    )
    mlp_metrics, mlp_log, _ = run_model(
        "MLP baseline", mlp_model,
        X_num_train, X_cat_train, y_train, all_train_pairs, feature_names,
        X_num_val, valid["X_cat"], valid["pairs"], valid["relevant"],
    )

    dcn_model = DCNV2(
        num_numerical=num_numerical,
        categorical_cardinalities=cardinalities,
        embedding_dim=8, cross_layers=2, mlp_dims=(64, 32), dropout=0.2,
    )
    dcn_metrics, dcn_log, _ = run_model(
        "DCN V2", dcn_model,
        X_num_train, X_cat_train, y_train, all_train_pairs, feature_names,
        X_num_val, valid["X_cat"], valid["pairs"], valid["relevant"],
    )

    pop_list = recent_popularity(con, valid["train_end"], 7, 12)
    pop_preds = {cid: global_list_to_article_ids(pop_list) for cid in valid["relevant"]}
    pop_metrics = evaluate_ranking(pop_preds, valid["relevant"], k=12)

    valid_pairs_by_cust: dict[str, list[str]] = {}
    for cid, aid in valid["pairs"]:
        valid_pairs_by_cust.setdefault(cid, []).append(aid)
    oracle_preds: dict[str, list[str]] = {}
    for cid, cands in valid_pairs_by_cust.items():
        rel = valid["relevant"].get(cid, set())
        oracle_preds[cid] = [a for a in cands if a in rel] + [a for a in cands if a not in rel]
    oracle_metrics = evaluate_ranking(oracle_preds, valid["relevant"], k=12)

    print(f"\n{'='*60}")
    print("Final comparison (same candidate pools, same validation)")
    print(f"{'='*60}\n")
    print(f"  {'Ranker':<25}  {'MAP@12':>8}  {'R@12':>8}  {'HR@12':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Popularity':<25}  {pop_metrics['map_at_12']:>8.4f}  "
          f"{pop_metrics['recall_at_12']:>8.4f}  {pop_metrics['hit_rate_at_12']:>8.4f}")
    print(f"  {'LambdaRank R1':<25}  {'0.0256':>8}  {'0.0477':>8}  {'0.1205':>8}")
    print(f"  {'MLP baseline':<25}  {mlp_metrics['map_at_12']:>8.4f}  "
          f"{mlp_metrics['recall_at_12']:>8.4f}  {mlp_metrics['hit_rate_at_12']:>8.4f}")
    print(f"  {'DCN V2':<25}  {dcn_metrics['map_at_12']:>8.4f}  "
          f"{dcn_metrics['recall_at_12']:>8.4f}  {dcn_metrics['hit_rate_at_12']:>8.4f}")
    print(f"  {'Oracle':<25}  {oracle_metrics['map_at_12']:>8.4f}  "
          f"{oracle_metrics['recall_at_12']:>8.4f}  {oracle_metrics['hit_rate_at_12']:>8.4f}")

    report = {
        "experiment": "R2_debugged",
        "training": {
            "snapshots": len(TRAIN_SNAPSHOTS),
            "total_positives": total_pos,
            "total_pairs": len(all_train_pairs),
        },
        "overfit_test_passed": can_overfit,
        "mlp_metrics": mlp_metrics,
        "mlp_log": mlp_log.to_dict(),
        "dcn_metrics": dcn_metrics,
        "dcn_log": dcn_log.to_dict(),
        "oracle_metrics": oracle_metrics,
        "popularity_baseline": pop_metrics,
        "lambdarank_r1": {"map_at_12": 0.0256, "recall_at_12": 0.0477, "hit_rate_at_12": 0.1205},
        "total_time_s": time.monotonic() - total_start,
    }

    (OUTPUT_DIR / "dcn_v2_debugged_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )

    print(f"\nTotal runtime: {time.monotonic() - total_start:.1f}s")
    print(f"Report: {OUTPUT_DIR / 'dcn_v2_debugged_report.json'}")
    con.close()


if __name__ == "__main__":
    main()
