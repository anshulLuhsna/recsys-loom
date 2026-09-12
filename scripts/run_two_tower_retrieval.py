#!/usr/bin/env python3
"""Train and evaluate a feature-built two-tower as a sixth retrieval source."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.candidates import CandidateRecord
from recsys_loom.metrics import average_precision_at_k
from recsys_loom.retrieval_eval import (
    incremental_union_recall,
    pairwise_overlap,
    retrieval_report,
)
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
    similarity_diagnostic,
    tiny_overfit_check,
    train_two_tower,
)

TRAIN_SNAPSHOTS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
]
FINAL_SNAPSHOT = {
    "cutoff": "2020-09-15",
    "target_start": "2020-09-16",
    "target_end": "2020-09-22",
}
K_VALUES = (100, 300, 500)
OUTPUT_DIR = ROOT / "artifacts" / "two_tower"
SOURCE_ORDER = [
    "recent_7d_pop",
    "repeat_purchase",
    "cooccurrence",
    "als",
    "content",
]


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
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW customers AS
        SELECT *
        FROM read_csv(
            '{ROOT / "customers.csv"}',
            header = true,
            all_varchar = true
        )
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


def build_experiment_snapshot(
    con: duckdb.DuckDBPyConnection,
    catalog: Catalog,
    specification: dict[str, str],
    sample_size: int,
    random_seed: int,
) -> SnapshotData:
    customer_ids = load_target_customers(
        con,
        specification["target_start"],
        specification["target_end"],
        sample_size=sample_size,
        random_seed=random_seed,
    )
    snapshot = build_snapshot(
        con=con,
        catalog=catalog,
        cutoff=specification["cutoff"],
        target_start=specification["target_start"],
        target_end=specification["target_end"],
        customer_ids=customer_ids,
        max_history=50,
        history_half_life_days=45.0,
        observed_items_only=True,
    )
    print(
        f"  {snapshot.cutoff}: customers={len(snapshot.customer_ids):,}, "
        f"eligible_positives={snapshot.positive_count:,}, "
        f"zero_history={snapshot.sparse_history_count:,}",
        flush=True,
    )
    return snapshot


def load_existing_candidates(
    path: Path,
) -> dict[str, dict[str, list[str]]]:
    candidates: dict[str, dict[str, list[str]]] = {
        source: {} for source in SOURCE_ORDER
    }
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_name = row["source_name"]
            if source_name not in candidates:
                continue
            candidates[source_name].setdefault(row["customer_id"], []).append(
                row["article_id"]
            )
    return candidates


def records_to_candidates(
    records: list[CandidateRecord],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for record in records:
        result.setdefault(record.customer_id, []).append(record.article_id)
    return result


def write_two_tower_candidates(
    path: Path,
    records: list[CandidateRecord],
) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "customer_id",
                "article_id",
                "two_tower_score",
                "two_tower_rank",
                "source_two_tower",
                "feature_cutoff",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.customer_id,
                    record.article_id,
                    record.source_score,
                    record.source_rank,
                    1,
                    record.feature_cutoff,
                ]
            )


def build_candidate_records(
    customer_ids: list[str],
    article_ids: list[str],
    scores: np.ndarray,
    article_indices: np.ndarray,
    cutoff: str,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for customer_index, customer_id in enumerate(customer_ids):
        for rank, (score, article_index) in enumerate(
            zip(scores[customer_index], article_indices[customer_index]),
            start=1,
        ):
            if int(article_index) < 0:
                continue
            records.append(
                CandidateRecord(
                    customer_id=customer_id,
                    article_id=article_ids[int(article_index)],
                    source_name="two_tower",
                    source_rank=rank,
                    source_score=float(score),
                    model_version="two_tower_metadata_history_64d_v1",
                    feature_cutoff=cutoff,
                )
            )
    return records


def oracle_map(
    relevant_by_customer: dict[str, set[str]],
    candidates_by_source: dict[str, dict[str, list[str]]],
    k_per_source: int,
) -> float:
    scores: list[float] = []
    for customer_id, relevant in relevant_by_customer.items():
        union: list[str] = []
        seen: set[str] = set()
        for candidates in candidates_by_source.values():
            for article_id in candidates.get(customer_id, [])[:k_per_source]:
                if article_id not in seen:
                    seen.add(article_id)
                    union.append(article_id)
        ordered = [item for item in union if item in relevant]
        ordered.extend(item for item in union if item not in relevant)
        scores.append(average_precision_at_k(relevant, ordered, 12))
    return float(np.mean(scores)) if scores else 0.0


def norm_summary(embeddings: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
        "std": float(norms.std()),
    }


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    baseline_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else OUTPUT_DIR / f"existing_candidates_{sample_size}.tsv.gz"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    con = duckdb.connect()
    setup_connection(con)
    print("Building metadata vocabularies...", flush=True)
    catalog = build_catalog(con)
    print(
        f"  articles={len(catalog.article_ids):,}, "
        f"article_fields={len(catalog.article_cardinalities)}",
        flush=True,
    )

    print("\nBuilding temporal snapshots...", flush=True)
    training_snapshots = [
        build_experiment_snapshot(
            con,
            catalog,
            specification,
            sample_size,
            random_seed=42 + index,
        )
        for index, specification in enumerate(TRAIN_SNAPSHOTS)
    ]
    final_snapshot = build_experiment_snapshot(
        con,
        catalog,
        FINAL_SNAPSHOT,
        sample_size,
        random_seed=42,
    )

    tuning_train = training_snapshots[:3]
    tuning_validation = training_snapshots[3]
    normalizer = UserNumericalNormalizer.fit(tuning_train)
    initial_model = make_model(catalog)

    print("\nTiny overfit sanity check...", flush=True)
    overfit = tiny_overfit_check(
        initial_model,
        tuning_train[0],
        catalog,
        normalizer,
    )
    print(f"  {overfit}", flush=True)
    if not bool(overfit["passed"]):
        raise RuntimeError("Two-tower failed the tiny overfit sanity check")

    print("\nSelecting epoch on the latest historical snapshot...", flush=True)
    tuned_model, training_log = train_two_tower(
        model=initial_model,
        train_snapshots=tuning_train,
        validation_snapshot=tuning_validation,
        catalog=catalog,
        normalizer=normalizer,
        epochs=30,
        batch_size=256,
        random_negatives=256,
        temperature=0.07,
        patience=5,
    )
    selected_epoch = max(training_log.selected_epoch, 1)

    print(
        f"\nRefitting from scratch on all four snapshots for {selected_epoch} epochs...",
        flush=True,
    )
    final_normalizer = UserNumericalNormalizer.fit(training_snapshots)
    final_model, refit_losses = fit_for_fixed_epochs(
        model=make_model(catalog),
        snapshots=training_snapshots,
        catalog=catalog,
        normalizer=final_normalizer,
        epochs=selected_epoch,
        batch_size=256,
        random_negatives=256,
        temperature=0.07,
    )

    diagnostic = similarity_diagnostic(
        final_model,
        final_snapshot,
        catalog,
        final_normalizer,
    )
    print(f"\nHeld-out similarity diagnostic: {diagnostic.to_dict()}", flush=True)

    print("\nEncoding articles and validation users...", flush=True)
    item_embeddings = encode_all_items(final_model, catalog)
    user_embeddings = encode_snapshot_users(
        final_model,
        final_snapshot,
        final_normalizer,
    )
    item_norms = norm_summary(item_embeddings)
    user_norms = norm_summary(user_embeddings)

    embeddings_path = OUTPUT_DIR / f"two_tower_embeddings_{sample_size}.npz"
    ann_results_path = OUTPUT_DIR / f"two_tower_ann_results_{sample_size}.npz"
    ann_metrics_path = OUTPUT_DIR / f"two_tower_ann_metrics_{sample_size}.json"
    index_path = OUTPUT_DIR / f"two_tower_ivf_{sample_size}.faiss"
    np.savez_compressed(
        embeddings_path,
        item_embeddings=item_embeddings,
        user_embeddings=user_embeddings,
        eligible_item_indices=final_snapshot.eligible_item_indices,
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
    ann_scores = ann_results["scores"]
    ann_article_indices = ann_results["article_indices"]
    ann_metrics = json.loads(ann_metrics_path.read_text(encoding="utf-8"))
    ann_overlap = float(ann_metrics["ann_exact_top20"]["top_k_id_overlap"])
    sparse_mask = final_snapshot.history_weights.sum(axis=1) == 0
    sparse_user_norm_mean = (
        float(np.linalg.norm(user_embeddings[sparse_mask], axis=1).mean())
        if np.any(sparse_mask)
        else None
    )

    records = build_candidate_records(
        final_snapshot.customer_ids,
        catalog.article_ids,
        ann_scores,
        ann_article_indices,
        final_snapshot.cutoff,
    )
    two_tower_by_customer = records_to_candidates(records)
    candidate_path = OUTPUT_DIR / f"two_tower_candidates_{sample_size}.tsv.gz"
    write_two_tower_candidates(candidate_path, records)

    candidates_by_source: dict[str, dict[str, list[str]]] = {
        "two_tower": two_tower_by_customer
    }
    baseline_available = baseline_path.exists()
    if baseline_available:
        existing = load_existing_candidates(baseline_path)
        candidates_by_source = {**existing, "two_tower": two_tower_by_customer}
    else:
        print(
            f"\nBaseline candidates not found at {baseline_path}; "
            "standalone metrics will still be reported.",
            flush=True,
        )

    full_report = retrieval_report(
        final_snapshot.relevant_by_customer,
        candidates_by_source,
        len(catalog.article_ids),
        K_VALUES,
    )

    union_comparison: dict[str, Any] = {}
    oracle_comparison: dict[str, float] = {}
    als_overlap: dict[str, float] = {}
    if baseline_available:
        existing_only = {
            source: candidates_by_source[source] for source in SOURCE_ORDER
        }
        six_sources = {
            **existing_only,
            "two_tower": candidates_by_source["two_tower"],
        }
        for k in K_VALUES:
            five_steps = incremental_union_recall(
                final_snapshot.relevant_by_customer,
                existing_only,
                SOURCE_ORDER,
                k,
            )
            six_steps = incremental_union_recall(
                final_snapshot.relevant_by_customer,
                six_sources,
                [*SOURCE_ORDER, "two_tower"],
                k,
            )
            five_recall = float(five_steps[-1]["union_recall"])
            six_recall = float(six_steps[-1]["union_recall"])
            union_comparison[f"at_{k}"] = {
                "existing_5_source_recall": five_recall,
                "new_6_source_recall": six_recall,
                "two_tower_marginal_recall": six_recall - five_recall,
                "existing_average_pool": five_steps[-1]["avg_pool_size"],
                "new_average_pool": six_steps[-1]["avg_pool_size"],
            }
            overlap = pairwise_overlap(
                {
                    "als": existing_only["als"],
                    "two_tower": two_tower_by_customer,
                },
                final_snapshot.customer_ids,
                k,
            )
            als_overlap[f"at_{k}"] = overlap["als"]["two_tower"]

        oracle_comparison = {
            "existing_5_source_oracle_map_at_12": oracle_map(
                final_snapshot.relevant_by_customer,
                existing_only,
                500,
            ),
            "new_6_source_oracle_map_at_12": oracle_map(
                final_snapshot.relevant_by_customer,
                six_sources,
                500,
            ),
        }

    checkpoint_path = OUTPUT_DIR / f"two_tower_model_{sample_size}.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "embedding_dim": final_model.embedding_dim,
            "article_cardinalities": catalog.article_cardinalities,
            "customer_cardinalities": catalog.customer_cardinalities,
            "normalizer_mean": final_normalizer.mean,
            "normalizer_std": final_normalizer.std,
            "article_vocabularies": catalog.article_vocabularies,
            "customer_vocabularies": catalog.customer_vocabularies,
        },
        checkpoint_path,
    )
    report: dict[str, Any] = {
        "contract": {
            "training_snapshots": TRAIN_SNAPSHOTS,
            "final_snapshot": FINAL_SNAPSHOT,
            "sample_size": sample_size,
            "observed_items_only": True,
            "max_history": 50,
            "history_half_life_days": 45.0,
        },
        "architecture": {
            "embedding_dimension": 64,
            "article_categorical_fields": 8,
            "customer_categorical_fields": 2,
            "user_numerical_fields": USER_NUMERICAL_FIELDS,
            "parameter_count": final_model.parameter_count,
            "uses_customer_id_embedding": False,
            "loss": "sampled softmax",
            "negatives": "in-batch plus 256 sampled eligible unpurchased items",
            "temperature": 0.07,
        },
        "training": {
            "tiny_overfit": overfit,
            "epoch_selection": training_log.to_dict(),
            "selected_epoch": selected_epoch,
            "refit_losses": refit_losses,
        },
        "sanity_checks": {
            "held_out_similarity": diagnostic.to_dict(),
            "item_embedding_norms": item_norms,
            "user_embedding_norms": user_norms,
            "ann_exact_top20_overlap": ann_overlap,
            "ann": ann_metrics,
            "sparse_history_customers": final_snapshot.sparse_history_count,
            "sparse_user_embedding_norm_mean": sparse_user_norm_mean,
        },
        "retrieval": full_report,
        "union_comparison": union_comparison,
        "als_overlap": als_overlap,
        "oracle_comparison": oracle_comparison,
        "artifacts": {
            "candidates": str(candidate_path),
            "model": str(checkpoint_path),
            "index": str(index_path),
            "embeddings": str(embeddings_path),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    report_path = OUTPUT_DIR / f"two_tower_report_{sample_size}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    standalone = full_report["standalone"]["two_tower"]
    print("\nTwo-tower standalone retrieval", flush=True)
    for k in K_VALUES:
        print(
            f"  R@{k}: {standalone[f'recall_at_{k}']:.4f}  "
            f"HR@{k}: {standalone[f'hit_rate_at_{k}']:.4f}",
            flush=True,
        )
    print(f"  coverage: {standalone['catalog_coverage']:.4f}", flush=True)
    if union_comparison:
        print("\nMarginal union value", flush=True)
        for k in K_VALUES:
            values = union_comparison[f"at_{k}"]
            print(
                f"  K={k}: five={values['existing_5_source_recall']:.4f} "
                f"six={values['new_6_source_recall']:.4f} "
                f"delta={values['two_tower_marginal_recall']:+.4f}",
                flush=True,
            )
    print(f"\nReport: {report_path}")
    print(f"Runtime: {time.monotonic() - started:.1f}s")
    con.close()


if __name__ == "__main__":
    main()
