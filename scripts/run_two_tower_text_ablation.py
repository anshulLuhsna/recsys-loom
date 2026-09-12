#!/usr/bin/env python3
"""Matched metadata-only versus metadata+text two-tower ablation."""

from __future__ import annotations

import csv
import gc
import gzip
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.ranking.cached_data import BASE_SOURCES
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
K_VALUES = [50, 100, 300, 500]
OUTPUT_DIR = ROOT / "artifacts" / "text_retrieval" / "two_tower_ablation"
TEXT_DIR = ROOT / "artifacts" / "text_retrieval"


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
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


def make_model(
    catalog: Catalog,
    text_embedding_dim: int,
) -> TwoTowerModel:
    return TwoTowerModel(
        article_cardinalities=catalog.article_cardinalities,
        customer_cardinalities=catalog.customer_cardinalities,
        numerical_dim=len(USER_NUMERICAL_FIELDS),
        embedding_dim=64,
        categorical_embedding_dim=8,
        customer_embedding_dim=4,
        hidden_dim=128,
        dropout=0.1,
        text_embedding_dim=text_embedding_dim,
        text_projection_dim=64,
    )


def target_customers(
    con: duckdb.DuckDBPyConnection,
    specification: dict[str, str],
    sample_size: int,
    final_snapshot: bool,
) -> list[str]:
    if final_snapshot:
        return load_target_customers(
            con,
            specification["target_start"],
            specification["target_end"],
            sample_size,
            random_seed=42,
        )
    rows = con.sql(f"""
        SELECT DISTINCT customer_id
        FROM transactions
        WHERE transaction_date BETWEEN
            DATE '{specification["target_start"]}'
            AND DATE '{specification["target_end"]}'
        ORDER BY customer_id
        LIMIT {int(sample_size)}
    """).fetchall()
    return [row[0] for row in rows]


def existing_candidate_path(
    specification: dict[str, str],
    sample_size: int,
    final_snapshot: bool,
) -> Path:
    if final_snapshot:
        return (
            ROOT
            / "artifacts"
            / "two_tower"
            / f"existing_candidates_{sample_size}.tsv.gz"
        )
    return (
        ROOT
        / "artifacts"
        / "ranking"
        / "candidate_cache"
        / f"existing_{specification['cutoff']}_{sample_size}.tsv.gz"
    )


def load_existing_union(
    path: Path,
    customer_ids: list[str],
) -> dict[str, list[str]]:
    by_source: dict[str, dict[str, list[str]]] = {
        source: {} for source in BASE_SOURCES
    }
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source = row["source_name"]
            if source in by_source:
                by_source[source].setdefault(row["customer_id"], []).append(
                    row["article_id"]
                )
    result: dict[str, list[str]] = {}
    for customer_id in customer_ids:
        seen: set[str] = set()
        values: list[str] = []
        for source in BASE_SOURCES:
            for article_id in by_source[source].get(customer_id, [])[:500]:
                if article_id not in seen:
                    seen.add(article_id)
                    values.append(article_id)
        result[customer_id] = values
    return result


def write_candidates(
    path: Path,
    customer_ids: list[str],
    article_ids: list[str],
    scores: np.ndarray,
    article_indices: np.ndarray,
    cutoff: str,
) -> dict[str, list[str]]:
    candidates = {customer_id: [] for customer_id in customer_ids}
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["customer_id", "article_id", "rank", "score", "feature_cutoff"])
        for customer_index, customer_id in enumerate(customer_ids):
            for rank, (score, item_index) in enumerate(
                zip(scores[customer_index], article_indices[customer_index]),
                start=1,
            ):
                if int(item_index) < 0:
                    continue
                article_id = article_ids[int(item_index)]
                candidates[customer_id].append(article_id)
                writer.writerow(
                    [customer_id, article_id, rank, float(score), cutoff]
                )
    return candidates


def retrieval_metrics(
    relevance: dict[str, set[str]],
    candidates: dict[str, list[str]],
    catalog_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for k in K_VALUES:
        hits = 0
        customers_with_hit = 0
        predicted_items: set[str] = set()
        for customer_id, relevant in relevance.items():
            proposed = candidates.get(customer_id, [])[:k]
            customer_hits = len(relevant.intersection(proposed))
            hits += customer_hits
            customers_with_hit += int(customer_hits > 0)
            predicted_items.update(proposed)
        total_relevant = sum(len(items) for items in relevance.values())
        report[f"at_{k}"] = {
            "recall": hits / total_relevant if total_relevant else 0.0,
            "hit_rate": customers_with_hit / len(relevance),
            "catalog_coverage": len(predicted_items) / catalog_size,
            "matched_relevant_pairs": hits,
        }
    return report


def union_metrics(
    relevance: dict[str, set[str]],
    existing: dict[str, list[str]],
    two_tower: dict[str, list[str]],
    k: int = 50,
) -> dict[str, float | int]:
    hits = 0
    customers_with_hit = 0
    candidate_count = 0
    oracle_values: list[float] = []
    for customer_id, relevant in relevance.items():
        seen = set(existing.get(customer_id, []))
        seen.update(two_tower.get(customer_id, [])[:k])
        customer_hits = len(relevant.intersection(seen))
        hits += customer_hits
        customers_with_hit += int(customer_hits > 0)
        candidate_count += len(seen)
        denominator = min(len(relevant), 12)
        oracle_values.append(
            min(customer_hits, 12) / denominator if denominator else 0.0
        )
    total_relevant = sum(len(items) for items in relevance.values())
    customers = len(relevance)
    return {
        "candidate_recall": hits / total_relevant if total_relevant else 0.0,
        "candidate_hit_rate": customers_with_hit / customers,
        "oracle_map_at_12": float(np.mean(oracle_values)),
        "average_candidate_count": candidate_count / customers,
        "matched_relevant_pairs": hits,
    }


def overlap_metrics(
    metadata: dict[str, list[str]],
    text: dict[str, list[str]],
    relevance: dict[str, set[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in K_VALUES:
        intersections = 0
        unions = 0
        shared_relevant = 0
        combined_relevant = 0
        for customer_id, relevant in relevance.items():
            metadata_set = set(metadata.get(customer_id, [])[:k])
            text_set = set(text.get(customer_id, [])[:k])
            intersections += len(metadata_set.intersection(text_set))
            unions += len(metadata_set.union(text_set))
            metadata_hits = metadata_set.intersection(relevant)
            text_hits = text_set.intersection(relevant)
            shared_relevant += len(metadata_hits.intersection(text_hits))
            combined_relevant += len(metadata_hits.union(text_hits))
        result[f"at_{k}"] = {
            "candidate_jaccard": intersections / unions if unions else 0.0,
            "relevant_hit_jaccard": (
                shared_relevant / combined_relevant if combined_relevant else 0.0
            ),
        }
    return result


def item_diagnostics(
    con: duckdb.DuckDBPyConnection,
    cutoff: str,
    relevance: dict[str, set[str]],
    metadata_candidates: dict[str, list[str]],
    text_candidates: dict[str, list[str]],
    k: int = 500,
) -> dict[str, Any]:
    rows = con.sql(f"""
        WITH article_stats AS (
            SELECT
                article_id,
                COUNT(*) AS purchase_count,
                MIN(transaction_date) AS first_purchase
            FROM transactions
            WHERE transaction_date <= DATE '{cutoff}'
            GROUP BY article_id
        ),
        metadata_groups AS (
            SELECT
                article_id,
                COUNT(*) OVER (
                    PARTITION BY
                        product_type_name,
                        colour_group_name,
                        garment_group_name,
                        section_name
                ) AS metadata_group_size
            FROM articles
        )
        SELECT
            a.article_id,
            COALESCE(s.purchase_count, 0),
            s.first_purchase,
            a.product_group_name,
            LENGTH(COALESCE(a.detail_desc, '')),
            g.metadata_group_size
        FROM articles a
        LEFT JOIN article_stats s USING (article_id)
        INNER JOIN metadata_groups g USING (article_id)
    """).fetchall()
    purchase_counts = np.asarray([int(row[1]) for row in rows])
    description_lengths = np.asarray([int(row[4]) for row in rows])
    popularity_threshold = float(np.quantile(purchase_counts, 0.8))
    rich_description_threshold = float(np.median(description_lengths))
    article_data = {
        article_id: {
            "purchase_count": int(purchase_count),
            "first_purchase": first_purchase,
            "product_group": product_group or "Unknown",
            "description_length": int(description_length),
            "metadata_group_size": int(metadata_group_size),
        }
        for (
            article_id,
            purchase_count,
            first_purchase,
            product_group,
            description_length,
            metadata_group_size,
        ) in rows
    }
    accumulators: dict[str, dict[str, int]] = {}

    def add(segment: str, metadata_hit: bool, text_hit: bool) -> None:
        values = accumulators.setdefault(
            segment,
            {
                "relevant_pairs": 0,
                "metadata_hits": 0,
                "text_hits": 0,
                "text_only_hits": 0,
                "metadata_only_hits": 0,
            },
        )
        values["relevant_pairs"] += 1
        values["metadata_hits"] += int(metadata_hit)
        values["text_hits"] += int(text_hit)
        values["text_only_hits"] += int(text_hit and not metadata_hit)
        values["metadata_only_hits"] += int(metadata_hit and not text_hit)

    for customer_id, relevant_items in relevance.items():
        metadata_set = set(metadata_candidates.get(customer_id, [])[:k])
        text_set = set(text_candidates.get(customer_id, [])[:k])
        for article_id in relevant_items:
            item = article_data.get(article_id)
            if item is None:
                continue
            metadata_hit = article_id in metadata_set
            text_hit = article_id in text_set
            segments = [
                (
                    "popular"
                    if int(item["purchase_count"]) >= popularity_threshold
                    else "long_tail"
                ),
                (
                    "rich_description"
                    if int(item["description_length"]) >= rich_description_threshold
                    else "short_description"
                ),
                (
                    "ambiguous_metadata"
                    if int(item["metadata_group_size"]) >= 50
                    else "specific_metadata"
                ),
            ]
            if int(item["purchase_count"]) <= 10:
                segments.append("low_history_item")
            first_purchase = item["first_purchase"]
            if first_purchase is not None:
                age = (date.fromisoformat(cutoff) - first_purchase).days
                segments.append("newer_90d" if age <= 90 else "established")
            segments.append(f"group:{item['product_group']}")
            for segment in segments:
                add(segment, metadata_hit, text_hit)

    diagnostics = {
        segment: {
            **values,
            "metadata_recall": (
                values["metadata_hits"] / values["relevant_pairs"]
            ),
            "text_recall": values["text_hits"] / values["relevant_pairs"],
            "recall_delta": (
                (values["text_hits"] - values["metadata_hits"])
                / values["relevant_pairs"]
            ),
        }
        for segment, values in accumulators.items()
    }
    group_names = sorted(
        (
            name for name in diagnostics
            if name.startswith("group:")
        ),
        key=lambda name: diagnostics[name]["relevant_pairs"],
        reverse=True,
    )[:10]
    return {
        "k": k,
        "popularity_threshold": popularity_threshold,
        "rich_description_character_threshold": rich_description_threshold,
        "metadata_ambiguity_group_size_threshold": 50,
        "segments": {
            name: values
            for name, values in diagnostics.items()
            if not name.startswith("group:")
        },
        "top_product_groups": {
            name.removeprefix("group:"): diagnostics[name]
            for name in group_names
        },
    }


def train_arm(
    arm: str,
    training_window: list[SnapshotData],
    inference_snapshot: SnapshotData,
    catalog: Catalog,
    text_embeddings: np.ndarray | None,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    arm_directory = OUTPUT_DIR / arm
    arm_directory.mkdir(parents=True, exist_ok=True)
    text_dim = 0 if text_embeddings is None else int(text_embeddings.shape[1])
    torch.manual_seed(42)
    np.random.seed(42)
    tuning_normalizer = UserNumericalNormalizer.fit(training_window[:3])
    initial_model = make_model(catalog, text_dim)
    overfit = tiny_overfit_check(
        initial_model,
        training_window[0],
        catalog,
        tuning_normalizer,
        text_embeddings=text_embeddings,
        device_name=device_name,
    )
    if not bool(overfit["passed"]):
        raise RuntimeError(f"{arm} failed tiny overfit check")
    torch.manual_seed(42)
    _, training_log = train_two_tower(
        make_model(catalog, text_dim),
        training_window[:3],
        training_window[3],
        catalog,
        tuning_normalizer,
        text_embeddings=text_embeddings,
        epochs=20,
        patience=4,
        verbose=False,
        device_name=device_name,
    )
    selected_epoch = max(training_log.selected_epoch, 1)
    final_normalizer = UserNumericalNormalizer.fit(training_window)
    torch.manual_seed(42)
    model, refit_losses = fit_for_fixed_epochs(
        make_model(catalog, text_dim),
        training_window,
        catalog,
        final_normalizer,
        epochs=selected_epoch,
        text_embeddings=text_embeddings,
        device_name=device_name,
    )
    diagnostic = similarity_diagnostic(
        model,
        inference_snapshot,
        catalog,
        final_normalizer,
        text_embeddings=text_embeddings,
        device_name=device_name,
    )
    item_embeddings = encode_all_items(
        model,
        catalog,
        text_embeddings=text_embeddings,
    )
    user_embeddings = encode_snapshot_users(
        model,
        inference_snapshot,
        final_normalizer,
        text_embeddings=text_embeddings,
    )
    stem = f"{inference_snapshot.cutoff}_{len(inference_snapshot.customer_ids)}"
    embeddings_path = arm_directory / f"ann_input_{stem}.npz"
    results_path = arm_directory / f"ann_result_{stem}.npz"
    metrics_path = arm_directory / f"ann_metrics_{stem}.json"
    index_path = arm_directory / f"ann_index_{stem}.faiss"
    np.savez_compressed(
        embeddings_path,
        item_embeddings=item_embeddings,
        user_embeddings=user_embeddings,
        eligible_item_indices=inference_snapshot.eligible_item_indices,
    )
    ann_started = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_two_tower_ann.py"),
            str(embeddings_path),
            str(results_path),
            str(index_path),
            str(metrics_path),
        ],
        check=True,
    )
    ann_seconds = time.perf_counter() - ann_started
    with np.load(results_path) as arrays:
        scores = arrays["scores"].astype(np.float32)
        article_indices = arrays["article_indices"].astype(np.int64)
    candidate_path = arm_directory / f"candidates_{stem}.tsv.gz"
    candidates = write_candidates(
        candidate_path,
        inference_snapshot.customer_ids,
        catalog.article_ids,
        scores,
        article_indices,
        inference_snapshot.cutoff,
    )
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "normalizer_mean": final_normalizer.mean,
            "normalizer_std": final_normalizer.std,
            "selected_epoch": selected_epoch,
            "text_embedding_dim": text_dim,
        },
        arm_directory / f"model_{stem}.pt",
    )
    report = {
        "arm": arm,
        "parameter_count": model.parameter_count,
        "selected_epoch": selected_epoch,
        "training_log": training_log.to_dict(),
        "refit_losses": refit_losses,
        "overfit_check": overfit,
        "similarity": diagnostic.to_dict(),
        "item_embedding_norm_mean": float(
            np.linalg.norm(item_embeddings, axis=1).mean()
        ),
        "user_embedding_norm_mean": float(
            np.linalg.norm(user_embeddings, axis=1).mean()
        ),
        "ann_seconds": ann_seconds,
        "ann_ms_per_customer": (
            ann_seconds * 1000.0 / len(inference_snapshot.customer_ids)
        ),
        "ann": json.loads(metrics_path.read_text(encoding="utf-8")),
        "candidate_path": str(candidate_path),
    }
    embeddings_path.unlink()
    results_path.unlink()
    index_path.unlink()
    del model, item_embeddings, user_embeddings, scores, article_indices
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return report, candidates


def main() -> None:
    training_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    evaluation_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    final_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    con = duckdb.connect()
    setup_connection(con)
    catalog = build_catalog(con)
    text_article_ids = [
        str(value)
        for value in np.load(TEXT_DIR / "text_article_ids.npy")
    ]
    if text_article_ids != catalog.article_ids:
        raise RuntimeError("Text embedding article IDs do not align with catalog")
    text_embeddings = np.load(
        TEXT_DIR / "text_embeddings.f32.npy",
        mmap_mode="r",
    )

    weekly_data: list[SnapshotData] = []
    for index, specification in enumerate(WEEKLY_SNAPSHOTS):
        customer_ids = load_target_customers(
            con,
            specification["target_start"],
            specification["target_end"],
            sample_size=training_sample,
            random_seed=100 + index,
        )
        weekly_data.append(
            build_snapshot(
                con,
                catalog,
                specification["cutoff"],
                specification["target_start"],
                specification["target_end"],
                customer_ids,
                observed_items_only=True,
            )
        )

    fold_reports: list[dict[str, Any]] = []
    final_candidates: dict[str, dict[str, list[str]]] = {}
    final_relevance: dict[str, set[str]] = {}
    for ranking_index, specification in enumerate(RANKING_SNAPSHOTS):
        sample_size = final_sample if ranking_index == 4 else evaluation_sample
        customers = target_customers(
            con,
            specification,
            sample_size,
            final_snapshot=ranking_index == 4,
        )
        inference_snapshot = build_snapshot(
            con,
            catalog,
            specification["cutoff"],
            specification["target_start"],
            specification["target_end"],
            customers,
            observed_items_only=True,
        )
        training_window = weekly_data[ranking_index : ranking_index + 4]
        print(f"\nAblation cutoff {specification['cutoff']}...", flush=True)
        metadata_report, metadata_candidates = train_arm(
            "metadata",
            training_window,
            inference_snapshot,
            catalog,
            None,
            device_name,
        )
        text_report, text_candidates = train_arm(
            "metadata_text",
            training_window,
            inference_snapshot,
            catalog,
            text_embeddings,
            device_name,
        )
        existing = load_existing_union(
            existing_candidate_path(
                specification,
                sample_size,
                final_snapshot=ranking_index == 4,
            ),
            customers,
        )
        fold_report = {
            "snapshot": specification,
            "customers": sample_size,
            "metadata": {
                **metadata_report,
                "retrieval": retrieval_metrics(
                    inference_snapshot.relevant_by_customer,
                    metadata_candidates,
                    len(catalog.article_ids),
                ),
                "union_at_50": union_metrics(
                    inference_snapshot.relevant_by_customer,
                    existing,
                    metadata_candidates,
                ),
            },
            "metadata_text": {
                **text_report,
                "retrieval": retrieval_metrics(
                    inference_snapshot.relevant_by_customer,
                    text_candidates,
                    len(catalog.article_ids),
                ),
                "union_at_50": union_metrics(
                    inference_snapshot.relevant_by_customer,
                    existing,
                    text_candidates,
                ),
            },
            "candidate_overlap": overlap_metrics(
                metadata_candidates,
                text_candidates,
                inference_snapshot.relevant_by_customer,
            ),
        }
        fold_reports.append(fold_report)
        if ranking_index == 4:
            final_candidates = {
                "metadata": metadata_candidates,
                "metadata_text": text_candidates,
            }
            final_relevance = inference_snapshot.relevant_by_customer
        del inference_snapshot, existing
        if ranking_index != 4:
            del metadata_candidates, text_candidates
        gc.collect()

    report = {
        "contract": {
            "training_customers_per_week": training_sample,
            "historical_evaluation_customers": evaluation_sample,
            "final_evaluation_customers": final_sample,
            "user_tower": "unchanged recency-weighted shared-item aggregation",
            "item_embedding_dim": 64,
            "text_embedding_dim": int(text_embeddings.shape[1]),
            "text_projection_dim": 64,
            "loss": "unchanged sampled softmax",
            "ann": "unchanged FAISS IVFFlat",
            "initial_candidate_budget": 50,
            "text_encoder_frozen": True,
            "images_used": False,
        },
        "folds": fold_reports,
        "final_candidate_counts": {
            arm: sum(len(values) for values in candidates.values())
            for arm, candidates in final_candidates.items()
        },
        "final_item_diagnostics": item_diagnostics(
            con,
            RANKING_SNAPSHOTS[-1]["cutoff"],
            final_relevance,
            final_candidates["metadata"],
            final_candidates["metadata_text"],
        ),
    }
    report_path = OUTPUT_DIR / (
        f"two_tower_text_ablation_{training_sample}_{evaluation_sample}_{final_sample}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
