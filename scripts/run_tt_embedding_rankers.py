#!/usr/bin/env python3
"""Matched R1/R2/R3 ranking ablation with temporal two-tower embeddings."""

from __future__ import annotations

import csv
import gc
import gzip
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.ranking.cached_data import (
    SIX_SOURCES,
    SnapshotSpec,
    build_feature_data,
    load_candidate_pool,
    load_relevance,
)
from recsys_loom.ranking.dcn_v2 import DCNV2, FeatureNormalizer, SimpleMLPRanker
from recsys_loom.ranking.temporal_neural import (
    PairEmbeddingStore,
    evaluate_scores,
    evaluate_temporal_ranker,
    fit_temporal_neural_fixed_epochs,
    train_temporal_neural_ranker,
)
from recsys_loom.two_tower.data import build_catalog

SNAPSHOTS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
    {"cutoff": "2020-09-15", "target_start": "2020-09-16", "target_end": "2020-09-22"},
]
OUTPUT_DIR = ROOT / "artifacts" / "ranking"


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


def make_spec(snapshot: dict[str, str], sample_size: int) -> SnapshotSpec:
    cutoff = snapshot["cutoff"]
    return SnapshotSpec(
        cutoff=cutoff,
        target_start=snapshot["target_start"],
        target_end=snapshot["target_end"],
        customer_count=sample_size,
        existing_candidates_path=(
            ROOT
            / "artifacts"
            / "ranking"
            / "candidate_cache"
            / f"existing_{cutoff}_{sample_size}.tsv.gz"
        ),
        two_tower_candidates_path=(
            ROOT
            / "artifacts"
            / "two_tower"
            / "ranking_snapshots"
            / f"candidates_{cutoff}_{sample_size}.tsv.gz"
        ),
    )


def candidate_customer_order(path: Path) -> list[str]:
    order: list[str] = []
    previous: str | None = None
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            customer_id = row["customer_id"]
            if customer_id != previous:
                order.append(customer_id)
                previous = customer_id
    return order


def load_embedding_store(
    specification: SnapshotSpec,
    data: dict[str, np.ndarray],
) -> PairEmbeddingStore:
    path = (
        ROOT
        / "artifacts"
        / "two_tower"
        / "ranking_snapshots"
        / f"embeddings_{specification.cutoff}_{specification.customer_count}.npz"
    )
    with np.load(path) as arrays:
        user_embeddings = arrays["user_embeddings"].astype(np.float32)
        item_embeddings = arrays["item_embeddings"].astype(np.float32)
    embedding_customer_ids = candidate_customer_order(
        specification.two_tower_candidates_path
    )
    embedding_row = {
        customer_id: index
        for index, customer_id in enumerate(embedding_customer_ids)
    }
    reordered_users = user_embeddings[
        [embedding_row[str(customer_id)] for customer_id in data["customer_ids"]]
    ]
    return PairEmbeddingStore(
        user_embeddings=reordered_users,
        item_embeddings=item_embeddings,
        pair_user_indices=data["pair_customer_indices"],
        pair_item_indices=data["pair_article_indices"],
    )


def concatenate_snapshots(
    snapshots: list[dict[str, np.ndarray]],
    stores: list[PairEmbeddingStore],
) -> tuple[dict[str, np.ndarray], PairEmbeddingStore]:
    user_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    pair_user_parts: list[np.ndarray] = []
    pair_item_parts: list[np.ndarray] = []
    user_offset = 0
    item_offset = 0
    for data, store in zip(snapshots, stores):
        user_parts.append(store.user_embeddings)
        item_parts.append(store.item_embeddings)
        pair_user_parts.append(store.pair_user_indices + user_offset)
        pair_item_parts.append(store.pair_item_indices + item_offset)
        user_offset += len(store.user_embeddings)
        item_offset += len(store.item_embeddings)
    combined = {
        "features": np.concatenate([data["features"] for data in snapshots]),
        "labels": np.concatenate([data["labels"] for data in snapshots]),
        "groups": np.concatenate([data["groups"] for data in snapshots]),
        "pair_categories": np.concatenate(
            [data["pair_categories"] for data in snapshots]
        ),
        "pair_article_indices": np.concatenate(
            [data["pair_article_indices"] for data in snapshots]
        ),
        "feature_names": snapshots[0]["feature_names"],
        "category_cardinalities": snapshots[0]["category_cardinalities"],
    }
    embedding_store = PairEmbeddingStore(
        user_embeddings=np.concatenate(user_parts),
        item_embeddings=np.concatenate(item_parts),
        pair_user_indices=np.concatenate(pair_user_parts).astype(np.int32),
        pair_item_indices=np.concatenate(pair_item_parts).astype(np.int32),
    )
    return combined, embedding_store


def embedding_alignment(
    data: dict[str, np.ndarray],
    store: PairEmbeddingStore,
) -> dict[str, float | int]:
    score_index = list(data["feature_names"]).index("score_two_tower")
    source_index = list(data["feature_names"]).index("is_two_tower")
    source_rows = np.flatnonzero(data["features"][:, source_index] == 1)[:10_000]
    users = store.user_embeddings[store.pair_user_indices[source_rows]]
    items = store.item_embeddings[store.pair_item_indices[source_rows]]
    computed = np.sum(users * items, axis=1)
    recorded = data["features"][source_rows, score_index]
    errors = np.abs(computed - recorded)
    return {
        "checked_pairs": len(source_rows),
        "mean_absolute_error": float(errors.mean()),
        "max_absolute_error": float(errors.max()),
    }


def run_isolated_lambdarank(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep LightGBM and PyTorch OpenMP runtimes in separate processes."""
    with tempfile.TemporaryDirectory(
        prefix="tt_lambdarank_",
        dir=OUTPUT_DIR,
    ) as temporary:
        base = Path(temporary)
        inputs = base / "inputs"
        outputs = base / "outputs"
        inputs.mkdir()
        np.save(inputs / "train_features.npy", train["features"])
        np.save(inputs / "train_labels.npy", train["labels"])
        np.save(inputs / "train_groups.npy", train["groups"])
        np.save(inputs / "validation_features.npy", validation["features"])
        np.save(inputs / "validation_labels.npy", validation["labels"])
        np.save(inputs / "validation_groups.npy", validation["groups"])
        (inputs / "metadata.json").write_text(
            json.dumps(
                {
                    "feature_names": [
                        str(value) for value in train["feature_names"]
                    ]
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_lambdarank_array_scores.py"),
                str(inputs),
                str(outputs),
            ],
            check=True,
        )
        return (
            np.load(outputs / "train_scores.npy").astype(np.float32),
            np.load(outputs / "validation_scores.npy").astype(np.float32),
        )


def make_model(
    model_name: str,
    numerical_features: int,
    cardinalities: list[int],
    dense_context_dim: int,
) -> torch.nn.Module:
    arguments = {
        "num_numerical": numerical_features,
        "categorical_cardinalities": cardinalities,
        "embedding_dim": 8,
        "mlp_dims": (64, 32),
        "dropout": 0.2,
        "dense_context_dim": dense_context_dim,
    }
    if model_name == "mlp":
        return SimpleMLPRanker(**arguments)
    if model_name == "dcn_v2":
        return DCNV2(**arguments, cross_layers=2)
    raise ValueError(f"Unknown model: {model_name}")


def run_neural_arm(
    model_name: str,
    tuning_train: dict[str, np.ndarray],
    tuning_store: PairEmbeddingStore,
    tuning_validation: dict[str, np.ndarray],
    tuning_validation_store: PairEmbeddingStore,
    tuning_validation_relevance: dict[str, set[str]],
    all_train: dict[str, np.ndarray],
    all_train_store: PairEmbeddingStore,
    validation: dict[str, np.ndarray],
    validation_store: PairEmbeddingStore,
    validation_relevance: dict[str, set[str]],
    article_ids: list[str],
    lambdarank_hard_scores: np.ndarray,
) -> dict[str, Any]:
    feature_names = [str(value) for value in all_train["feature_names"]]
    cardinalities = [
        int(value) for value in all_train["category_cardinalities"]
    ]

    tuning_normalizer = FeatureNormalizer()
    tuning_normalizer.fit(tuning_train["features"], feature_names)
    tuning_train_features = tuning_normalizer.transform(tuning_train["features"])
    tuning_validation_features = tuning_normalizer.transform(
        tuning_validation["features"]
    )
    tuning_model = make_model(
        model_name,
        tuning_train_features.shape[1],
        cardinalities,
        tuning_store.context_dim,
    )
    tuning_model, tuning_log = train_temporal_neural_ranker(
        tuning_model,
        tuning_train_features,
        tuning_train["pair_categories"],
        tuning_train["labels"],
        tuning_train["groups"],
        tuning_store,
        feature_names,
        tuning_validation_features,
        tuning_validation["pair_categories"],
        tuning_validation["labels"],
        tuning_validation["groups"],
        tuning_validation_store,
        [str(value) for value in tuning_validation["customer_ids"]],
        tuning_validation["pair_article_indices"],
        article_ids,
        tuning_validation_relevance,
    )
    selected_epoch = max(tuning_log.best_epoch, 1)
    del tuning_model, tuning_train_features, tuning_validation_features
    gc.collect()

    final_normalizer = FeatureNormalizer()
    final_normalizer.fit(all_train["features"], feature_names)
    train_features = final_normalizer.transform(all_train["features"])
    validation_features = final_normalizer.transform(validation["features"])
    final_model = make_model(
        model_name,
        train_features.shape[1],
        cardinalities,
        all_train_store.context_dim,
    )
    parameter_count = sum(
        parameter.numel() for parameter in final_model.parameters()
    )
    final_model, refit_losses = fit_temporal_neural_fixed_epochs(
        final_model,
        train_features,
        all_train["pair_categories"],
        all_train["labels"],
        all_train["groups"],
        all_train_store,
        feature_names,
        selected_epoch,
        hard_scores=lambdarank_hard_scores,
    )
    metrics = evaluate_temporal_ranker(
        final_model,
        validation_features,
        validation["pair_categories"],
        validation["groups"],
        validation_store,
        [str(value) for value in validation["customer_ids"]],
        validation["pair_article_indices"],
        article_ids,
        validation_relevance,
    )
    return {
        "metrics": metrics,
        "parameter_count": parameter_count,
        "selected_epoch": selected_epoch,
        "tuning_log": tuning_log.to_dict(),
        "refit_train_losses": refit_losses,
    }


def main() -> None:
    started = time.monotonic()
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    setup_connection(con)
    catalog = build_catalog(con)
    specifications = [make_spec(snapshot, sample_size) for snapshot in SNAPSHOTS]
    missing = [
        str(path)
        for specification in specifications
        for path in (
            specification.existing_candidates_path,
            specification.two_tower_candidates_path,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing candidate artifacts:\n" + "\n".join(missing))

    data_by_snapshot: list[dict[str, np.ndarray]] = []
    stores: list[PairEmbeddingStore] = []
    relevance_by_snapshot: list[dict[str, set[str]]] = []
    alignment_reports: list[dict[str, object]] = []
    for specification in specifications:
        print(f"Building features for {specification.cutoff}...", flush=True)
        pool = load_candidate_pool(specification)
        relevance = load_relevance(
            con,
            pool.customer_ids,
            specification.target_start,
            specification.target_end,
        )
        data = build_feature_data(
            con,
            pool,
            relevance,
            specification.cutoff,
            SIX_SOURCES,
            catalog.article_to_index,
        )
        store = load_embedding_store(specification, data)
        alignment_reports.append(
            {
                "cutoff": specification.cutoff,
                **embedding_alignment(data, store),
            }
        )
        data_by_snapshot.append(data)
        stores.append(store)
        relevance_by_snapshot.append(relevance)
        del pool
        gc.collect()

    tuning_row_count = sum(
        len(snapshot["labels"]) for snapshot in data_by_snapshot[:3]
    )
    tuning_group_count = sum(
        len(snapshot["groups"]) for snapshot in data_by_snapshot[:3]
    )
    all_train, all_train_store = concatenate_snapshots(
        data_by_snapshot[:4],
        stores[:4],
    )
    tuning_train = {
        "features": all_train["features"][:tuning_row_count],
        "labels": all_train["labels"][:tuning_row_count],
        "groups": all_train["groups"][:tuning_group_count],
        "pair_categories": all_train["pair_categories"][:tuning_row_count],
        "pair_article_indices": all_train["pair_article_indices"][:tuning_row_count],
        "feature_names": all_train["feature_names"],
        "category_cardinalities": all_train["category_cardinalities"],
    }
    tuning_store = all_train_store
    tuning_validation = data_by_snapshot[3]
    tuning_validation_store = stores[3]
    validation = data_by_snapshot[4]
    validation_store = stores[4]
    del data_by_snapshot, stores
    gc.collect()

    print("Training matched six-source LambdaRank R1...", flush=True)
    lambdarank_hard_scores, validation_lambdarank_scores = (
        run_isolated_lambdarank(all_train, validation)
    )
    lambdarank_metrics = evaluate_scores(
        validation_lambdarank_scores,
        validation["groups"],
        [str(value) for value in validation["customer_ids"]],
        validation["pair_article_indices"],
        catalog.article_ids,
        relevance_by_snapshot[4],
    )
    oracle_metrics = evaluate_scores(
        validation["labels"].astype(np.float32),
        validation["groups"],
        [str(value) for value in validation["customer_ids"]],
        validation["pair_article_indices"],
        catalog.article_ids,
        relevance_by_snapshot[4],
    )

    results: dict[str, Any] = {}
    for model_name in ("mlp", "dcn_v2"):
        print(f"\nRunning {model_name} + TT embeddings...", flush=True)
        results[model_name] = run_neural_arm(
            model_name,
            tuning_train,
            tuning_store,
            tuning_validation,
            tuning_validation_store,
            relevance_by_snapshot[3],
            all_train,
            all_train_store,
            validation,
            validation_store,
            relevance_by_snapshot[4],
            catalog.article_ids,
            lambdarank_hard_scores,
        )
        gc.collect()

    report = {
        "contract": {
            "customers_per_snapshot": sample_size,
            "candidate_sources": SIX_SOURCES,
            "embedding_dimension_per_tower": 64,
            "tuning_train_cutoffs": [
                specification.cutoff for specification in specifications[:3]
            ],
            "tuning_validation_cutoff": specifications[3].cutoff,
            "final_refit_cutoffs": [
                specification.cutoff for specification in specifications[:4]
            ],
            "untouched_final_cutoff": specifications[4].cutoff,
            "negative_sampling": (
                "50% random candidates, 50% hard candidates pooled from "
                "source scores; final refit also includes LambdaRank-hard scores"
            ),
        },
        "embedding_alignment": alignment_reports,
        "rankers": {
            "six_source_lambdarank": {"metrics": lambdarank_metrics},
            "mlp_tt": results["mlp"],
            "dcn_v2_tt": results["dcn_v2"],
            "oracle": {"metrics": oracle_metrics},
        },
        "runtime_seconds": time.monotonic() - started,
    }
    report_path = OUTPUT_DIR / f"tt_embedding_ranker_report_{sample_size}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
