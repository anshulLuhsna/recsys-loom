#!/usr/bin/env python3
"""Select a two-tower candidate budget with leakage-safe temporal folds."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.metrics import average_precision_at_k
from recsys_loom.ranking.cached_data import (
    BASE_SOURCES,
    SIX_SOURCES,
    SnapshotSpec,
    build_feature_data,
    load_candidate_pool,
    load_relevance,
)
from recsys_loom.ranking.ranker import train_ranker

TT_BUDGETS = [0, 25, 50, 100, 200, 300]
SNAPSHOTS = [
    {"cutoff": "2020-08-17", "target_start": "2020-08-18", "target_end": "2020-08-24"},
    {"cutoff": "2020-08-24", "target_start": "2020-08-25", "target_end": "2020-08-31"},
    {"cutoff": "2020-08-31", "target_start": "2020-09-01", "target_end": "2020-09-07"},
    {"cutoff": "2020-09-07", "target_start": "2020-09-08", "target_end": "2020-09-14"},
    {"cutoff": "2020-09-15", "target_start": "2020-09-16", "target_end": "2020-09-22"},
]
SELECTION_FOLDS = [2, 3]
FINAL_FOLD = 4
BUCKET_ORDER = ["0", "1-2", "3-5", "6-10", "11-20", "20+"]
MAP_TOLERANCE = 0.0002
OUTPUT_DIR = ROOT / "artifacts" / "ranking"


def setup_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
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


def make_specs(
    training_customers: int,
    validation_customers: int,
) -> list[SnapshotSpec]:
    specifications: list[SnapshotSpec] = []
    for index, snapshot in enumerate(SNAPSHOTS):
        sample_size = (
            validation_customers if index == FINAL_FOLD else training_customers
        )
        if index == FINAL_FOLD:
            existing = (
                ROOT
                / "artifacts"
                / "two_tower"
                / f"existing_candidates_{sample_size}.tsv.gz"
            )
            two_tower = (
                ROOT
                / "artifacts"
                / "two_tower"
                / f"two_tower_candidates_{sample_size}.tsv.gz"
            )
        else:
            existing = (
                ROOT
                / "artifacts"
                / "ranking"
                / "candidate_cache"
                / f"existing_{snapshot['cutoff']}_{sample_size}.tsv.gz"
            )
            two_tower = (
                ROOT
                / "artifacts"
                / "two_tower"
                / "ranking_snapshots"
                / f"candidates_{snapshot['cutoff']}_{sample_size}.tsv.gz"
            )
        specifications.append(
            SnapshotSpec(
                cutoff=snapshot["cutoff"],
                target_start=snapshot["target_start"],
                target_end=snapshot["target_end"],
                customer_count=sample_size,
                existing_candidates_path=existing,
                two_tower_candidates_path=two_tower,
            )
        )
    return specifications


def history_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return "20+"


def load_history_buckets(
    con: duckdb.DuckDBPyConnection,
    customer_ids: list[str],
    cutoff: str,
) -> list[str]:
    con.execute("CREATE OR REPLACE TEMP TABLE budget_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO budget_customers VALUES (?)",
        [(customer_id,) for customer_id in customer_ids],
    )
    rows = con.sql(f"""
        SELECT c.customer_id, COUNT(t.article_id) AS history_count
        FROM budget_customers c
        LEFT JOIN transactions t
          ON c.customer_id = t.customer_id
         AND t.transaction_date <= DATE '{cutoff}'
        GROUP BY c.customer_id
    """).fetchall()
    con.execute("DROP TABLE budget_customers")
    counts = {customer_id: int(count) for customer_id, count in rows}
    return [history_bucket(counts.get(customer_id, 0)) for customer_id in customer_ids]


def apply_budget(
    data: dict[str, np.ndarray],
    budget: int | dict[str, int],
    variable_source: str = "two_tower",
    source_names: list[str] = SIX_SOURCES,
    fixed_source_budgets: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    feature_names = [str(value) for value in data["feature_names"]]
    fixed_source_budgets = fixed_source_budgets or {}
    source_columns = {
        source: [
            feature_names.index(f"is_{source}"),
            feature_names.index(f"rank_{source}"),
            feature_names.index(f"rank_pct_{source}"),
            feature_names.index(f"score_{source}"),
        ]
        for source in source_names
    }
    source_count_column = feature_names.index("num_sources")
    keep = np.zeros(len(data["features"]), dtype=np.bool_)
    suppressions = {
        source: np.zeros(len(data["features"]), dtype=np.bool_)
        for source in source_names
    }
    group_sizes: list[int] = []
    offset = 0

    for customer_index, group_size_value in enumerate(data["groups"]):
        group_size = int(group_size_value)
        end = offset + group_size
        customer_budget = (
            budget[data["history_buckets"][customer_index]]
            if isinstance(budget, dict)
            else budget
        )
        group_keep = np.zeros(group_size, dtype=np.bool_)
        for source in source_names:
            columns = source_columns[source]
            present = data["features"][offset:end, columns[0]] == 1.0
            source_budget = (
                customer_budget
                if source == variable_source
                else fixed_source_budgets.get(source)
            )
            allowed = present
            if source_budget is not None:
                rank = data["features"][offset:end, columns[1]]
                allowed = present & (rank <= source_budget) & np.isfinite(rank)
            group_keep |= allowed
            suppressions[source][offset:end] = present & ~allowed
        keep[offset:end] = group_keep
        group_sizes.append(int(group_keep.sum()))
        offset = end

    retained_features = data["features"][keep].copy()
    for source in source_names:
        columns = source_columns[source]
        retained_suppression = suppressions[source][keep]
        retained_features[retained_suppression, columns[0]] = 0.0
        suppressed_rows = np.flatnonzero(retained_suppression)
        retained_features[np.ix_(suppressed_rows, columns[1:])] = np.nan
        retained_features[retained_suppression, source_count_column] -= 1.0
    return {
        "features": retained_features,
        "labels": data["labels"][keep],
        "groups": np.asarray(group_sizes, dtype=np.int32),
        "pair_article_indices": data["pair_article_indices"][keep],
        "customer_ids": data["customer_ids"],
        "history_buckets": data["history_buckets"],
        "feature_names": data["feature_names"],
    }


def combine_training(
    snapshots: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        "features": np.concatenate([snapshot["features"] for snapshot in snapshots]),
        "labels": np.concatenate([snapshot["labels"] for snapshot in snapshots]),
        "groups": np.concatenate([snapshot["groups"] for snapshot in snapshots]),
        "feature_names": snapshots[0]["feature_names"],
    }


def candidate_metrics(
    data: dict[str, np.ndarray],
    relevance: dict[str, set[str]],
) -> dict[str, float | int]:
    offset = 0
    matched = 0
    customers_with_hit = 0
    oracle_values: list[float] = []
    for customer_id, group_size_value in zip(
        data["customer_ids"],
        data["groups"],
    ):
        group_size = int(group_size_value)
        end = offset + group_size
        hits = int(data["labels"][offset:end].sum())
        relevant_count = len(relevance.get(str(customer_id), set()))
        matched += hits
        customers_with_hit += int(hits > 0)
        denominator = min(relevant_count, 12)
        oracle_values.append(
            min(hits, 12) / denominator if denominator else 0.0
        )
        offset = end
    total_relevant = sum(len(items) for items in relevance.values())
    customer_count = len(data["groups"])
    return {
        "average_candidate_count": len(data["labels"]) / customer_count,
        "candidate_recall": matched / total_relevant if total_relevant else 0.0,
        "candidate_hit_rate": (
            customers_with_hit / customer_count if customer_count else 0.0
        ),
        "oracle_map_at_12": float(np.mean(oracle_values)),
        "matched_relevant_pairs": matched,
        "total_candidates": len(data["labels"]),
    }


def ranking_metrics(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    article_ids: list[str],
    relevance: dict[str, set[str]],
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    overall = {
        "aps": [],
        "hits": 0,
        "customers_with_hit": 0,
        "relevant": 0,
        "customers": 0,
        "candidate_count": 0,
    }
    by_bucket = {
        bucket: {
            "aps": [],
            "hits": 0,
            "customers_with_hit": 0,
            "relevant": 0,
            "customers": 0,
            "candidate_count": 0,
        }
        for bucket in BUCKET_ORDER
    }
    offset = 0
    for customer_index, (customer_id_value, group_size_value) in enumerate(
        zip(data["customer_ids"], data["groups"])
    ):
        customer_id = str(customer_id_value)
        group_size = int(group_size_value)
        end = offset + group_size
        take = min(12, group_size)
        local = np.argpartition(scores[offset:end], -take)[-take:]
        local = local[np.argsort(-scores[offset:end][local])]
        predicted = [
            article_ids[int(data["pair_article_indices"][offset + int(index)])]
            for index in local
        ]
        relevant = relevance.get(customer_id, set())
        hits = len(set(predicted).intersection(relevant))
        ap = average_precision_at_k(relevant, predicted, 12)
        bucket = str(data["history_buckets"][customer_index])
        for accumulator in (overall, by_bucket[bucket]):
            accumulator["aps"].append(ap)
            accumulator["hits"] += hits
            accumulator["customers_with_hit"] += int(hits > 0)
            accumulator["relevant"] += len(relevant)
            accumulator["customers"] += 1
            accumulator["candidate_count"] += group_size
        offset = end

    def finalize(accumulator: dict[str, Any]) -> dict[str, float | int]:
        customers = int(accumulator["customers"])
        relevant = int(accumulator["relevant"])
        return {
            "map_at_12": (
                float(np.mean(accumulator["aps"])) if customers else 0.0
            ),
            "recall_at_12": (
                int(accumulator["hits"]) / relevant if relevant else 0.0
            ),
            "hit_rate_at_12": (
                int(accumulator["customers_with_hit"]) / customers
                if customers else 0.0
            ),
            "average_candidate_count": (
                int(accumulator["candidate_count"]) / customers
                if customers else 0.0
            ),
            "customers": customers,
        }

    return finalize(overall), {
        bucket: finalize(values) for bucket, values in by_bucket.items()
    }


def run_arm(
    max_data: list[dict[str, np.ndarray]],
    relevance: list[dict[str, set[str]]],
    validation_index: int,
    budget: int | dict[str, int],
    article_ids: list[str],
    variable_source: str = "two_tower",
    source_names: list[str] = SIX_SOURCES,
    fixed_source_budgets: dict[str, int] | None = None,
) -> dict[str, Any]:
    filtered_train = [
        apply_budget(
            snapshot,
            budget,
            variable_source,
            source_names,
            fixed_source_budgets,
        )
        for snapshot in max_data[:validation_index]
    ]
    validation = apply_budget(
        max_data[validation_index],
        budget,
        variable_source,
        source_names,
        fixed_source_budgets,
    )
    train = combine_training(filtered_train)
    del filtered_train
    gc.collect()
    model = train_ranker(
        train["features"],
        train["labels"],
        train["groups"],
        [str(value) for value in train["feature_names"]],
        validation["features"],
        validation["labels"],
        validation["groups"],
        verbose=0,
    )
    scoring_started = time.perf_counter()
    scores = model.predict(validation["features"])
    scoring_seconds = time.perf_counter() - scoring_started
    ranker, by_bucket = ranking_metrics(
        scores,
        validation,
        article_ids,
        relevance[validation_index],
    )
    candidates = candidate_metrics(validation, relevance[validation_index])
    ranker["ranker_scoring_ms_per_customer"] = (
        scoring_seconds * 1000.0 / len(validation["groups"])
    )
    del train, validation, model, scores
    gc.collect()
    return {
        "candidate": candidates,
        "ranker": ranker,
        "ranker_by_history_bucket": by_bucket,
    }


def choose_budgets(
    selection_results: dict[int, list[dict[str, Any]]],
) -> tuple[int, dict[str, int], dict[str, Any]]:
    global_means = {
        budget: float(
            np.mean([fold["ranker"]["map_at_12"] for fold in folds])
        )
        for budget, folds in selection_results.items()
    }
    best_global_map = max(global_means.values())
    global_budget = min(
        budget
        for budget, mean_map in global_means.items()
        if mean_map >= best_global_map - MAP_TOLERANCE
    )

    policy: dict[str, int] = {}
    bucket_evidence: dict[str, Any] = {}
    for bucket in BUCKET_ORDER:
        mean_maps = {
            budget: float(
                np.mean(
                    [
                        fold["ranker_by_history_bucket"][bucket]["map_at_12"]
                        for fold in folds
                    ]
                )
            )
            for budget, folds in selection_results.items()
        }
        best_map = max(mean_maps.values())
        selected = min(
            budget
            for budget, mean_map in mean_maps.items()
            if mean_map >= best_map - MAP_TOLERANCE
        )
        policy[bucket] = selected
        bucket_evidence[bucket] = {
            "mean_map_by_budget": mean_maps,
            "best_mean_map": best_map,
            "selected_budget": selected,
        }
    return global_budget, policy, {
        "map_tolerance": MAP_TOLERANCE,
        "global_mean_map_by_budget": global_means,
        "selected_global_budget": global_budget,
        "history_policy": bucket_evidence,
    }


def add_efficiency(
    fold_results: dict[int, dict[str, Any]],
) -> None:
    baseline = fold_results[0]["candidate"]
    for budget, result in fold_results.items():
        candidates = result["candidate"]
        added_relevant = (
            int(candidates["matched_relevant_pairs"])
            - int(baseline["matched_relevant_pairs"])
        )
        added_candidates = (
            int(candidates["total_candidates"])
            - int(baseline["total_candidates"])
        )
        result["candidate"]["marginal_relevant_per_10k_added_candidates"] = (
            added_relevant / added_candidates * 10_000
            if added_candidates > 0
            else 0.0
        )


def main() -> None:
    started = time.monotonic()
    training_customers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    validation_customers = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specifications = make_specs(training_customers, validation_customers)
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

    con = duckdb.connect()
    setup_connection(con)
    article_ids = [
        row[0] for row in con.sql("SELECT article_id FROM articles ORDER BY article_id").fetchall()
    ]
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    max_data: list[dict[str, np.ndarray]] = []
    relevance: list[dict[str, set[str]]] = []
    for specification in specifications:
        print(f"Building max-K snapshot {specification.cutoff}...", flush=True)
        pool = load_candidate_pool(specification)
        snapshot_relevance = load_relevance(
            con,
            pool.customer_ids,
            specification.target_start,
            specification.target_end,
        )
        data = build_feature_data(
            con,
            pool,
            snapshot_relevance,
            specification.cutoff,
            SIX_SOURCES,
            article_to_index,
        )
        data["history_buckets"] = np.asarray(
            load_history_buckets(con, pool.customer_ids, specification.cutoff)
        )
        max_data.append(data)
        relevance.append(snapshot_relevance)
        del pool
        gc.collect()

    selection_results: dict[int, list[dict[str, Any]]] = {
        budget: [] for budget in TT_BUDGETS
    }
    for validation_index in SELECTION_FOLDS:
        print(
            f"\nSelection fold {specifications[validation_index].cutoff}",
            flush=True,
        )
        for budget in TT_BUDGETS:
            print(f"  TT K={budget}...", flush=True)
            selection_results[budget].append(
                run_arm(
                    max_data,
                    relevance,
                    validation_index,
                    budget,
                    article_ids,
                )
            )

    global_budget, history_policy, selection = choose_budgets(selection_results)
    print(f"\nFrozen global K: {global_budget}", flush=True)
    print(f"Frozen history policy: {history_policy}", flush=True)

    print("\nFinal fold budget sweep...", flush=True)
    final_results: dict[int, dict[str, Any]] = {}
    for budget in TT_BUDGETS:
        print(f"  TT K={budget}...", flush=True)
        final_results[budget] = run_arm(
            max_data,
            relevance,
            FINAL_FOLD,
            budget,
            article_ids,
        )
    add_efficiency(final_results)

    print("\nFinal frozen history-policy evaluation...", flush=True)
    final_policy_result = run_arm(
        max_data,
        relevance,
        FINAL_FOLD,
        history_policy,
        article_ids,
    )

    report = {
        "contract": {
            "training_customers_per_snapshot": training_customers,
            "final_validation_customers": validation_customers,
            "two_tower_budgets": TT_BUDGETS,
            "popularity_budget": 100,
            "existing_source_budget": 500,
            "two_tower_sweep_maximum": 300,
            "cached_two_tower_depth": {
                "training_snapshots": 300,
                "final_snapshot": 500,
            },
            "selection_fold_cutoffs": [
                specifications[index].cutoff for index in SELECTION_FOLDS
            ],
            "final_cutoff": specifications[FINAL_FOLD].cutoff,
            "latency_definition": "LambdaRank scoring only; milliseconds per customer",
            "selection_rule": (
                "smallest K within 0.0002 MAP@12 of the best earlier-fold mean"
            ),
        },
        "selection_results": {
            str(budget): folds for budget, folds in selection_results.items()
        },
        "selection": selection,
        "frozen_history_policy": history_policy,
        "final_budget_sweep": {
            str(budget): result for budget, result in final_results.items()
        },
        "final_frozen_global_budget": {
            "budget": global_budget,
            "result": final_results[global_budget],
        },
        "final_frozen_history_policy": final_policy_result,
        "runtime_seconds": time.monotonic() - started,
    }
    report_path = OUTPUT_DIR / (
        f"two_tower_budget_sweep_{training_customers}_{validation_customers}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
