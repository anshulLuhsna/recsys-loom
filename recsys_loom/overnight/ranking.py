"""Shared ranking experiment helpers for the overnight program."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from recsys_loom.metrics import average_precision_at_k
from recsys_loom.overnight.protocol import (
    BASELINE_SOURCES,
    CACHE_DIR,
    DEV_CUSTOMERS,
    EXISTING_SOURCE_BUDGET,
    EXTRA_TRAINING_SNAPSHOTS,
    POPULARITY_BUDGET,
    ROOT,
    SNAPSHOTS,
    TT_BUDGET,
    ensure_directories,
)
from recsys_loom.ranking.cached_data import (
    SIX_SOURCES,
    SnapshotSpec,
    build_feature_data,
    load_candidate_pool,
    load_relevance,
)
from recsys_loom.ranking.ranker import train_ranker

HISTORY_BUCKETS = ["0", "1-2", "3-5", "6-10", "11-20", "20+"]


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


def snapshot_spec(snapshot: dict[str, str], customer_count: int) -> SnapshotSpec:
    cutoff = snapshot["cutoff"]
    return SnapshotSpec(
        cutoff=cutoff,
        target_start=snapshot["target_start"],
        target_end=snapshot["target_end"],
        customer_count=customer_count,
        existing_candidates_path=(
            ROOT
            / "artifacts"
            / "ranking"
            / "candidate_cache"
            / f"existing_{cutoff}_{customer_count}.tsv.gz"
        ),
        two_tower_candidates_path=(
            ROOT
            / "artifacts"
            / "two_tower"
            / "ranking_snapshots"
            / f"candidates_{cutoff}_{customer_count}.tsv.gz"
        ),
    )


def development_specs(
    customer_count: int = DEV_CUSTOMERS,
    include_extra_training: bool = False,
) -> list[SnapshotSpec]:
    snapshots = (
        [*EXTRA_TRAINING_SNAPSHOTS, *SNAPSHOTS]
        if include_extra_training
        else list(SNAPSHOTS)
    )
    return [snapshot_spec(snapshot, customer_count) for snapshot in snapshots]


def apply_budget(
    data: dict[str, NDArray],
    budget: int | dict[str, int],
    variable_source: str = "two_tower",
    source_names: list[str] | None = None,
    fixed_source_budgets: dict[str, int] | None = None,
) -> dict[str, NDArray]:
    source_names = list(source_names or SIX_SOURCES)
    feature_names = [str(value) for value in data["feature_names"]]
    defaults = {
        source: EXISTING_SOURCE_BUDGET
        for source in source_names
        if source != variable_source
    }
    defaults["recent_7d_pop"] = POPULARITY_BUDGET
    if variable_source in defaults:
        del defaults[variable_source]
    if fixed_source_budgets:
        defaults.update(fixed_source_budgets)
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
            budget[str(data["history_buckets"][customer_index])]
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
                else defaults.get(source)
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

    retained = {
        "features": retained_features,
        "labels": data["labels"][keep],
        "groups": np.asarray(group_sizes, dtype=np.int32),
        "pair_article_indices": data["pair_article_indices"][keep],
        "customer_ids": data["customer_ids"],
        "history_buckets": data["history_buckets"],
        "feature_names": data["feature_names"],
    }
    if "pair_categories" in data:
        retained["pair_categories"] = data["pair_categories"][keep]
    return retained


def combine_training(snapshots: list[dict[str, NDArray]]) -> dict[str, NDArray]:
    combined = {
        "features": np.concatenate(
            [snapshot["features"] for snapshot in snapshots]
        ),
        "labels": np.concatenate([snapshot["labels"] for snapshot in snapshots]),
        "groups": np.concatenate([snapshot["groups"] for snapshot in snapshots]),
        "feature_names": snapshots[0]["feature_names"],
    }
    if all("customer_ids" in snapshot for snapshot in snapshots):
        combined["customer_ids"] = np.concatenate(
            [snapshot["customer_ids"] for snapshot in snapshots]
        )
    if all("history_buckets" in snapshot for snapshot in snapshots):
        combined["history_buckets"] = np.concatenate(
            [snapshot["history_buckets"] for snapshot in snapshots]
        )
    return combined


def load_history_buckets(
    connection: Any,
    customer_ids: list[str],
    cutoff: str,
) -> list[str]:
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE overnight_hist_customers (customer_id VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO overnight_hist_customers VALUES (?)",
        [(customer_id,) for customer_id in customer_ids],
    )
    rows = connection.sql(f"""
        SELECT c.customer_id, COUNT(t.article_id) AS history_count
        FROM overnight_hist_customers c
        LEFT JOIN transactions t
          ON c.customer_id = t.customer_id
         AND t.transaction_date <= DATE '{cutoff}'
        GROUP BY c.customer_id
    """).fetchall()
    connection.execute("DROP TABLE overnight_hist_customers")
    counts = {customer_id: int(count) for customer_id, count in rows}
    return [history_bucket(counts.get(customer_id, 0)) for customer_id in customer_ids]


def feature_cache_path(
    cutoff: str,
    customer_count: int,
    tag: str = "",
) -> Path:
    suffix = f"_{tag}" if tag else ""
    return CACHE_DIR / f"features_{cutoff}_{customer_count}_six{suffix}.npz"


def load_or_build_snapshot(
    connection: Any,
    specification: SnapshotSpec,
    article_to_index: dict[str, int],
    source_names: list[str] | None = None,
    cache_tag: str = "",
) -> tuple[dict[str, NDArray], dict[str, set[str]]]:
    ensure_directories()
    source_names = list(source_names or BASELINE_SOURCES)
    cache_path = feature_cache_path(
        specification.cutoff,
        specification.customer_count,
        cache_tag,
    )
    relevance = None
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        data = {key: loaded[key] for key in loaded.files}
        customer_ids = [str(value) for value in data["customer_ids"]]
        relevance = load_relevance(
            connection,
            customer_ids,
            specification.target_start,
            specification.target_end,
        )
        return data, relevance

    pool = load_candidate_pool(specification)
    relevance = load_relevance(
        connection,
        pool.customer_ids,
        specification.target_start,
        specification.target_end,
    )
    data = build_feature_data(
        connection,
        pool,
        relevance,
        specification.cutoff,
        source_names,
        article_to_index,
    )
    data["history_buckets"] = np.asarray(
        load_history_buckets(connection, pool.customer_ids, specification.cutoff)
    )
    np.savez(cache_path, **data)
    del pool
    gc.collect()
    return data, relevance


def candidate_metrics(
    data: dict[str, NDArray],
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
        "average_candidate_count": (
            len(data["labels"]) / customer_count if customer_count else 0.0
        ),
        "candidate_recall": matched / total_relevant if total_relevant else 0.0,
        "candidate_hit_rate": (
            customers_with_hit / customer_count if customer_count else 0.0
        ),
        "oracle_map_at_12": float(np.mean(oracle_values)) if oracle_values else 0.0,
        "matched_relevant_pairs": matched,
        "training_positives": int(data["labels"].sum()),
        "total_candidates": len(data["labels"]),
        "customers": customer_count,
    }


def ranking_metrics(
    scores: NDArray[np.floating],
    data: dict[str, NDArray],
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
        for bucket in HISTORY_BUCKETS
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
                if customers
                else 0.0
            ),
            "average_candidate_count": (
                int(accumulator["candidate_count"]) / customers
                if customers
                else 0.0
            ),
            "customers": customers,
        }

    return finalize(overall), {
        bucket: finalize(values) for bucket, values in by_bucket.items()
    }


def fit_ranker(
    train_snapshots: list[dict[str, NDArray]],
    validation: dict[str, NDArray] | None = None,
    params_update: dict[str, Any] | None = None,
    **train_kwargs: Any,
) -> Any:
    train = combine_training(train_snapshots)
    valid_features = validation["features"] if validation is not None else None
    valid_labels = validation["labels"] if validation is not None else None
    valid_groups = validation["groups"] if validation is not None else None
    model = train_ranker(
        train["features"],
        train["labels"],
        train["groups"],
        [str(value) for value in train["feature_names"]],
        valid_features,
        valid_labels,
        valid_groups,
        verbose=train_kwargs.pop("verbose", 0),
        params_update=params_update,
        **train_kwargs,
    )
    del train
    gc.collect()
    return model


def evaluate_model(
    model: Any,
    validation: dict[str, NDArray],
    relevance: dict[str, set[str]],
    article_ids: list[str],
) -> dict[str, Any]:
    scores = model.predict(validation["features"])
    ranker, by_bucket = ranking_metrics(
        scores,
        validation,
        article_ids,
        relevance,
    )
    return {
        "candidate": candidate_metrics(validation, relevance),
        "ranker": ranker,
        "ranker_by_history_bucket": by_bucket,
        "scores": np.asarray(scores, dtype=np.float32),
        "training_positives": int(
            sum(int(snapshot["labels"].sum()) for snapshot in [validation])
        ),
    }


def apply_baseline_budget(data: dict[str, NDArray]) -> dict[str, NDArray]:
    return apply_budget(data, TT_BUDGET)


def _quantile_name(value: float, low: float, high: float) -> str:
    if value <= low:
        return "long_tail"
    if value <= high:
        return "medium"
    return "head"


def _novelty_name(times_bought: float, type_affinity: float) -> str:
    if times_bought > 0:
        return "exact_repeat"
    if type_affinity > 0:
        return "same_product_type"
    return "novel"


def _source_name(count: float) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    return "4+"


def _summarize_ranks(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        return {"n": 0, "median_rank": None, "top12_rate": 0.0}
    ordered = sorted(ranks)
    mid = ordered[len(ordered) // 2]
    return {
        "n": len(ranks),
        "median_rank": float(mid),
        "top12_rate": sum(rank <= 12 for rank in ranks) / len(ranks),
    }


def segment_report(
    scores: NDArray[np.floating],
    data: dict[str, NDArray],
    article_ids: list[str],
    relevance: dict[str, set[str]],
) -> dict[str, Any]:
    overall, by_history = ranking_metrics(scores, data, article_ids, relevance)
    names = [str(value) for value in data["feature_names"]]
    index = {name: position for position, name in enumerate(names)}
    popularity = np.nan_to_num(data["features"][:, index["item_purchases_30d"]], nan=0.0)
    positives = data["labels"] > 0
    if int(positives.sum()) >= 3:
        low, high = np.quantile(popularity[positives], [1.0 / 3.0, 2.0 / 3.0])
    else:
        low, high = 0.0, 0.0
    buckets = {
        "popularity": {},
        "novelty": {},
        "source_count": {},
    }
    offset = 0
    for group_size_value in data["groups"]:
        group_size = int(group_size_value)
        end = offset + group_size
        order = np.argsort(-scores[offset:end])
        ranks = np.empty(group_size, dtype=np.int32)
        ranks[order] = np.arange(1, group_size + 1)
        for local in range(group_size):
            if data["labels"][offset + local] <= 0:
                continue
            row = data["features"][offset + local]
            pop_name = _quantile_name(float(popularity[offset + local]), float(low), float(high))
            novelty = _novelty_name(
                float(np.nan_to_num(row[index["times_bought_item"]], nan=0.0)),
                float(np.nan_to_num(row[index["affinity_product_type_name"]], nan=0.0)),
            )
            sources = _source_name(float(row[index["num_sources"]]))
            for family, key in (
                ("popularity", pop_name),
                ("novelty", novelty),
                ("source_count", sources),
            ):
                buckets[family].setdefault(key, []).append(int(ranks[local]))
        offset = end
    return {
        "overall": overall,
        "by_history_bucket": by_history,
        "retrieved_positives": {
            family: {key: _summarize_ranks(ranks) for key, ranks in values.items()}
            for family, values in buckets.items()
        },
        "candidate": candidate_metrics(data, relevance),
    }
