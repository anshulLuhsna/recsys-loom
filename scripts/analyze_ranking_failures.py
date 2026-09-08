#!/usr/bin/env python3
"""Diagnose which retrieved positives LambdaRank buries."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.protocol import (
    DEV_CUSTOMERS,
    OVERNIGHT_DIR,
    SELECTION_FOLDS,
    SNAPSHOTS,
    ensure_directories,
)
from recsys_loom.overnight.ranking import (
    apply_baseline_budget,
    development_specs,
    fit_ranker,
    load_or_build_snapshot,
)

SOURCE_FLAGS = [
    "recent_7d_pop",
    "repeat_purchase",
    "cooccurrence",
    "als",
    "content",
    "two_tower",
]


def _quantile_bucket(value: float, q1: float, q2: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= q1:
        return "long_tail"
    if value <= q2:
        return "medium"
    return "head"


def _source_count_bucket(count: float) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    return "4+"


def _novelty(times_bought: float, type_affinity: float, garment_affinity: float) -> str:
    if np.isfinite(times_bought) and times_bought > 0:
        return "exact_repeat"
    if np.isfinite(type_affinity) and type_affinity > 0:
        return "same_product_type"
    if np.isfinite(garment_affinity) and garment_affinity > 0:
        return "same_garment_group"
    return "novel"


def _primary_source(row: np.ndarray, columns: dict[str, dict[str, int]]) -> str:
    present = [
        source
        for source in SOURCE_FLAGS
        if row[columns[source]["is"]] == 1.0
    ]
    if not present:
        return "none"
    if len(present) > 1:
        return "multi_source"
    return present[0]


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "n": 0,
            "median_rank": None,
            "top12_rate": 0.0,
            "mean_ap_contribution": 0.0,
        }
    ranks = np.asarray([row["rank"] for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "median_rank": float(np.median(ranks)),
        "p90_rank": float(np.quantile(ranks, 0.9)),
        "top12_rate": float(np.mean([rank <= 12 for rank in ranks])),
        "mean_source_count": float(
            np.mean([row["source_count"] for row in rows])
        ),
    }


def analyze_fold(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    feature_index: dict[str, int],
) -> list[dict[str, object]]:
    columns = {
        source: {
            "is": feature_index[f"is_{source}"],
            "rank": feature_index[f"rank_{source}"],
            "score": feature_index[f"score_{source}"],
        }
        for source in SOURCE_FLAGS
    }
    records: list[dict[str, object]] = []
    offset = 0
    for customer_index, group_size_value in enumerate(data["groups"]):
        group_size = int(group_size_value)
        end = offset + group_size
        group_scores = scores[offset:end]
        order = np.argsort(-group_scores)
        ranks = np.empty(group_size, dtype=np.int32)
        ranks[order] = np.arange(1, group_size + 1)
        labels = data["labels"][offset:end]
        features = data["features"][offset:end]
        for local_index in np.flatnonzero(labels > 0):
            row = features[int(local_index)]
            records.append(
                {
                    "customer_index": int(customer_index),
                    "history_bucket": str(data["history_buckets"][customer_index]),
                    "rank": int(ranks[int(local_index)]),
                    "score": float(group_scores[int(local_index)]),
                    "source_count": float(row[feature_index["num_sources"]]),
                    "popularity_30d": float(row[feature_index["item_purchases_30d"]]),
                    "times_bought_item": float(row[feature_index["times_bought_item"]]),
                    "type_affinity": float(
                        row[feature_index["affinity_product_type_name"]]
                    ),
                    "garment_affinity": float(
                        row[feature_index["affinity_garment_group_name"]]
                    ),
                    "primary_source": _primary_source(row, columns),
                    "sources": [
                        source
                        for source in SOURCE_FLAGS
                        if row[columns[source]["is"]] == 1.0
                    ],
                }
            )
        offset = end
    return records


def feature_separability(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    feature_index: dict[str, int],
) -> dict[str, object]:
    """Compare buried long-tail positives with high-ranked negatives."""
    offset = 0
    buried_pos: list[np.ndarray] = []
    hard_neg: list[np.ndarray] = []
    for group_size_value in data["groups"]:
        group_size = int(group_size_value)
        end = offset + group_size
        group_scores = scores[offset:end]
        labels = data["labels"][offset:end]
        features = data["features"][offset:end]
        order = np.argsort(-group_scores)
        ranks = np.empty(group_size, dtype=np.int32)
        ranks[order] = np.arange(1, group_size + 1)
        pop = features[:, feature_index["item_purchases_30d"]]
        finite_pop = pop[np.isfinite(pop)]
        if len(finite_pop) == 0:
            offset = end
            continue
        tail_cut = float(np.quantile(finite_pop, 0.33))
        for local_index, label in enumerate(labels):
            popularity = pop[local_index]
            if not np.isfinite(popularity):
                continue
            if label > 0 and ranks[local_index] > 12 and popularity <= tail_cut:
                buried_pos.append(features[local_index])
            if label == 0 and ranks[local_index] <= 12 and popularity <= tail_cut:
                hard_neg.append(features[local_index])
        offset = end

    interesting = [
        "num_sources",
        "score_als",
        "score_two_tower",
        "item_purchases_30d",
        "affinity_product_type_name",
        "times_bought_item",
        "rank_als",
        "rank_two_tower",
    ]
    summary = {}
    if not buried_pos or not hard_neg:
        return {"buried_long_tail_positives": len(buried_pos), "hard_negatives": len(hard_neg)}
    pos = np.vstack(buried_pos)
    neg = np.vstack(hard_neg)
    for name in interesting:
        column = feature_index[name]
        pos_values = pos[:, column]
        neg_values = neg[:, column]
        summary[name] = {
            "positive_median": float(np.nanmedian(pos_values)),
            "negative_median": float(np.nanmedian(neg_values)),
            "positive_mean": float(np.nanmean(pos_values)),
            "negative_mean": float(np.nanmean(neg_values)),
        }
    summary["buried_long_tail_positives"] = len(buried_pos)
    summary["hard_negatives"] = len(hard_neg)
    return summary


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    connection = connect()
    article_ids = catalog_article_ids(connection)
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    specifications = development_specs(DEV_CUSTOMERS)
    snapshots = []
    for specification in specifications:
        print(f"Loading snapshot {specification.cutoff}...", flush=True)
        data, _relevance = load_or_build_snapshot(
            connection,
            specification,
            article_to_index,
        )
        snapshots.append(apply_baseline_budget(data))

    all_records: list[dict[str, object]] = []
    fold_summaries = []
    for validation_index in SELECTION_FOLDS:
        cutoff = SNAPSHOTS[validation_index]["cutoff"]
        print(f"Scoring fold {cutoff}...", flush=True)
        model = fit_ranker(snapshots[:validation_index])
        validation = snapshots[validation_index]
        scores = np.asarray(model.predict(validation["features"]), dtype=np.float32)
        feature_index = {
            str(name): index
            for index, name in enumerate(validation["feature_names"])
        }
        records = analyze_fold(scores, validation, feature_index)
        for record in records:
            record["cutoff"] = cutoff
        all_records.extend(records)
        fold_summaries.append(
            {
                "cutoff": cutoff,
                "retrieved_positives": len(records),
                "median_rank": float(np.median([row["rank"] for row in records])),
                "top12_rate": float(
                    np.mean([row["rank"] <= 12 for row in records])
                ),
                "separability": feature_separability(
                    scores, validation, feature_index
                ),
                "feature_importance": [
                    {
                        "feature": name,
                        "gain": float(gain),
                    }
                    for name, gain in zip(
                        [str(value) for value in validation["feature_names"]],
                        model.feature_importance(importance_type="gain"),
                    )
                ],
            }
        )

    popularities = np.asarray(
        [row["popularity_30d"] for row in all_records],
        dtype=np.float64,
    )
    finite = popularities[np.isfinite(popularities)]
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    for row in all_records:
        row["popularity_bucket"] = _quantile_bucket(
            float(row["popularity_30d"]), float(q1), float(q2)
        )
        row["source_count_bucket"] = _source_count_bucket(float(row["source_count"]))
        row["novelty"] = _novelty(
            float(row["times_bought_item"]),
            float(row["type_affinity"]),
            float(row["garment_affinity"]),
        )

    def group(key: str) -> dict[str, object]:
        buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in all_records:
            buckets[str(row[key])].append(row)
        return {name: _summarize(values) for name, values in sorted(buckets.items())}

    report = {
        "question": (
            "When a future purchase is already in the candidate pool, "
            "which positives does LambdaRank bury?"
        ),
        "folds": fold_summaries,
        "retrieved_positives": len(all_records),
        "overall": _summarize(all_records),
        "by_popularity_bucket": group("popularity_bucket"),
        "by_source_count": group("source_count_bucket"),
        "by_novelty": group("novelty"),
        "by_history_bucket": group("history_bucket"),
        "by_primary_source": group("primary_source"),
        "popularity_tertiles_30d": {
            "q33": float(q1),
            "q66": float(q2),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    output_path = OVERNIGHT_DIR / "ranking_failure_analysis.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "retrieved_positives": report["retrieved_positives"],
        "overall": report["overall"],
        "by_source_count": report["by_source_count"],
        "by_popularity_bucket": report["by_popularity_bucket"],
        "by_novelty": report["by_novelty"],
        "path": str(output_path),
    }, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
