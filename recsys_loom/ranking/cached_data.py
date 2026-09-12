"""Shared candidate and feature caches for controlled ranking experiments."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from numpy.typing import NDArray

from recsys_loom.candidates import CandidateRecord
from recsys_loom.ranking.features import CATEGORY_FEATURES, build_features
from recsys_loom.ranking.ranker import build_groups, build_labels

BASE_SOURCES = [
    "recent_7d_pop",
    "repeat_purchase",
    "cooccurrence",
    "als",
    "content",
]
SIX_SOURCES = [*BASE_SOURCES, "two_tower"]
SEVEN_SOURCES = [*SIX_SOURCES, "image"]


@dataclass(frozen=True, slots=True)
class SnapshotSpec:
    cutoff: str
    target_start: str
    target_end: str
    customer_count: int
    existing_candidates_path: Path
    two_tower_candidates_path: Path
    image_candidates_path: Path | None = None


@dataclass(slots=True)
class CandidatePool:
    customer_ids: list[str]
    records_by_source: dict[str, list[CandidateRecord]]
    articles_by_source: dict[str, dict[str, list[str]]]


def load_candidate_pool(specification: SnapshotSpec) -> CandidatePool:
    """Load both arms while interning repeated IDs to constrain memory."""
    records_by_source: dict[str, list[CandidateRecord]] = {
        source: [] for source in SEVEN_SOURCES
    }
    articles_by_source: dict[str, dict[str, list[str]]] = {
        source: {} for source in SEVEN_SOURCES
    }
    customer_order: list[str] = []
    seen_customers: set[str] = set()

    with gzip.open(
        specification.existing_candidates_path,
        "rt",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_name = row["source_name"]
            if source_name not in BASE_SOURCES:
                continue
            customer_id = sys.intern(row["customer_id"])
            article_id = sys.intern(row["article_id"])
            if customer_id not in seen_customers:
                seen_customers.add(customer_id)
                customer_order.append(customer_id)
            record = CandidateRecord(
                customer_id=customer_id,
                article_id=article_id,
                source_name=source_name,
                source_rank=int(row["rank"]),
                source_score=float(row["score"]),
                feature_cutoff=row.get("feature_cutoff", specification.cutoff),
            )
            records_by_source[source_name].append(record)
            articles_by_source[source_name].setdefault(customer_id, []).append(
                article_id
            )

    with gzip.open(
        specification.two_tower_candidates_path,
        "rt",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            customer_id = sys.intern(row["customer_id"])
            article_id = sys.intern(row["article_id"])
            if customer_id not in seen_customers:
                continue
            rank_value = row.get("rank", row.get("two_tower_rank", "0"))
            score_value = row.get("score", row.get("two_tower_score", "0"))
            record = CandidateRecord(
                customer_id=customer_id,
                article_id=article_id,
                source_name="two_tower",
                source_rank=int(rank_value),
                source_score=float(score_value),
                model_version="two_tower_metadata_history_64d_v1",
                feature_cutoff=row.get("feature_cutoff", specification.cutoff),
            )
            records_by_source["two_tower"].append(record)
            articles_by_source["two_tower"].setdefault(customer_id, []).append(
                article_id
            )

    if specification.image_candidates_path is not None:
        with gzip.open(
            specification.image_candidates_path,
            "rt",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                customer_id = sys.intern(row["customer_id"])
                article_id = sys.intern(row["article_id"])
                if customer_id not in seen_customers:
                    continue
                record = CandidateRecord(
                    customer_id=customer_id,
                    article_id=article_id,
                    source_name="image",
                    source_rank=int(row["rank"]),
                    source_score=float(row["score"]),
                    model_version="dinov2_small_visual_profile_v1",
                    feature_cutoff=row.get(
                        "feature_cutoff",
                        specification.cutoff,
                    ),
                )
                records_by_source["image"].append(record)
                articles_by_source["image"].setdefault(
                    customer_id,
                    [],
                ).append(article_id)

    return CandidatePool(
        customer_ids=customer_order,
        records_by_source=records_by_source,
        articles_by_source=articles_by_source,
    )


def load_relevance(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    target_start: str,
    target_end: str,
) -> dict[str, set[str]]:
    con.execute("CREATE OR REPLACE TEMP TABLE cached_eval_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO cached_eval_customers VALUES (?)",
        [(customer_id,) for customer_id in customer_ids],
    )
    rows = con.sql(f"""
        SELECT t.customer_id, t.article_id
        FROM transactions t
        INNER JOIN cached_eval_customers c USING (customer_id)
        WHERE t.transaction_date BETWEEN DATE '{target_start}' AND DATE '{target_end}'
        GROUP BY t.customer_id, t.article_id
    """).fetchall()
    con.execute("DROP TABLE cached_eval_customers")

    relevance = {customer_id: set() for customer_id in customer_ids}
    for customer_id, article_id in rows:
        relevance[customer_id].add(article_id)
    return relevance


def build_union(
    pool: CandidatePool,
    source_names: Sequence[str],
    k_per_source: int = 500,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for customer_id in pool.customer_ids:
        seen: set[str] = set()
        articles: list[str] = []
        for source_name in source_names:
            source_articles = pool.articles_by_source[source_name].get(
                customer_id, []
            )
            for article_id in source_articles[:k_per_source]:
                if article_id not in seen:
                    seen.add(article_id)
                    articles.append(article_id)
        result[customer_id] = articles
    return result


def candidate_pool_metrics(
    relevance: dict[str, set[str]],
    candidates: dict[str, list[str]],
) -> dict[str, float | int]:
    customer_count = len(relevance)
    total_relevant = sum(len(items) for items in relevance.values())
    hits = 0
    customers_with_hit = 0
    macro_recalls: list[float] = []
    candidate_counts: list[int] = []

    for customer_id, relevant in relevance.items():
        proposed = set(candidates.get(customer_id, []))
        customer_hits = len(relevant.intersection(proposed))
        hits += customer_hits
        customers_with_hit += int(customer_hits > 0)
        macro_recalls.append(customer_hits / len(relevant) if relevant else 0.0)
        candidate_counts.append(len(proposed))

    return {
        "customers": customer_count,
        "matched_relevance_pairs": hits,
        "micro_recall": hits / total_relevant if total_relevant else 0.0,
        "macro_recall": float(np.mean(macro_recalls)) if macro_recalls else 0.0,
        "hit_rate": customers_with_hit / customer_count if customer_count else 0.0,
        "average_candidate_count": (
            float(np.mean(candidate_counts)) if candidate_counts else 0.0
        ),
    }


def oracle_map_at_12(
    relevance: dict[str, set[str]],
    candidates: dict[str, list[str]],
) -> float:
    values: list[float] = []
    for customer_id, relevant in relevance.items():
        hits = relevant.intersection(candidates.get(customer_id, []))
        denominator = min(len(relevant), 12)
        values.append(min(len(hits), 12) / denominator if denominator else 0.0)
    return float(np.mean(values)) if values else 0.0


def build_feature_cache(
    con: duckdb.DuckDBPyConnection,
    pool: CandidatePool,
    relevance: dict[str, set[str]],
    cutoff: str,
    source_names: Sequence[str],
    catalog_article_to_index: dict[str, int],
    output_path: Path,
) -> dict[str, object]:
    data = build_feature_data(
        con,
        pool,
        relevance,
        cutoff,
        source_names,
        catalog_article_to_index,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data)
    return {
        "path": str(output_path),
        "pairs": len(data["labels"]),
        "positives": int(data["labels"].sum()),
        "groups": len(data["groups"]),
        "features": len(data["feature_names"]),
    }


def build_feature_data(
    con: duckdb.DuckDBPyConnection,
    pool: CandidatePool,
    relevance: dict[str, set[str]],
    cutoff: str,
    source_names: Sequence[str],
    catalog_article_to_index: dict[str, int],
) -> dict[str, NDArray]:
    """Build compact numeric pair data without persisting a large cache."""
    selected_records = {
        source_name: pool.records_by_source[source_name]
        for source_name in source_names
    }
    feature_names, features, pairs = build_features(
        con,
        pool.customer_ids,
        selected_records,
        cutoff,
        500,
        source_names=source_names,
    )
    labels = build_labels(pairs, relevance)
    groups, _ = build_groups(pairs)
    customer_to_index = {
        customer_id: index for index, customer_id in enumerate(pool.customer_ids)
    }
    pair_customer_indices = np.fromiter(
        (customer_to_index[customer_id] for customer_id, _ in pairs),
        dtype=np.int32,
        count=len(pairs),
    )
    pair_article_indices = np.fromiter(
        (catalog_article_to_index[article_id] for _, article_id in pairs),
        dtype=np.int32,
        count=len(pairs),
    )
    article_columns = ", ".join(CATEGORY_FEATURES)
    article_rows = con.sql(
        f"SELECT article_id, {article_columns} FROM articles ORDER BY article_id"
    ).fetchall()
    vocabularies: list[dict[str, int]] = [
        {} for _ in CATEGORY_FEATURES
    ]
    article_categories: dict[str, list[int]] = {}
    for row in article_rows:
        encoded: list[int] = []
        for field_index in range(len(CATEGORY_FEATURES)):
            value = row[field_index + 1] or ""
            vocabulary = vocabularies[field_index]
            if value and value not in vocabulary:
                vocabulary[value] = len(vocabulary) + 1
            encoded.append(vocabulary.get(value, 0))
        article_categories[row[0]] = encoded
    pair_categories = np.asarray(
        [article_categories[article_id] for _, article_id in pairs],
        dtype=np.int32,
    )
    return {
        "features": features,
        "labels": labels.astype(np.int8),
        "groups": groups,
        "pair_customer_indices": pair_customer_indices,
        "pair_article_indices": pair_article_indices,
        "pair_categories": pair_categories,
        "customer_ids": np.asarray(pool.customer_ids),
        "feature_names": np.asarray(feature_names),
        "category_cardinalities": np.asarray(
            [len(vocabulary) for vocabulary in vocabularies],
            dtype=np.int32,
        ),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
