"""Leakage-safe user visual profiles over frozen article image embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ImageEmbeddingStore:
    article_ids: list[str]
    article_to_index: dict[str, int]
    embeddings: NDArray[np.float32]
    status: NDArray[np.uint8]

    @classmethod
    def load(cls, directory: Path) -> ImageEmbeddingStore:
        article_ids_array = np.load(directory / "article_ids.npy")
        article_ids = [str(value) for value in article_ids_array]
        return cls(
            article_ids=article_ids,
            article_to_index={
                article_id: index
                for index, article_id in enumerate(article_ids)
            },
            embeddings=np.load(
                directory / "image_embeddings.f32.npy",
                mmap_mode="r",
            ),
            status=np.load(
                directory / "image_embedding_status.u8.npy",
                mmap_mode="r",
            ),
        )


@dataclass(slots=True)
class VisualProfiles:
    customer_ids: list[str]
    embeddings: NDArray[np.float32]
    history_item_indices: list[set[int]]
    customers_with_profile: NDArray[np.bool_]


def build_visual_profiles(
    con: duckdb.DuckDBPyConnection,
    store: ImageEmbeddingStore,
    customer_ids: Sequence[str],
    cutoff: str,
    max_history: int = 50,
    half_life_days: float = 45.0,
) -> VisualProfiles:
    """Aggregate only image-bearing purchases available by the cutoff."""
    ids = list(customer_ids)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE image_profile_customers "
        "(customer_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO image_profile_customers VALUES (?)",
        [(customer_id,) for customer_id in ids],
    )
    rows = con.sql(f"""
        WITH ranked AS (
            SELECT
                t.customer_id,
                t.article_id,
                (DATE '{cutoff}' - t.transaction_date)::INT AS days_ago,
                ROW_NUMBER() OVER (
                    PARTITION BY t.customer_id
                    ORDER BY t.transaction_date DESC, t.article_id
                ) AS history_rank
            FROM transactions t
            INNER JOIN image_profile_customers c USING (customer_id)
            WHERE t.transaction_date <= DATE '{cutoff}'
        )
        SELECT customer_id, article_id, days_ago, history_rank
        FROM ranked
        ORDER BY customer_id, history_rank
    """).fetchall()
    con.execute("DROP TABLE image_profile_customers")

    customer_to_index = {
        customer_id: index for index, customer_id in enumerate(ids)
    }
    profiles = np.zeros(
        (len(ids), store.embeddings.shape[1]),
        dtype=np.float32,
    )
    weight_sums = np.zeros(len(ids), dtype=np.float32)
    histories: list[set[int]] = [set() for _ in ids]
    decay = math.log(2.0) / half_life_days
    for customer_id, article_id, days_ago, history_rank in rows:
        item_index = store.article_to_index.get(article_id)
        if item_index is None or store.status[item_index] != 1:
            continue
        customer_index = customer_to_index[customer_id]
        histories[customer_index].add(item_index)
        if int(history_rank) > max_history:
            continue
        weight = math.exp(-decay * max(float(days_ago), 0.0))
        profiles[customer_index] += store.embeddings[item_index] * weight
        weight_sums[customer_index] += weight

    available = weight_sums > 0
    profiles[available] /= weight_sums[available, None]
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles[available] /= np.maximum(norms[available], 1e-12)
    return VisualProfiles(
        customer_ids=ids,
        embeddings=profiles,
        history_item_indices=histories,
        customers_with_profile=available,
    )


def eligible_image_item_indices(
    con: duckdb.DuckDBPyConnection,
    store: ImageEmbeddingStore,
    cutoff: str,
) -> NDArray[np.int64]:
    rows = con.sql(f"""
        SELECT DISTINCT article_id
        FROM transactions
        WHERE transaction_date <= DATE '{cutoff}'
        ORDER BY article_id
    """).fetchall()
    return np.asarray(
        [
            store.article_to_index[row[0]]
            for row in rows
            if row[0] in store.article_to_index
            and store.status[store.article_to_index[row[0]]] == 1
        ],
        dtype=np.int64,
    )


def filter_history_from_ann(
    article_indices: NDArray[np.int64],
    scores: NDArray[np.float32],
    histories: list[set[int]],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    filtered_indices = np.full(
        (len(histories), k),
        -1,
        dtype=np.int64,
    )
    filtered_scores = np.full(
        (len(histories), k),
        -np.inf,
        dtype=np.float32,
    )
    for customer_index, history in enumerate(histories):
        write_index = 0
        for item_index, score in zip(
            article_indices[customer_index],
            scores[customer_index],
        ):
            if int(item_index) < 0 or int(item_index) in history:
                continue
            filtered_indices[customer_index, write_index] = int(item_index)
            filtered_scores[customer_index, write_index] = float(score)
            write_index += 1
            if write_index == k:
                break
    return filtered_indices, filtered_scores
