"""Content/metadata-neighbor candidate generator using vector retrieval.

Architecture:
    represent once → index once → retrieve top-K

Instead of comparing every customer against every article in SQL (68K × 105K
= 7 billion comparisons), we:

1. Encode each article as a multi-hot vector from its categorical features.
   Each unique attribute value (e.g. "Black", "Vest top") is one dimension.
   An article with 8 attributes becomes a sparse vector with 8 non-zero entries
   out of ~735 total dimensions.

2. Encode each customer as the recency-weighted average of their purchased
   article vectors. This summarizes their taste in the same vector space.

3. Build a FAISS index over all article vectors. FAISS is optimized C++ —
   it finds nearest neighbors across 105K vectors in seconds.

4. For each customer, query the index for top-K nearest articles by cosine
   similarity. Filter out already-purchased articles.

This is the same architecture used by production embedding-based retrieval.
When text/image embeddings replace these categorical vectors later, the
retrieval code stays the same — only the representation changes.

Categorical features used:
    product_type_name, product_group_name, colour_group_name, section_name,
    garment_group_name, department_name, index_name, graphical_appearance_name
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import duckdb
import numpy as np
from numpy.typing import NDArray

from recsys_loom.candidates import CandidateRecord

CONTENT_FEATURES = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "department_name",
    "index_name",
    "graphical_appearance_name",
]


def _build_vocabulary(
    con: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, dict[str, int]], int]:
    """Build a global vocabulary mapping (feature, value) → dimension index."""
    vocab: dict[str, dict[str, int]] = {}
    offset = 0
    for feature in CONTENT_FEATURES:
        rows = con.sql(f"""
            SELECT DISTINCT {feature} AS val
            FROM articles
            WHERE {feature} IS NOT NULL AND TRIM({feature}) != ''
            ORDER BY val
        """).fetchall()
        vocab[feature] = {}
        for (val,) in rows:
            vocab[feature][val] = offset
            offset += 1
    return vocab, offset


def _encode_articles(
    con: duckdb.DuckDBPyConnection,
    vocab: dict[str, dict[str, int]],
    ndim: int,
) -> tuple[list[str], NDArray[np.float32]]:
    """Encode all articles as normalized multi-hot vectors."""
    cols = ", ".join(CONTENT_FEATURES)
    rows = con.sql(f"SELECT article_id, {cols} FROM articles").fetchall()

    article_ids: list[str] = []
    matrix = np.zeros((len(rows), ndim), dtype=np.float32)

    for i, row in enumerate(rows):
        article_ids.append(row[0])
        for j, feature in enumerate(CONTENT_FEATURES):
            val = row[j + 1]
            if val and val.strip() and feature in vocab and val in vocab[feature]:
                matrix[i, vocab[feature][val]] = 1.0

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    return article_ids, matrix


def _encode_customers(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    train_end: str,
    article_id_to_idx: dict[str, int],
    article_matrix: NDArray[np.float32],
    half_life_days: float,
) -> NDArray[np.float32]:
    """Encode customers as recency-weighted averages of purchased article vectors."""
    ndim = article_matrix.shape[1]
    decay_rate = math.log(2) / half_life_days

    con.execute("CREATE OR REPLACE TEMP TABLE content_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO content_customers VALUES (?)",
        [(c,) for c in customer_ids],
    )

    customer_idx = {cid: i for i, cid in enumerate(customer_ids)}
    customer_matrix = np.zeros((len(customer_ids), ndim), dtype=np.float32)

    rows = con.sql(f"""
        SELECT t.customer_id, t.article_id,
               (DATE '{train_end}' - t.transaction_date)::INT AS days_ago
        FROM transactions t
        INNER JOIN content_customers cc ON cc.customer_id = t.customer_id
        WHERE t.transaction_date <= DATE '{train_end}'
    """).fetchall()

    con.execute("DROP TABLE IF EXISTS content_customers")

    for cid, aid, days_ago in rows:
        if aid not in article_id_to_idx:
            continue
        weight = math.exp(-decay_rate * float(days_ago))
        customer_matrix[customer_idx[cid]] += weight * article_matrix[article_id_to_idx[aid]]

    norms = np.linalg.norm(customer_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    customer_matrix /= norms

    return customer_matrix


def _load_purchased(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    train_end: str,
) -> dict[str, set[str]]:
    """Load each customer's already-purchased article set."""
    con.execute("CREATE OR REPLACE TEMP TABLE content_purch_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO content_purch_customers VALUES (?)",
        [(c,) for c in customer_ids],
    )

    rows = con.sql(f"""
        SELECT t.customer_id, t.article_id
        FROM transactions t
        INNER JOIN content_purch_customers cpc ON cpc.customer_id = t.customer_id
        WHERE t.transaction_date <= DATE '{train_end}'
    """).fetchall()

    con.execute("DROP TABLE IF EXISTS content_purch_customers")

    purchased: dict[str, set[str]] = {cid: set() for cid in customer_ids}
    for cid, aid in rows:
        purchased[cid].add(aid)
    return purchased


def content_candidates(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    train_end: str,
    half_life_days: float = 30.0,
    k: int = 500,
    retrieval_oversample: int = 100,
) -> list[CandidateRecord]:
    """Retrieve content-similar candidates using FAISS vector retrieval.

    Args:
        retrieval_oversample: extra candidates to retrieve before filtering
            out already-purchased items, so we still have ~k after filtering.
    """
    vocab, ndim = _build_vocabulary(con)
    article_ids, article_matrix = _encode_articles(con, vocab, ndim)
    article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    customer_matrix = _encode_customers(
        con, customer_ids, train_end,
        article_id_to_idx, article_matrix, half_life_days,
    )

    fetch_k = min(k + retrieval_oversample, len(article_ids))
    similarity = customer_matrix @ article_matrix.T
    indices_batch = np.argsort(-similarity, axis=1)[:, :fetch_k]
    scores_batch = np.take_along_axis(similarity, indices_batch, axis=1)

    purchased = _load_purchased(con, customer_ids, train_end)

    records: list[CandidateRecord] = []
    for i, cid in enumerate(customer_ids):
        rank = 0
        already_bought = purchased.get(cid, set())
        for j in range(fetch_k):
            idx = int(indices_batch[i, j])
            if idx < 0:
                continue
            aid = article_ids[idx]
            if aid in already_bought:
                continue
            rank += 1
            if rank > k:
                break
            records.append(CandidateRecord(
                customer_id=cid,
                article_id=aid,
                source_name="content",
                source_rank=rank,
                source_score=float(scores_batch[i, j]),
                model_version=f"multihot_{ndim}d_hl{half_life_days:.0f}d",
                feature_cutoff=train_end,
            ))

    return records
