"""Weighted ALS collaborative filtering candidate generator.

Based on Hu, Koren, Volinsky (2008) — Collaborative Filtering for
Implicit Feedback Datasets. Purchases express confidence, not ratings.

Uses the `implicit` library for the ALS solver.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import numpy as np
from scipy.sparse import csr_matrix

from recsys_loom.candidates import CandidateRecord

try:
    from implicit.als import AlternatingLeastSquares
except ImportError as exc:
    raise ImportError("pip install implicit") from exc


def als_candidates(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    train_end: str,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 15,
    confidence_weight: float = 40.0,
    k: int = 500,
    random_state: int = 42,
) -> list[CandidateRecord]:
    """Train ALS on purchase counts and retrieve top-K per customer.

    Confidence = 1 + alpha * count, following the original paper.
    """
    rows = con.sql(f"""
        SELECT customer_id, article_id, COUNT(*) AS purchase_count
        FROM transactions
        WHERE transaction_date <= DATE '{train_end}'
        GROUP BY customer_id, article_id
    """).fetchall()

    customer_set = set(customer_ids)

    all_customers = sorted({r[0] for r in rows})
    all_articles = sorted({r[1] for r in rows})
    customer_to_idx = {c: i for i, c in enumerate(all_customers)}
    article_to_idx = {a: i for i, a in enumerate(all_articles)}
    idx_to_article = {i: a for a, i in article_to_idx.items()}

    row_indices = []
    col_indices = []
    values = []
    for cid, aid, count in rows:
        row_indices.append(customer_to_idx[cid])
        col_indices.append(article_to_idx[aid])
        values.append(float(count))

    user_item = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(all_customers), len(all_articles)),
    )

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    confidence = user_item * confidence_weight
    model.fit(confidence)

    eval_indices = [
        customer_to_idx[c] for c in customer_ids if c in customer_to_idx
    ]
    cold_customers = [c for c in customer_ids if c not in customer_to_idx]

    records: list[CandidateRecord] = []

    if eval_indices:
        ids_arr, scores_arr = model.recommend(
            eval_indices,
            user_item[eval_indices],
            N=k,
            filter_already_liked_items=False,
        )

        for i, user_idx in enumerate(eval_indices):
            cid = all_customers[user_idx]
            for rank, (article_idx, score) in enumerate(
                zip(ids_arr[i], scores_arr[i]), start=1
            ):
                records.append(CandidateRecord(
                    customer_id=cid,
                    article_id=idx_to_article[int(article_idx)],
                    source_name="als",
                    source_rank=rank,
                    source_score=float(score),
                    model_version=f"als_f{factors}_r{regularization}_i{iterations}_a{confidence_weight:.0f}",
                    feature_cutoff=train_end,
                ))

    for cid in cold_customers:
        pass

    return records
