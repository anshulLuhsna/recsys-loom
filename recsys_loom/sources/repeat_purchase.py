"""Repeat-purchase candidate generator.

Hypothesis: customers re-buy articles they've purchased before.
Ranked by recency-weighted purchase count per customer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import duckdb

from recsys_loom.candidates import CandidateRecord


def repeat_purchase_candidates(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    train_end: str,
    half_life_days: float = 30.0,
    k: int = 500,
) -> list[CandidateRecord]:
    """Return each customer's own previously purchased articles.

    Scored by exponential time-decay so recent re-purchases rank higher.
    """
    decay_rate = math.log(2) / half_life_days

    rows = con.sql(f"""
        WITH scored AS (
            SELECT
                customer_id,
                article_id,
                SUM(EXP(-{decay_rate} * (DATE '{train_end}' - transaction_date))) AS score,
                COUNT(*) AS purchase_count
            FROM transactions
            WHERE transaction_date <= DATE '{train_end}'
            GROUP BY customer_id, article_id
        ), ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY score DESC, article_id ASC
                ) AS rank
            FROM scored
        )
        SELECT customer_id, article_id, rank, score
        FROM ranked
        WHERE rank <= {k}
        ORDER BY customer_id, rank
    """).fetchall()

    customer_set = set(customer_ids)
    return [
        CandidateRecord(
            customer_id=cid,
            article_id=aid,
            source_name="repeat_purchase",
            source_rank=rank,
            source_score=float(score),
            model_version=f"repeat_decay_hl{half_life_days:.0f}d",
            feature_cutoff=train_end,
        )
        for cid, aid, rank, score in rows
        if cid in customer_set
    ]
