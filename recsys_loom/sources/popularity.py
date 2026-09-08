"""Popularity-based candidate generators for retrieval evaluation.

Returns a global ranked list (same for all customers) as a plain list
of (article_id, score) tuples.  The evaluation harness broadcasts this
to all customers without materializing millions of duplicate records.
"""

from __future__ import annotations

import math

import duckdb


GlobalCandidateList = list[tuple[str, float]]


def all_history_popularity(
    con: duckdb.DuckDBPyConnection,
    train_end: str,
    k: int = 500,
) -> GlobalCandidateList:
    """Top-K articles by total purchase count through train_end."""
    return con.sql(f"""
        SELECT article_id, COUNT(*)::DOUBLE AS score
        FROM transactions
        WHERE transaction_date <= DATE '{train_end}'
        GROUP BY article_id
        ORDER BY score DESC, article_id ASC
        LIMIT {k}
    """).fetchall()


def recent_popularity(
    con: duckdb.DuckDBPyConnection,
    train_end: str,
    lookback_days: int = 7,
    k: int = 500,
) -> GlobalCandidateList:
    """Top-K articles by purchase count in the last N days of training."""
    return con.sql(f"""
        SELECT article_id, COUNT(*)::DOUBLE AS score
        FROM transactions
        WHERE transaction_date BETWEEN
            DATE '{train_end}' - INTERVAL '{lookback_days} days' + INTERVAL '1 day'
            AND DATE '{train_end}'
        GROUP BY article_id
        ORDER BY score DESC, article_id ASC
        LIMIT {k}
    """).fetchall()


def time_decayed_popularity(
    con: duckdb.DuckDBPyConnection,
    train_end: str,
    half_life_days: float = 14.0,
    k: int = 500,
) -> GlobalCandidateList:
    """Top-K articles scored by exponential time-decay from train_end."""
    decay_rate = math.log(2) / half_life_days
    return con.sql(f"""
        SELECT
            article_id,
            SUM(EXP(-{decay_rate} * (DATE '{train_end}' - transaction_date))) AS score
        FROM transactions
        WHERE transaction_date <= DATE '{train_end}'
        GROUP BY article_id
        ORDER BY score DESC, article_id ASC
        LIMIT {k}
    """).fetchall()


def global_list_to_article_ids(candidates: GlobalCandidateList) -> list[str]:
    """Extract the ordered article ID list from a global candidate list."""
    return [article_id for article_id, _ in candidates]
