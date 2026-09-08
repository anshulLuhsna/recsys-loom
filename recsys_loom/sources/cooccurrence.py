"""Item-item co-purchase candidate generator.

Uses same-day co-purchase pairs scored by PMI. The self-join is bounded
by restricting to the last N days of training data, which keeps the
intermediate result manageable while capturing recent purchase patterns.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb

from recsys_loom.candidates import CandidateRecord


def cooccurrence_candidates(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    train_end: str,
    pair_lookback_days: int = 90,
    neighbors_per_article: int = 50,
    candidates_per_customer: int = 500,
    min_co_count: int = 5,
) -> list[CandidateRecord]:
    """For each customer, retrieve PMI neighbors of their purchased articles.

    The co-purchase self-join is restricted to the last pair_lookback_days
    to keep it bounded.  Full history is used for customer lookup.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE recent_baskets AS
        SELECT customer_id, transaction_date, article_id
        FROM transactions
        WHERE transaction_date BETWEEN
            DATE '{train_end}' - INTERVAL '{pair_lookback_days} days' + INTERVAL '1 day'
            AND DATE '{train_end}'
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE item_pairs AS
        WITH pairs AS (
            SELECT
                a.article_id AS article_a,
                b.article_id AS article_b,
                COUNT(*) AS co_count
            FROM recent_baskets a
            INNER JOIN recent_baskets b
                ON a.customer_id = b.customer_id
                AND a.transaction_date = b.transaction_date
                AND a.article_id < b.article_id
            GROUP BY a.article_id, b.article_id
            HAVING co_count >= {min_co_count}
        ), marginals AS (
            SELECT article_id, COUNT(*) AS freq
            FROM recent_baskets
            GROUP BY article_id
        ), total AS (
            SELECT COUNT(*)::DOUBLE AS n FROM recent_baskets
        )
        SELECT
            p.article_a,
            p.article_b,
            p.co_count,
            LN(p.co_count * t.n / (ma.freq * mb.freq)) AS pmi
        FROM pairs p
        CROSS JOIN total t
        INNER JOIN marginals ma ON ma.article_id = p.article_a
        INNER JOIN marginals mb ON mb.article_id = p.article_b
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE top_neighbors AS
        WITH bidirectional AS (
            SELECT article_a AS seed, article_b AS neighbor, pmi
            FROM item_pairs
            UNION ALL
            SELECT article_b AS seed, article_a AS neighbor, pmi
            FROM item_pairs
        ), ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY seed ORDER BY pmi DESC, neighbor ASC
                ) AS rank
            FROM bidirectional
        )
        SELECT seed, neighbor, pmi, rank
        FROM ranked
        WHERE rank <= {neighbors_per_article}
    """)

    con.execute("CREATE OR REPLACE TEMP TABLE eval_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO eval_customers VALUES (?)",
        [(c,) for c in customer_ids],
    )

    rows = con.sql(f"""
        WITH eval_history AS (
            SELECT DISTINCT t.customer_id, t.article_id
            FROM transactions t
            INNER JOIN eval_customers ec ON ec.customer_id = t.customer_id
            WHERE t.transaction_date <= DATE '{train_end}'
        ), expanded AS (
            SELECT
                eh.customer_id,
                tn.neighbor AS article_id,
                SUM(tn.pmi) AS agg_score
            FROM eval_history eh
            INNER JOIN top_neighbors tn ON tn.seed = eh.article_id
            LEFT JOIN eval_history already
                ON already.customer_id = eh.customer_id
                AND already.article_id = tn.neighbor
            WHERE already.article_id IS NULL
            GROUP BY eh.customer_id, tn.neighbor
        ), final_ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY agg_score DESC, article_id ASC
                ) AS rank
            FROM expanded
        )
        SELECT customer_id, article_id, rank, agg_score
        FROM final_ranked
        WHERE rank <= {candidates_per_customer}
        ORDER BY customer_id, rank
    """).fetchall()

    con.execute("DROP TABLE IF EXISTS eval_customers")
    con.execute("DROP TABLE IF EXISTS recent_baskets")
    con.execute("DROP TABLE IF EXISTS item_pairs")
    con.execute("DROP TABLE IF EXISTS top_neighbors")

    customer_set = set(customer_ids)
    return [
        CandidateRecord(
            customer_id=cid,
            article_id=aid,
            source_name="cooccurrence",
            source_rank=rank,
            source_score=float(score),
            model_version=f"pmi_{pair_lookback_days}d_top{neighbors_per_article}_min{min_co_count}",
            feature_cutoff=train_end,
        )
        for cid, aid, rank, score in rows
        if cid in customer_set
    ]
