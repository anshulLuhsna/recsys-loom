"""Leakage-safe global-popularity baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import duckdb

from recsys_loom.metrics import ranking_metrics_at_k


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def run_popularity_baseline(
    transactions_path: Path,
    articles_path: Path,
    output_directory: Path,
    train_end: str,
    validation_start: str,
    validation_end: str,
    training_start: str | None = None,
    k: int = 12,
    threads: int = 2,
) -> dict[str, Any]:
    """Fit global purchase-count popularity and score the hidden validation week."""
    output_directory.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute("SET memory_limit = '6GB'")

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id
        FROM read_csv(
            '{_sql_path(transactions_path)}',
            header = true,
            all_varchar = true
        )
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT article_id
        FROM read_csv(
            '{_sql_path(articles_path)}',
            header = true,
            all_varchar = true
        )
    """)

    training_filter = f"transaction_date <= DATE '{train_end}'"
    if training_start is not None:
        training_filter = (
            f"transaction_date BETWEEN DATE '{training_start}' AND DATE '{train_end}'"
        )

    top_rows = con.sql(f"""
        SELECT article_id, COUNT(*) AS purchase_count
        FROM transactions
        WHERE {training_filter}
        GROUP BY article_id
        ORDER BY purchase_count DESC, article_id ASC
        LIMIT {k}
    """).fetchall()
    top_articles = [
        {"article_id": article_id, "purchase_count": purchase_count}
        for article_id, purchase_count in top_rows
    ]
    prediction = [row["article_id"] for row in top_articles]

    validation_rows = con.sql(f"""
        WITH train_customers AS (
            SELECT DISTINCT customer_id
            FROM transactions
            WHERE transaction_date <= DATE '{train_end}'
        ), validation_labels AS (
            SELECT DISTINCT customer_id, article_id
            FROM transactions
            WHERE transaction_date BETWEEN
                  DATE '{validation_start}' AND DATE '{validation_end}'
        )
        SELECT
            v.customer_id,
            LIST(v.article_id ORDER BY v.article_id) AS relevant_articles,
            t.customer_id IS NULL AS is_cold_customer
        FROM validation_labels v
        LEFT JOIN train_customers t USING (customer_id)
        GROUP BY v.customer_id, t.customer_id
        ORDER BY v.customer_id
    """).fetchall()

    catalog_size = con.sql("SELECT COUNT(*) FROM articles").fetchone()[0]
    relevant_by_customer = {
        customer_id: set(relevant_articles)
        for customer_id, relevant_articles, _ in validation_rows
    }
    predictions_by_customer = {
        customer_id: prediction for customer_id in relevant_by_customer
    }
    cold_relevance = {
        customer_id: set(relevant_articles)
        for customer_id, relevant_articles, is_cold in validation_rows
        if is_cold
    }
    warm_relevance = {
        customer_id: set(relevant_articles)
        for customer_id, relevant_articles, is_cold in validation_rows
        if not is_cold
    }

    result: dict[str, Any] = {
        "definition": {
            "score": "training purchase row count",
            "training_start": training_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "k": k,
            "tie_break": "article_id ascending",
            "validation_labels": "unique customer_id and article_id pairs",
        },
        "top_articles": top_articles,
        "aggregate": ranking_metrics_at_k(
            relevant_by_customer,
            predictions_by_customer,
            catalog_size,
            k,
        ),
        "warm_customers": ranking_metrics_at_k(
            warm_relevance,
            predictions_by_customer,
            catalog_size,
            k,
        ),
        "cold_customers": ranking_metrics_at_k(
            cold_relevance,
            predictions_by_customer,
            catalog_size,
            k,
        ),
        "catalog_size": catalog_size,
        "unique_validation_labels": sum(map(len, relevant_by_customer.values())),
    }

    with (output_directory / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_id", "prediction"])
        joined_prediction = " ".join(prediction)
        for customer_id in relevant_by_customer:
            writer.writerow([customer_id, joined_prediction])

    with (output_directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    con.close()
    return result
