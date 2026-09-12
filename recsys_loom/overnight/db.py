"""Capped DuckDB connections for overnight jobs."""

from __future__ import annotations

from pathlib import Path

import duckdb

from recsys_loom.overnight.protocol import ROOT, TMP_DIR, ensure_directories


def connect(
    memory_limit: str = "6GB",
    threads: int = 4,
) -> duckdb.DuckDBPyConnection:
    ensure_directories()
    connection = duckdb.connect()
    connection.execute(f"SET threads = {int(threads)}")
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute(f"SET temp_directory = '{TMP_DIR}'")
    transactions = ROOT / "transactions_train.csv"
    articles = ROOT / "articles.csv"
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id
        FROM read_csv('{transactions}', header=true, all_varchar=true)
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT * FROM read_csv('{articles}', header=true, all_varchar=true)
    """)
    return connection


def catalog_article_ids(connection: duckdb.DuckDBPyConnection) -> list[str]:
    rows = connection.sql(
        "SELECT article_id FROM articles ORDER BY article_id"
    ).fetchall()
    return [row[0] for row in rows]


def write_id_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    values: list[str],
) -> None:
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE {table_name} (customer_id VARCHAR)"
    )
    connection.executemany(
        f"INSERT INTO {table_name} VALUES (?)",
        [(value,) for value in values],
    )
