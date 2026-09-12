#!/usr/bin/env python3
"""Run the leakage-aware H&M data audit and write reproducible summaries."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "data_audit"
REPORT_PATH = ROOT / "reports" / "data_audit.md"
TRAIN_END = "2020-09-15"
VALIDATION_START = "2020-09-16"
VALIDATION_END = "2020-09-22"


def normalize(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 10)
    return value


def markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "NULL"
        return str(normalize(value)).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def missingness_query(table: str, columns: list[str]) -> str:
    aggregates = ["COUNT(*) AS total_rows"]
    for index, column in enumerate(columns):
        identifier = quote_identifier(column)
        aggregates.append(
            f"COUNT(*) FILTER (WHERE {identifier} IS NULL OR TRIM({identifier}) = '') AS missing_{index}"
        )
    rows = []
    for index, column in enumerate(columns):
        escaped_column = column.replace("'", "''")
        rows.append(
            f"SELECT '{escaped_column}' AS column_name, missing_{index} AS missing_rows, "
            f"total_rows AS rows, ROUND(100.0 * missing_{index} / total_rows, 4) AS missing_percent "
            "FROM counts"
        )
    return "WITH counts AS (SELECT " + ", ".join(aggregates) + f" FROM {table}) " + " UNION ALL ".join(rows)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '6GB'")
    con.execute("PRAGMA enable_progress_bar")

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW transactions AS
        SELECT
            TRY_CAST(t_dat AS DATE) AS transaction_date,
            customer_id,
            article_id,
            TRY_CAST(price AS DOUBLE) AS price,
            TRY_CAST(sales_channel_id AS INTEGER) AS sales_channel_id,
            t_dat AS raw_date,
            price AS raw_price,
            sales_channel_id AS raw_sales_channel_id
        FROM read_csv(
            '{ROOT / 'transactions_train.csv'}',
            header = true,
            all_varchar = true
        )
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW customers AS
        SELECT *
        FROM read_csv(
            '{ROOT / 'customers.csv'}',
            header = true,
            all_varchar = true
        )
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT *
        FROM read_csv(
            '{ROOT / 'articles.csv'}',
            header = true,
            all_varchar = true
        )
    """)

    with (ROOT / "customers.csv").open(newline="", encoding="utf-8") as handle:
        customer_columns = next(csv.reader(handle))
    with (ROOT / "articles.csv").open(newline="", encoding="utf-8") as handle:
        article_columns = next(csv.reader(handle))

    queries = {
        "dataset_overview": """
            SELECT 'transactions' AS dataset, COUNT(*) AS rows,
                   COUNT(DISTINCT customer_id) AS distinct_primary_entity,
                   COUNT(DISTINCT article_id) AS distinct_secondary_entity
            FROM transactions
            UNION ALL
            SELECT 'customers', COUNT(*), COUNT(DISTINCT customer_id), NULL
            FROM customers
            UNION ALL
            SELECT 'articles', COUNT(*), COUNT(DISTINCT article_id), NULL
            FROM articles
        """,
        "transaction_quality": """
            SELECT
                MIN(transaction_date) AS minimum_date,
                MAX(transaction_date) AS maximum_date,
                COUNT(*) FILTER (WHERE transaction_date IS NULL) AS invalid_dates,
                COUNT(*) FILTER (WHERE customer_id IS NULL OR TRIM(customer_id) = '') AS missing_customers,
                COUNT(*) FILTER (WHERE article_id IS NULL OR TRIM(article_id) = '') AS missing_articles,
                COUNT(*) FILTER (WHERE price IS NULL) AS invalid_prices,
                COUNT(*) FILTER (WHERE price <= 0) AS nonpositive_prices,
                COUNT(*) FILTER (WHERE sales_channel_id IS NULL) AS invalid_channels,
                MIN(LENGTH(customer_id)) AS minimum_customer_id_length,
                MAX(LENGTH(customer_id)) AS maximum_customer_id_length,
                MIN(LENGTH(article_id)) AS minimum_article_id_length,
                MAX(LENGTH(article_id)) AS maximum_article_id_length
            FROM transactions
        """,
        "catalog_key_quality": """
            SELECT 'customers' AS dataset,
                   COUNT(*) AS rows,
                   COUNT(DISTINCT customer_id) AS distinct_keys,
                   COUNT(*) FILTER (WHERE customer_id IS NULL OR TRIM(customer_id) = '') AS missing_keys
            FROM customers
            UNION ALL
            SELECT 'articles', COUNT(*), COUNT(DISTINCT article_id),
                   COUNT(*) FILTER (WHERE article_id IS NULL OR TRIM(article_id) = '')
            FROM articles
        """,
        "customer_missingness": missingness_query("customers", customer_columns),
        "article_missingness": missingness_query("articles", article_columns),
        "customer_profile_quality": """
            SELECT
                COUNT(*) FILTER (WHERE age IS NOT NULL AND TRY_CAST(age AS INTEGER) IS NULL) AS invalid_ages,
                MIN(TRY_CAST(age AS INTEGER)) AS minimum_age,
                QUANTILE_CONT(TRY_CAST(age AS INTEGER), 0.25) AS p25_age,
                MEDIAN(TRY_CAST(age AS INTEGER)) AS median_age,
                QUANTILE_CONT(TRY_CAST(age AS INTEGER), 0.75) AS p75_age,
                MAX(TRY_CAST(age AS INTEGER)) AS maximum_age,
                COUNT(*) FILTER (WHERE TRY_CAST(age AS INTEGER) < 16) AS ages_below_16,
                COUNT(*) FILTER (WHERE TRY_CAST(age AS INTEGER) > 100) AS ages_above_100,
                COUNT(DISTINCT postal_code) AS distinct_postal_codes
            FROM customers
        """,
        "customer_category_domains": """
            WITH values AS (
                SELECT 'FN' AS field, COALESCE(FN, '<NULL>') AS value FROM customers
                UNION ALL
                SELECT 'Active', COALESCE(Active, '<NULL>') FROM customers
                UNION ALL
                SELECT 'club_member_status', COALESCE(club_member_status, '<NULL>') FROM customers
                UNION ALL
                SELECT 'fashion_news_frequency', COALESCE(fashion_news_frequency, '<NULL>') FROM customers
            ), counts AS (
                SELECT field, value, COUNT(*) AS customers
                FROM values
                GROUP BY field, value
            )
            SELECT field, value, customers,
                   ROUND(100.0 * customers / SUM(customers) OVER (PARTITION BY field), 4) AS percent
            FROM counts
            ORDER BY field, customers DESC, value
        """,
        "article_catalog_shape": """
            SELECT COUNT(DISTINCT product_code) AS product_codes,
                   COUNT(DISTINCT product_type_no) AS product_types,
                   COUNT(DISTINCT product_group_name) AS product_groups,
                   COUNT(DISTINCT department_no) AS departments,
                   COUNT(DISTINCT section_no) AS sections,
                   COUNT(DISTINCT garment_group_no) AS garment_groups,
                   COUNT(DISTINCT colour_group_code) AS colour_groups,
                   COUNT(*) FILTER (WHERE detail_desc IS NULL OR TRIM(detail_desc) = '') AS missing_descriptions
            FROM articles
        """,
        "foreign_key_coverage": """
            WITH transaction_customers AS (
                SELECT DISTINCT customer_id FROM transactions
            ), transaction_articles AS (
                SELECT DISTINCT article_id FROM transactions
            )
            SELECT 'customer_id' AS key,
                   COUNT(*) AS transaction_keys,
                   COUNT(*) FILTER (WHERE c.customer_id IS NULL) AS missing_from_catalog
            FROM transaction_customers t
            LEFT JOIN customers c USING (customer_id)
            UNION ALL
            SELECT 'article_id', COUNT(*),
                   COUNT(*) FILTER (WHERE a.article_id IS NULL)
            FROM transaction_articles t
            LEFT JOIN articles a USING (article_id)
        """,
        "sales_channels": """
            SELECT sales_channel_id, COUNT(*) AS transactions,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS percent
            FROM transactions
            GROUP BY sales_channel_id
            ORDER BY sales_channel_id
        """,
        "price_distribution": """
            SELECT
                MIN(price) AS minimum,
                QUANTILE_CONT(price, 0.25) AS p25,
                MEDIAN(price) AS median,
                AVG(price) AS mean,
                QUANTILE_CONT(price, 0.75) AS p75,
                QUANTILE_CONT(price, 0.99) AS p99,
                MAX(price) AS maximum
            FROM transactions
        """,
        "duplicate_summary": """
            WITH duplicate_groups AS (
                SELECT transaction_date, customer_id, article_id, price, sales_channel_id,
                       COUNT(*) AS occurrences
                FROM transactions
                GROUP BY ALL
                HAVING COUNT(*) > 1
            )
            SELECT COUNT(*) AS duplicate_groups,
                   SUM(occurrences) AS rows_in_duplicate_groups,
                   SUM(occurrences - 1) AS excess_duplicate_rows,
                   MAX(occurrences) AS largest_group
            FROM duplicate_groups
        """,
        "duplicate_multiplicity": """
            WITH duplicate_groups AS (
                SELECT transaction_date, customer_id, article_id, price, sales_channel_id,
                       COUNT(*) AS occurrences
                FROM transactions
                GROUP BY ALL
                HAVING COUNT(*) > 1
            )
            SELECT occurrences, COUNT(*) AS groups,
                   SUM(occurrences) AS rows_in_groups
            FROM duplicate_groups
            GROUP BY occurrences
            ORDER BY occurrences
            LIMIT 20
        """,
        "customer_activity": """
            WITH counts AS (
                SELECT customer_id, COUNT(*) AS purchases,
                       COUNT(DISTINCT transaction_date) AS active_days
                FROM transactions
                GROUP BY customer_id
            )
            SELECT COUNT(*) AS customers,
                   MIN(purchases) AS minimum_purchases,
                   MEDIAN(purchases) AS median_purchases,
                   ROUND(AVG(purchases), 4) AS mean_purchases,
                   QUANTILE_CONT(purchases, 0.90) AS p90_purchases,
                   QUANTILE_CONT(purchases, 0.99) AS p99_purchases,
                   MAX(purchases) AS maximum_purchases,
                   MEDIAN(active_days) AS median_active_days,
                   QUANTILE_CONT(active_days, 0.99) AS p99_active_days
            FROM counts
        """,
        "article_popularity": """
            WITH counts AS (
                SELECT article_id, COUNT(*) AS purchases
                FROM transactions
                GROUP BY article_id
            ), ranked AS (
                SELECT *, ROW_NUMBER() OVER (ORDER BY purchases DESC, article_id) AS rank,
                       SUM(purchases) OVER () AS total_purchases
                FROM counts
            )
            SELECT COUNT(*) AS purchased_articles,
                   MIN(purchases) AS minimum_purchases,
                   MEDIAN(purchases) AS median_purchases,
                   ROUND(AVG(purchases), 4) AS mean_purchases,
                   QUANTILE_CONT(purchases, 0.90) AS p90_purchases,
                   QUANTILE_CONT(purchases, 0.99) AS p99_purchases,
                   MAX(purchases) AS maximum_purchases,
                   ROUND(100.0 * SUM(purchases) FILTER (WHERE rank <= 12) / MAX(total_purchases), 4) AS top_12_share_percent,
                   ROUND(100.0 * SUM(purchases) FILTER (WHERE rank <= 100) / MAX(total_purchases), 4) AS top_100_share_percent,
                   ROUND(100.0 * SUM(purchases) FILTER (WHERE rank <= 1000) / MAX(total_purchases), 4) AS top_1000_share_percent
            FROM ranked
        """,
        "daily_activity": """
            WITH counts AS (
                SELECT transaction_date, COUNT(*) AS transactions,
                       COUNT(DISTINCT customer_id) AS customers
                FROM transactions
                GROUP BY transaction_date
            )
            SELECT COUNT(*) AS observed_days,
                   MIN(transactions) AS minimum_transactions,
                   MEDIAN(transactions) AS median_transactions,
                   ROUND(AVG(transactions), 4) AS mean_transactions,
                   QUANTILE_CONT(transactions, 0.90) AS p90_transactions,
                   QUANTILE_CONT(transactions, 0.99) AS p99_transactions,
                   MAX(transactions) AS maximum_transactions,
                   MIN(customers) AS minimum_customers,
                   MEDIAN(customers) AS median_customers,
                   MAX(customers) AS maximum_customers
            FROM counts
        """,
        "temporal_split": f"""
            SELECT CASE
                       WHEN transaction_date <= DATE '{TRAIN_END}' THEN 'training'
                       WHEN transaction_date BETWEEN DATE '{VALIDATION_START}' AND DATE '{VALIDATION_END}' THEN 'validation'
                       ELSE 'outside'
                   END AS split,
                   COUNT(*) AS transactions,
                   COUNT(DISTINCT customer_id) AS customers,
                   COUNT(DISTINCT article_id) AS articles,
                   MIN(transaction_date) AS minimum_date,
                   MAX(transaction_date) AS maximum_date
            FROM transactions
            GROUP BY split
            ORDER BY split
        """,
        "validation_coldness": f"""
            WITH train_customers AS (
                SELECT customer_id, COUNT(*) AS history_purchases
                FROM transactions
                WHERE transaction_date <= DATE '{TRAIN_END}'
                GROUP BY customer_id
            ), train_articles AS (
                SELECT DISTINCT article_id
                FROM transactions
                WHERE transaction_date <= DATE '{TRAIN_END}'
            ), validation AS (
                SELECT customer_id, article_id
                FROM transactions
                WHERE transaction_date BETWEEN DATE '{VALIDATION_START}' AND DATE '{VALIDATION_END}'
            ), validation_customers AS (
                SELECT DISTINCT customer_id FROM validation
            ), validation_articles AS (
                SELECT DISTINCT article_id FROM validation
            )
            SELECT
                (SELECT COUNT(*) FROM validation_customers) AS validation_customers,
                (SELECT COUNT(*) FROM validation_customers v LEFT JOIN train_customers t USING (customer_id)
                 WHERE t.customer_id IS NULL) AS cold_customers,
                (SELECT COUNT(*) FROM validation_articles) AS validation_articles,
                (SELECT COUNT(*) FROM validation_articles v LEFT JOIN train_articles t USING (article_id)
                 WHERE t.article_id IS NULL) AS cold_articles,
                (SELECT COUNT(DISTINCT customer_id || ':' || article_id) FROM validation) AS unique_relevance_pairs
        """,
        "validation_target_distribution": f"""
            WITH labels AS (
                SELECT customer_id, COUNT(DISTINCT article_id) AS relevant_articles,
                       COUNT(*) AS transactions
                FROM transactions
                WHERE transaction_date BETWEEN DATE '{VALIDATION_START}' AND DATE '{VALIDATION_END}'
                GROUP BY customer_id
            )
            SELECT COUNT(*) AS customers,
                   MIN(relevant_articles) AS minimum_relevant_articles,
                   MEDIAN(relevant_articles) AS median_relevant_articles,
                   ROUND(AVG(relevant_articles), 4) AS mean_relevant_articles,
                   QUANTILE_CONT(relevant_articles, 0.90) AS p90_relevant_articles,
                   QUANTILE_CONT(relevant_articles, 0.99) AS p99_relevant_articles,
                   MAX(relevant_articles) AS maximum_relevant_articles,
                   SUM(transactions - relevant_articles) AS repeated_purchase_rows
            FROM labels
        """,
        "busiest_days": """
            SELECT transaction_date, COUNT(*) AS transactions,
                   COUNT(DISTINCT customer_id) AS customers
            FROM transactions
            GROUP BY transaction_date
            ORDER BY transactions DESC, transaction_date
            LIMIT 10
        """,
        "most_popular_articles": """
            SELECT t.article_id,
                   COALESCE(MAX(a.prod_name), '') AS product_name,
                   COUNT(*) AS purchases,
                   COUNT(DISTINCT t.customer_id) AS customers
            FROM transactions t
            LEFT JOIN articles a USING (article_id)
            GROUP BY t.article_id
            ORDER BY purchases DESC, t.article_id
            LIMIT 20
        """,
    }

    results: dict[str, dict[str, Any]] = {}
    total_start = time.monotonic()
    for index, (name, query) in enumerate(queries.items(), start=1):
        started = time.monotonic()
        print(f"[{index}/{len(queries)}] {name}", flush=True)
        relation = con.sql(query)
        columns = relation.columns
        rows = [list(row) for row in relation.fetchall()]
        elapsed = time.monotonic() - started
        results[name] = {
            "columns": columns,
            "rows": [[normalize(value) for value in row] for row in rows],
            "elapsed_seconds": round(elapsed, 3),
        }

        with (ARTIFACT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"    completed in {elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - total_start
    with (ARTIFACT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "duckdb_version": duckdb.__version__,
                "train_end": TRAIN_END,
                "validation_start": VALIDATION_START,
                "validation_end": VALIDATION_END,
                "total_elapsed_seconds": round(total_elapsed, 3),
                "queries": results,
            },
            handle,
            indent=2,
        )

    section_titles = {
        "dataset_overview": "Dataset overview",
        "transaction_quality": "Transaction integrity",
        "catalog_key_quality": "Catalog key integrity",
        "customer_missingness": "Customer-field missingness",
        "article_missingness": "Article-field missingness",
        "customer_profile_quality": "Customer profile quality",
        "customer_category_domains": "Customer category domains",
        "article_catalog_shape": "Article catalog shape",
        "foreign_key_coverage": "Foreign-key coverage",
        "sales_channels": "Sales channels",
        "price_distribution": "Normalized price distribution",
        "duplicate_summary": "Exact duplicate transactions",
        "duplicate_multiplicity": "Duplicate multiplicity",
        "customer_activity": "Customer activity",
        "article_popularity": "Article popularity and concentration",
        "daily_activity": "Daily activity distribution",
        "temporal_split": "Temporal split",
        "validation_coldness": "Validation coldness",
        "validation_target_distribution": "Validation target distribution",
        "busiest_days": "Busiest transaction days",
        "most_popular_articles": "Most-purchased articles",
    }
    lines = [
        "# H&M data audit",
        "",
        "Generated by `scripts/data_audit.py` from the immutable local CSV files.",
        "",
        "## Reproduction contract",
        "",
        f"- DuckDB version: `{duckdb.__version__}`",
        "- DuckDB threads: `2`",
        "- DuckDB memory limit: `6GB`",
        f"- Training cutoff: `{TRAIN_END}`",
        f"- Validation window: `{VALIDATION_START}` through `{VALIDATION_END}`",
        f"- Total query time: `{total_elapsed:.1f}` seconds",
        "- IDs were parsed as strings; dates, prices, and channels used `TRY_CAST`.",
        "",
    ]
    for name, result in results.items():
        lines.extend(
            [
                f"## {section_titles[name]}",
                "",
                markdown_table(result["columns"], result["rows"]),
                "",
                f"Query runtime: `{result['elapsed_seconds']}` seconds.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundaries",
            "",
            "- Exact duplicate purchase rows were measured but not removed. They may encode quantity.",
            "- Ranking relevance should use unique customer-article pairs even when transaction-frequency features preserve quantity.",
            "- The validation week was excluded from the training side of all coldness checks.",
            "- Offline recovery of later purchases does not establish causal lift or customer satisfaction.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit complete in {total_elapsed:.1f}s: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
