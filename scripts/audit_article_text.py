#!/usr/bin/env python3
"""Audit article text coverage and define the controlled V1 text input."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FREE_TEXT_FIELDS = ["detail_desc"]
NEW_SHORT_TEXT_FIELDS = ["prod_name"]
CATEGORICAL_TEXT_FIELDS = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]
OUTPUT_PATH = ROOT / "artifacts" / "text_retrieval" / "text_field_audit.json"


def field_report(
    con: duckdb.DuckDBPyConnection,
    field: str,
) -> dict[str, object]:
    populated, missing, average_characters, average_words, unique_values = con.sql(
        f"""
        SELECT
            COUNT(*) FILTER (
                WHERE {field} IS NOT NULL AND TRIM({field}) != ''
            ),
            COUNT(*) FILTER (
                WHERE {field} IS NULL OR TRIM({field}) = ''
            ),
            AVG(LENGTH({field})) FILTER (
                WHERE {field} IS NOT NULL AND TRIM({field}) != ''
            ),
            AVG(
                LENGTH(TRIM({field}))
                - LENGTH(REPLACE(TRIM({field}), ' ', ''))
                + 1
            ) FILTER (
                WHERE {field} IS NOT NULL AND TRIM({field}) != ''
            ),
            COUNT(DISTINCT {field}) FILTER (
                WHERE {field} IS NOT NULL AND TRIM({field}) != ''
            )
        FROM articles
        """
    ).fetchone()
    examples = [
        row[0]
        for row in con.sql(
            f"""
            SELECT {field}
            FROM articles
            WHERE {field} IS NOT NULL AND TRIM({field}) != ''
            GROUP BY {field}
            ORDER BY COUNT(*) DESC, {field}
            LIMIT 3
            """
        ).fetchall()
    ]
    total = int(populated) + int(missing)
    return {
        "populated": int(populated),
        "missing": int(missing),
        "coverage": int(populated) / total if total else 0.0,
        "average_characters": float(average_characters),
        "average_words": float(average_words),
        "unique_values": int(unique_values),
        "examples": examples,
    }


def main() -> None:
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW articles AS
        SELECT *
        FROM read_csv(
            '{ROOT / "articles.csv"}',
            header=true,
            all_varchar=true
        )
    """)
    fields = [
        *FREE_TEXT_FIELDS,
        *NEW_SHORT_TEXT_FIELDS,
        *CATEGORICAL_TEXT_FIELDS,
    ]
    report = {
        "article_count": int(con.sql("SELECT COUNT(*) FROM articles").fetchone()[0]),
        "fields": {
            field: field_report(con, field)
            for field in fields
        },
        "classification": {
            "descriptive_free_text": FREE_TEXT_FIELDS,
            "new_short_text": NEW_SHORT_TEXT_FIELDS,
            "categorical_metadata_already_encoded_by_item_tower": (
                CATEGORICAL_TEXT_FIELDS
            ),
        },
        "v1_text_input": {
            "included_fields": ["prod_name", "detail_desc"],
            "excluded_fields": CATEGORICAL_TEXT_FIELDS,
            "format": "Product: {prod_name}. Description: {detail_desc}",
            "reason": (
                "prod_name and detail_desc add lexical semantics not already "
                "represented as categorical item-tower inputs; categorical "
                "name fields stay excluded to avoid duplicate signal"
            ),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {OUTPUT_PATH}")
    con.close()


if __name__ == "__main__":
    main()
