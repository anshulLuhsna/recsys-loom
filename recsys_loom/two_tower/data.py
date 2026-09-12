"""Leakage-safe data preparation for learned two-tower retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import duckdb
import numpy as np
from numpy.typing import NDArray

ARTICLE_FIELDS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "department_name",
    "index_name",
    "graphical_appearance_name",
]

CUSTOMER_FIELDS = [
    "club_member_status",
    "fashion_news_frequency",
]

USER_NUMERICAL_FIELDS = [
    "log_purchase_count",
    "log_unique_article_count",
    "log_days_since_last_purchase",
    "log_history_span_days",
    "mean_normalized_price",
    "normalized_price_stddev",
    "channel_2_fraction",
    "age_scaled",
]


@dataclass(slots=True)
class Catalog:
    """Article metadata encoded without article-ID embeddings."""

    article_ids: list[str]
    article_features: NDArray[np.int64]
    article_to_index: dict[str, int]
    article_vocabularies: dict[str, dict[str, int]]
    customer_vocabularies: dict[str, dict[str, int]]

    @property
    def article_cardinalities(self) -> list[int]:
        return [len(self.article_vocabularies[field]) for field in ARTICLE_FIELDS]

    @property
    def customer_cardinalities(self) -> list[int]:
        return [len(self.customer_vocabularies[field]) for field in CUSTOMER_FIELDS]


@dataclass(slots=True)
class SnapshotData:
    """User histories and future labels for one temporal cutoff."""

    cutoff: str
    target_start: str
    target_end: str
    customer_ids: list[str]
    history_features: NDArray[np.int64]
    history_article_indices: NDArray[np.int64]
    history_weights: NDArray[np.float32]
    user_numerical: NDArray[np.float32]
    user_categorical: NDArray[np.int64]
    positive_item_indices: list[NDArray[np.int64]]
    history_item_indices: list[set[int]]
    eligible_item_indices: NDArray[np.int64]
    relevant_by_customer: dict[str, set[str]]

    @property
    def positive_count(self) -> int:
        return sum(len(items) for items in self.positive_item_indices)

    @property
    def sparse_history_count(self) -> int:
        return int(np.sum(self.history_weights.sum(axis=1) == 0))


@dataclass(slots=True)
class UserNumericalNormalizer:
    """Training-only standardization for user numerical features."""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    @classmethod
    def fit(cls, snapshots: Sequence[SnapshotData]) -> UserNumericalNormalizer:
        values = np.concatenate([snapshot.user_numerical for snapshot in snapshots], axis=0)
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        return ((values - self.mean) / self.std).astype(np.float32)


def build_catalog(con: duckdb.DuckDBPyConnection) -> Catalog:
    """Build stable categorical vocabularies and article feature rows."""
    article_columns = ", ".join(ARTICLE_FIELDS)
    article_rows = con.sql(
        f"SELECT article_id, {article_columns} FROM articles ORDER BY article_id"
    ).fetchall()

    article_vocabularies = _build_vocabularies(article_rows, ARTICLE_FIELDS, start=1)
    article_ids = [row[0] for row in article_rows]
    article_features = np.zeros(
        (len(article_rows), len(ARTICLE_FIELDS)),
        dtype=np.int64,
    )
    for row_index, row in enumerate(article_rows):
        for field_index, field in enumerate(ARTICLE_FIELDS):
            value = _clean_value(row[field_index + 1])
            article_features[row_index, field_index] = article_vocabularies[field].get(
                value, 0
            )

    customer_vocabularies: dict[str, dict[str, int]] = {}
    for field in CUSTOMER_FIELDS:
        values = [
            _clean_value(row[0])
            for row in con.sql(
                f"""
                SELECT DISTINCT {field}
                FROM customers
                WHERE {field} IS NOT NULL AND TRIM({field}) != ''
                ORDER BY {field}
                """
            ).fetchall()
        ]
        customer_vocabularies[field] = {
            value: index for index, value in enumerate(values, start=1)
        }

    return Catalog(
        article_ids=article_ids,
        article_features=article_features,
        article_to_index={article_id: i for i, article_id in enumerate(article_ids)},
        article_vocabularies=article_vocabularies,
        customer_vocabularies=customer_vocabularies,
    )


def load_target_customers(
    con: duckdb.DuckDBPyConnection,
    target_start: str,
    target_end: str,
    sample_size: int = 0,
    random_seed: int = 42,
) -> list[str]:
    """Return a deterministic sample of customers active in the target week."""
    rows = con.sql(f"""
        SELECT DISTINCT customer_id
        FROM transactions
        WHERE transaction_date BETWEEN DATE '{target_start}' AND DATE '{target_end}'
        ORDER BY customer_id
    """).fetchall()
    customer_ids = [row[0] for row in rows]
    if sample_size <= 0 or sample_size >= len(customer_ids):
        return customer_ids

    rng = np.random.default_rng(random_seed)
    selected = rng.choice(len(customer_ids), size=sample_size, replace=False)
    return sorted(customer_ids[int(i)] for i in selected)


def build_snapshot(
    con: duckdb.DuckDBPyConnection,
    catalog: Catalog,
    cutoff: str,
    target_start: str,
    target_end: str,
    customer_ids: Sequence[str],
    max_history: int = 50,
    history_half_life_days: float = 45.0,
    observed_items_only: bool = True,
) -> SnapshotData:
    """Construct model inputs using only transactions on or before ``cutoff``."""
    ids = list(customer_ids)
    _replace_customer_table(con, "two_tower_customers", ids)

    relevant_by_customer = _load_relevance(
        con,
        target_start,
        target_end,
        ids,
    )
    eligible_indices = _load_eligible_indices(
        con,
        catalog,
        cutoff,
        observed_items_only,
    )
    eligible_set = set(int(i) for i in eligible_indices)

    history_features = np.zeros(
        (len(ids), max_history, len(ARTICLE_FIELDS)),
        dtype=np.int64,
    )
    history_article_indices = np.full(
        (len(ids), max_history),
        -1,
        dtype=np.int64,
    )
    history_weights = np.zeros((len(ids), max_history), dtype=np.float32)
    history_item_indices: list[set[int]] = [set() for _ in ids]
    customer_to_index = {customer_id: i for i, customer_id in enumerate(ids)}

    history_rows = con.sql(f"""
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
            INNER JOIN two_tower_customers c USING (customer_id)
            WHERE t.transaction_date <= DATE '{cutoff}'
        )
        SELECT customer_id, article_id, days_ago, history_rank
        FROM ranked
        WHERE history_rank <= {int(max_history)}
        ORDER BY customer_id, history_rank
    """).fetchall()

    decay_rate = math.log(2.0) / history_half_life_days
    for customer_id, article_id, days_ago, history_rank in history_rows:
        item_index = catalog.article_to_index.get(article_id)
        if item_index is None:
            continue
        customer_index = customer_to_index[customer_id]
        slot = int(history_rank) - 1
        history_features[customer_index, slot] = catalog.article_features[item_index]
        history_article_indices[customer_index, slot] = item_index
        history_weights[customer_index, slot] = math.exp(
            -decay_rate * max(float(days_ago), 0.0)
        )
        history_item_indices[customer_index].add(item_index)

    user_numerical, user_categorical = _load_customer_features(
        con,
        catalog,
        cutoff,
        ids,
        customer_to_index,
    )

    positive_item_indices: list[NDArray[np.int64]] = []
    for customer_id in ids:
        positive_indices = sorted(
            catalog.article_to_index[article_id]
            for article_id in relevant_by_customer.get(customer_id, set())
            if article_id in catalog.article_to_index
            and catalog.article_to_index[article_id] in eligible_set
        )
        positive_item_indices.append(np.asarray(positive_indices, dtype=np.int64))

    con.execute("DROP TABLE IF EXISTS two_tower_customers")

    return SnapshotData(
        cutoff=cutoff,
        target_start=target_start,
        target_end=target_end,
        customer_ids=ids,
        history_features=history_features,
        history_article_indices=history_article_indices,
        history_weights=history_weights,
        user_numerical=user_numerical,
        user_categorical=user_categorical,
        positive_item_indices=positive_item_indices,
        history_item_indices=history_item_indices,
        eligible_item_indices=eligible_indices,
        relevant_by_customer=relevant_by_customer,
    )


def _load_relevance(
    con: duckdb.DuckDBPyConnection,
    target_start: str,
    target_end: str,
    customer_ids: Sequence[str],
) -> dict[str, set[str]]:
    rows = con.sql(f"""
        SELECT t.customer_id, t.article_id
        FROM transactions t
        INNER JOIN two_tower_customers c USING (customer_id)
        WHERE t.transaction_date BETWEEN DATE '{target_start}' AND DATE '{target_end}'
        GROUP BY t.customer_id, t.article_id
    """).fetchall()
    result = {customer_id: set() for customer_id in customer_ids}
    for customer_id, article_id in rows:
        result[customer_id].add(article_id)
    return result


def _load_eligible_indices(
    con: duckdb.DuckDBPyConnection,
    catalog: Catalog,
    cutoff: str,
    observed_items_only: bool,
) -> NDArray[np.int64]:
    if observed_items_only:
        rows = con.sql(f"""
            SELECT DISTINCT article_id
            FROM transactions
            WHERE transaction_date <= DATE '{cutoff}'
            ORDER BY article_id
        """).fetchall()
        indices = [
            catalog.article_to_index[row[0]]
            for row in rows
            if row[0] in catalog.article_to_index
        ]
    else:
        indices = list(range(len(catalog.article_ids)))
    return np.asarray(indices, dtype=np.int64)


def _load_customer_features(
    con: duckdb.DuckDBPyConnection,
    catalog: Catalog,
    cutoff: str,
    customer_ids: Sequence[str],
    customer_to_index: dict[str, int],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    aggregate_rows = con.sql(f"""
        SELECT
            t.customer_id,
            COUNT(*) AS purchase_count,
            COUNT(DISTINCT t.article_id) AS unique_article_count,
            (DATE '{cutoff}' - MAX(t.transaction_date))::INT AS days_since_last,
            (MAX(t.transaction_date) - MIN(t.transaction_date))::INT AS history_span,
            COALESCE(AVG(t.price), 0.0) AS mean_price,
            COALESCE(STDDEV_POP(t.price), 0.0) AS price_stddev,
            COALESCE(AVG(CASE WHEN t.sales_channel_id = 2 THEN 1.0 ELSE 0.0 END), 0.0)
                AS channel_2_fraction
        FROM transactions t
        INNER JOIN two_tower_customers c USING (customer_id)
        WHERE t.transaction_date <= DATE '{cutoff}'
        GROUP BY t.customer_id
    """).fetchall()

    customer_columns = ", ".join(f"c.{field}" for field in CUSTOMER_FIELDS)
    metadata_rows = con.sql(f"""
        SELECT c.customer_id, c.age, {customer_columns}
        FROM customers c
        INNER JOIN two_tower_customers selected USING (customer_id)
    """).fetchall()

    numerical = np.zeros(
        (len(customer_ids), len(USER_NUMERICAL_FIELDS)),
        dtype=np.float32,
    )
    categorical = np.zeros(
        (len(customer_ids), len(CUSTOMER_FIELDS)),
        dtype=np.int64,
    )

    for row in aggregate_rows:
        customer_index = customer_to_index[row[0]]
        numerical[customer_index] = np.asarray(
            [
                math.log1p(float(row[1])),
                math.log1p(float(row[2])),
                math.log1p(max(float(row[3]), 0.0)),
                math.log1p(max(float(row[4]), 0.0)),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                0.0,
            ],
            dtype=np.float32,
        )

    for row in metadata_rows:
        customer_index = customer_to_index[row[0]]
        age = _parse_float(row[1])
        numerical[customer_index, -1] = age / 100.0 if age is not None else 0.0
        for field_index, field in enumerate(CUSTOMER_FIELDS):
            value = _clean_value(row[field_index + 2])
            categorical[customer_index, field_index] = (
                catalog.customer_vocabularies[field].get(value, 0)
            )

    return numerical, categorical


def _replace_customer_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    customer_ids: Sequence[str],
) -> None:
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table_name} (customer_id VARCHAR)")
    if customer_ids:
        con.executemany(
            f"INSERT INTO {table_name} VALUES (?)",
            [(customer_id,) for customer_id in customer_ids],
        )


def _build_vocabularies(
    rows: Sequence[tuple],
    fields: Sequence[str],
    start: int,
) -> dict[str, dict[str, int]]:
    vocabularies: dict[str, dict[str, int]] = {}
    for field_index, field in enumerate(fields):
        values = sorted(
            {
                _clean_value(row[field_index + 1])
                for row in rows
                if _clean_value(row[field_index + 1])
            }
        )
        vocabularies[field] = {
            value: index
            for index, value in enumerate(values, start=start)
        }
    return vocabularies


def _clean_value(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _parse_float(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
