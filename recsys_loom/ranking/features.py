"""Feature engineering for the LambdaRank ranker.

Builds a feature matrix for each (customer, candidate_article) pair.
All features are computed strictly from data available before the cutoff.

Feature groups:
    1. Source flags and scores: which sources nominated this item
    2. Customer category affinity: how much of the customer's history
       matches this article's attributes
    3. Recency: days since last purchase of this item/type/group
    4. Frequency: how many times the customer bought this item/type/group
    5. Item popularity statistics: 7d/30d counts, growth, unique buyers
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import duckdb
import numpy as np
from numpy.typing import NDArray

from recsys_loom.candidates import CandidateRecord

SOURCE_NAMES = ["recent_7d_pop", "repeat_purchase", "cooccurrence", "als", "content"]

CATEGORY_FEATURES = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "department_name",
]


def _feature_columns(source_names: Sequence[str] = SOURCE_NAMES) -> list[str]:
    cols: list[str] = []
    for s in source_names:
        cols += [f"is_{s}", f"rank_{s}", f"rank_pct_{s}", f"score_{s}"]
    cols.append("num_sources")
    for cat in CATEGORY_FEATURES:
        cols.append(f"affinity_{cat}")
    cols += [
        "days_since_bought_item",
        "days_since_bought_product_type",
        "days_since_bought_garment_group",
        "times_bought_item",
        "times_bought_product_type",
        "times_bought_garment_group",
        "item_purchases_7d",
        "item_purchases_30d",
        "item_purchase_growth",
        "item_unique_buyers_30d",
    ]
    return cols


FEATURE_COLUMNS = _feature_columns()


def build_features(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    candidate_records_by_source: dict[str, list[CandidateRecord]],
    train_end: str,
    k_per_source: int = 500,
    source_names: Sequence[str] = SOURCE_NAMES,
) -> tuple[list[str], NDArray[np.float32], list[tuple[str, str]]]:
    """Build feature matrix for all (customer, candidate) pairs.

    Returns:
        feature_names: ordered list of feature column names
        X: (n_pairs, n_features) float32 matrix
        pairs: [(customer_id, article_id), ...] in same order as X rows
    """
    feature_columns = _feature_columns(source_names)
    source_offsets = {
        source_name: {
            "is": index * 4,
            "rank": index * 4 + 1,
            "rank_pct": index * 4 + 2,
            "score": index * 4 + 3,
        }
        for index, source_name in enumerate(source_names)
    }
    num_sources_col = len(source_names) * 4
    affinity_start = num_sources_col + 1
    recency_start = affinity_start + len(CATEGORY_FEATURES)
    item_stats_start = recency_start + 6
    cid_set = set(customer_ids)

    per_cust_art: dict[str, dict[str, dict[str, float]]] = {}
    for source_name, records in candidate_records_by_source.items():
        for r in records:
            if r.customer_id not in cid_set:
                continue
            art = per_cust_art.setdefault(r.customer_id, {}).setdefault(r.article_id, {})
            art[f"is_{source_name}"] = 1.0
            art[f"rank_{source_name}"] = float(r.source_rank)
            art[f"score_{source_name}"] = float(r.source_score)

    pairs: list[tuple[str, str]] = []
    for cid in customer_ids:
        for aid in per_cust_art.get(cid, {}):
            pairs.append((cid, aid))

    n_pairs = len(pairs)
    n_feat = len(feature_columns)
    X = np.full((n_pairs, n_feat), np.nan, dtype=np.float32)

    pair_to_row = {pair: i for i, pair in enumerate(pairs)}

    for source_name, records in candidate_records_by_source.items():
        if source_name not in source_offsets:
            continue
        off = source_offsets[source_name]
        for r in records:
            key = (r.customer_id, r.article_id)
            row = pair_to_row.get(key)
            if row is None:
                continue
            X[row, off["is"]] = 1.0
            X[row, off["rank"]] = float(r.source_rank)
            X[row, off["rank_pct"]] = float(r.source_rank) / k_per_source
            X[row, off["score"]] = float(r.source_score)

    for i in range(n_pairs):
        count = 0
        for source_name in source_names:
            if X[i, source_offsets[source_name]["is"]] == 1.0:
                count += 1
            else:
                X[i, source_offsets[source_name]["is"]] = 0.0
        X[i, num_sources_col] = count

    all_articles_needed = {aid for _, aid in pairs}
    article_meta = _load_article_metadata_fast(con, all_articles_needed)
    item_stats = _load_item_stats_fast(con, train_end, all_articles_needed)
    customer_profiles = _build_customer_profiles(con, customer_ids, train_end)

    for i, (cid, aid) in enumerate(pairs):
        meta = article_meta.get(aid)
        profile = customer_profiles.get(cid)

        if meta and profile:
            for j, cat in enumerate(CATEGORY_FEATURES):
                val = meta.get(cat, "")
                col = affinity_start + j
                if val and cat in profile["affinity"]:
                    X[i, col] = profile["affinity"][cat].get(val, 0.0)
                else:
                    X[i, col] = 0.0

            item_rec = profile.get("recency_item", {})
            ptype = meta.get("product_type_name", "")
            ggroup = meta.get("garment_group_name", "")

            X[i, recency_start] = item_rec.get(aid, -1.0)
            X[i, recency_start + 1] = profile.get("recency_ptype", {}).get(ptype, -1.0) if ptype else -1.0
            X[i, recency_start + 2] = profile.get("recency_ggroup", {}).get(ggroup, -1.0) if ggroup else -1.0
            X[i, recency_start + 3] = profile.get("freq_item", {}).get(aid, 0)
            X[i, recency_start + 4] = profile.get("freq_ptype", {}).get(ptype, 0) if ptype else 0
            X[i, recency_start + 5] = profile.get("freq_ggroup", {}).get(ggroup, 0) if ggroup else 0

        stats = item_stats.get(aid, {})
        X[i, item_stats_start] = stats.get("p7", 0.0)
        X[i, item_stats_start + 1] = stats.get("p30", 0.0)
        X[i, item_stats_start + 2] = stats.get("growth", 0.0)
        X[i, item_stats_start + 3] = stats.get("ub30", 0.0)

    return feature_columns, X, pairs


def _load_article_metadata_fast(
    con: duckdb.DuckDBPyConnection,
    article_ids: set[str],
) -> dict[str, dict[str, str]]:
    cols = ", ".join(CATEGORY_FEATURES)
    rows = con.sql(f"SELECT article_id, {cols} FROM articles").fetchall()
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        aid = row[0]
        if aid not in article_ids:
            continue
        result[aid] = {
            CATEGORY_FEATURES[j]: (row[j + 1] or "") for j in range(len(CATEGORY_FEATURES))
        }
    return result


def _load_item_stats_fast(
    con: duckdb.DuckDBPyConnection,
    train_end: str,
    article_ids: set[str],
) -> dict[str, dict[str, float]]:
    rows = con.sql(f"""
        SELECT
            article_id,
            COUNT(*) FILTER (
                WHERE transaction_date > DATE '{train_end}' - INTERVAL '7 days'
            ) AS p7,
            COUNT(*) FILTER (
                WHERE transaction_date > DATE '{train_end}' - INTERVAL '30 days'
            ) AS p30,
            COUNT(DISTINCT customer_id) FILTER (
                WHERE transaction_date > DATE '{train_end}' - INTERVAL '30 days'
            ) AS ub30
        FROM transactions
        WHERE transaction_date <= DATE '{train_end}'
        GROUP BY article_id
    """).fetchall()

    result: dict[str, dict[str, float]] = {}
    for aid, p7, p30, ub30 in rows:
        if aid not in article_ids:
            continue
        growth = (p7 / max(p30 - p7, 1)) if p30 > p7 else 0.0
        result[aid] = {"p7": float(p7), "p30": float(p30), "growth": growth, "ub30": float(ub30)}
    return result


def _build_customer_profiles(
    con: duckdb.DuckDBPyConnection,
    customer_ids: Sequence[str],
    train_end: str,
) -> dict[str, dict[str, Any]]:
    """Build affinity, recency, and frequency profiles per customer."""
    con.execute("CREATE OR REPLACE TEMP TABLE prof_customers (customer_id VARCHAR)")
    con.executemany(
        "INSERT INTO prof_customers VALUES (?)",
        [(c,) for c in customer_ids],
    )

    cat_cols = ", ".join(f"a.{c}" for c in CATEGORY_FEATURES)
    rows = con.sql(f"""
        SELECT
            t.customer_id,
            t.article_id,
            (DATE '{train_end}' - t.transaction_date)::INT AS days_ago,
            {cat_cols}
        FROM transactions t
        INNER JOIN prof_customers pc ON pc.customer_id = t.customer_id
        LEFT JOIN articles a ON a.article_id = t.article_id
        WHERE t.transaction_date <= DATE '{train_end}'
    """).fetchall()

    con.execute("DROP TABLE IF EXISTS prof_customers")

    profiles: dict[str, dict[str, Any]] = {}
    raw: dict[str, list[tuple]] = {cid: [] for cid in customer_ids}
    for row in rows:
        cid = row[0]
        if cid in raw:
            raw[cid].append(row)

    for cid, entries in raw.items():
        if not entries:
            continue

        aff: dict[str, dict[str, float]] = {}
        for j, cat in enumerate(CATEGORY_FEATURES):
            counts: dict[str, int] = {}
            for row in entries:
                val = row[3 + j] or ""
                if val:
                    counts[val] = counts.get(val, 0) + 1
            total = sum(counts.values()) or 1
            aff[cat] = {v: c / total for v, c in counts.items()}

        rec_item: dict[str, float] = {}
        rec_ptype: dict[str, float] = {}
        rec_ggroup: dict[str, float] = {}
        freq_item: dict[str, int] = {}
        freq_ptype: dict[str, int] = {}
        freq_ggroup: dict[str, int] = {}

        for row in entries:
            aid = row[1]
            days = float(row[2])
            ptype = row[3 + CATEGORY_FEATURES.index("product_type_name")] or ""
            ggroup = row[3 + CATEGORY_FEATURES.index("garment_group_name")] or ""

            if aid not in rec_item or days < rec_item[aid]:
                rec_item[aid] = days
            freq_item[aid] = freq_item.get(aid, 0) + 1

            if ptype:
                if ptype not in rec_ptype or days < rec_ptype[ptype]:
                    rec_ptype[ptype] = days
                freq_ptype[ptype] = freq_ptype.get(ptype, 0) + 1

            if ggroup:
                if ggroup not in rec_ggroup or days < rec_ggroup[ggroup]:
                    rec_ggroup[ggroup] = days
                freq_ggroup[ggroup] = freq_ggroup.get(ggroup, 0) + 1

        profiles[cid] = {
            "affinity": aff,
            "recency_item": rec_item,
            "recency_ptype": rec_ptype,
            "recency_ggroup": rec_ggroup,
            "freq_item": freq_item,
            "freq_ptype": freq_ptype,
            "freq_ggroup": freq_ggroup,
        }

    return profiles
