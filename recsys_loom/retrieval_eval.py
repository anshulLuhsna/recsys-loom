"""Retrieval evaluation harness for comparing candidate sources.

Runs multiple candidate generators on identical chronological folds
and produces standalone, union, overlap, and marginal recall reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recsys_loom.metrics import retrieval_metrics_at_k


def standalone_recall(
    relevant_by_customer: Mapping[str, set[str]],
    candidates_by_source: dict[str, dict[str, Sequence[str]]],
    catalog_size: int,
    k_values: Sequence[int] = (100, 300, 500),
) -> dict[str, dict[str, float | int]]:
    """Evaluate each source independently at multiple cutoffs."""
    return {
        source_name: retrieval_metrics_at_k(
            relevant_by_customer, candidates, catalog_size, k_values,
        )
        for source_name, candidates in candidates_by_source.items()
    }


def union_recall(
    relevant_by_customer: Mapping[str, set[str]],
    candidates_by_source: dict[str, dict[str, Sequence[str]]],
    catalog_size: int,
    k_per_source: int,
) -> dict[str, float | int]:
    """Evaluate the deduplicated union of all sources (each capped at k_per_source)."""
    customer_ids = list(relevant_by_customer)
    union: dict[str, list[str]] = {}

    for customer_id in customer_ids:
        seen: set[str] = set()
        merged: list[str] = []
        for candidates in candidates_by_source.values():
            for article_id in candidates.get(customer_id, [])[:k_per_source]:
                if article_id not in seen:
                    seen.add(article_id)
                    merged.append(article_id)
        union[customer_id] = merged

    total_candidates = sum(len(v) for v in union.values())
    avg_candidates = total_candidates / len(customer_ids) if customer_ids else 0

    metrics = retrieval_metrics_at_k(
        relevant_by_customer, union, catalog_size, k_values=(len(max(union.values(), key=len)),),
    )
    metrics["avg_candidates_after_dedup"] = avg_candidates
    return metrics


def marginal_recall(
    relevant_by_customer: Mapping[str, set[str]],
    candidates_by_source: dict[str, dict[str, Sequence[str]]],
    k_per_source: int,
) -> dict[str, dict[str, int | float]]:
    """For each source, count relevant items it uniquely recovers.

    A "unique recovery" is a (customer, article) pair where the article is
    relevant and appears in this source's top-K but in no other source's top-K.
    """
    source_names = list(candidates_by_source)
    customer_ids = set(relevant_by_customer)

    source_hits: dict[str, set[tuple[str, str]]] = {name: set() for name in source_names}

    for source_name, candidates in candidates_by_source.items():
        for customer_id in customer_ids:
            relevant = relevant_by_customer[customer_id]
            top_k = set(candidates.get(customer_id, [])[:k_per_source])
            for article_id in relevant.intersection(top_k):
                source_hits[source_name].add((customer_id, article_id))

    all_hits = set()
    for hits in source_hits.values():
        all_hits.update(hits)

    result: dict[str, dict[str, int | float]] = {}
    for source_name in source_names:
        others = set()
        for other_name, hits in source_hits.items():
            if other_name != source_name:
                others.update(hits)

        unique = source_hits[source_name] - others
        result[source_name] = {
            "total_hits": len(source_hits[source_name]),
            "unique_hits": len(unique),
            "shared_hits": len(source_hits[source_name]) - len(unique),
            "unique_fraction": (
                len(unique) / len(source_hits[source_name])
                if source_hits[source_name]
                else 0.0
            ),
        }

    return result


def pairwise_overlap(
    candidates_by_source: dict[str, dict[str, Sequence[str]]],
    customer_ids: Sequence[str],
    k_per_source: int,
) -> dict[str, dict[str, float]]:
    """Jaccard overlap between every pair of sources across all customers."""
    source_names = list(candidates_by_source)
    result: dict[str, dict[str, float]] = {}

    for i, name_a in enumerate(source_names):
        result[name_a] = {}
        for name_b in source_names[i:]:
            intersection_total = 0
            union_total = 0
            for customer_id in customer_ids:
                set_a = set(candidates_by_source[name_a].get(customer_id, [])[:k_per_source])
                set_b = set(candidates_by_source[name_b].get(customer_id, [])[:k_per_source])
                intersection_total += len(set_a & set_b)
                union_total += len(set_a | set_b)

            jaccard = intersection_total / union_total if union_total else 0.0
            result[name_a][name_b] = jaccard
            if name_a != name_b:
                result.setdefault(name_b, {})[name_a] = jaccard

    return result


def incremental_union_recall(
    relevant_by_customer: Mapping[str, set[str]],
    candidates_by_source: dict[str, dict[str, Sequence[str]]],
    source_order: Sequence[str],
    k_per_source: int,
) -> list[dict[str, Any]]:
    """Add sources one at a time (in source_order) and measure union recall.

    Returns a list of dicts, one per incremental addition, showing how
    recall grows as each source is added to the pool.
    """
    customer_ids = list(relevant_by_customer)
    total_relevant = sum(len(r) for r in relevant_by_customer.values())
    accumulated: dict[str, set[str]] = {cid: set() for cid in customer_ids}
    results: list[dict[str, Any]] = []

    for source_name in source_order:
        candidates = candidates_by_source.get(source_name, {})
        for cid in customer_ids:
            for aid in candidates.get(cid, [])[:k_per_source]:
                accumulated[cid].add(aid)

        hits = sum(
            len(relevant_by_customer[cid].intersection(accumulated[cid]))
            for cid in customer_ids
        )
        recall = hits / total_relevant if total_relevant else 0.0
        avg_pool = sum(len(accumulated[cid]) for cid in customer_ids) / len(customer_ids)

        results.append({
            "source_added": source_name,
            "cumulative_sources": len(results) + 1,
            "union_hits": hits,
            "union_recall": recall,
            "avg_pool_size": avg_pool,
        })

    return results


def retrieval_report(
    relevant_by_customer: Mapping[str, set[str]],
    candidates_by_source: dict[str, dict[str, Sequence[str]]],
    catalog_size: int,
    k_values: Sequence[int] = (100, 300, 500),
) -> dict[str, Any]:
    """Full retrieval comparison report across sources."""
    return {
        "standalone": standalone_recall(
            relevant_by_customer, candidates_by_source, catalog_size, k_values,
        ),
        "marginal": {
            f"at_{k}": marginal_recall(
                relevant_by_customer, candidates_by_source, k,
            )
            for k in k_values
        },
        "overlap": {
            f"at_{k}": pairwise_overlap(
                candidates_by_source, list(relevant_by_customer), k,
            )
            for k in k_values
        },
    }
