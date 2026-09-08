"""Ranking and retrieval metrics used by RecSys Loom experiments."""

from collections.abc import Mapping, Sequence


def average_precision_at_k(
    relevant: set[str],
    predicted: list[str],
    k: int = 12,
) -> float:
    """Return average precision at ``k`` for one customer."""
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()
    for rank, article_id in enumerate(predicted[:k], start=1):
        if article_id in relevant and article_id not in seen:
            hits += 1
            precision_sum += hits / rank
        seen.add(article_id)

    return precision_sum / min(len(relevant), k)


def ranking_metrics_at_k(
    relevant_by_customer: Mapping[str, set[str]],
    predictions_by_customer: Mapping[str, list[str]],
    catalog_size: int,
    k: int = 12,
) -> dict[str, float | int]:
    """Return macro ranking metrics over customers with known relevance."""
    customer_ids = list(relevant_by_customer)
    if not customer_ids:
        raise ValueError("At least one evaluated customer is required")
    if catalog_size <= 0:
        raise ValueError("Catalog size must be positive")

    average_precisions: list[float] = []
    recalls: list[float] = []
    customers_with_hit = 0
    matched_relevance_pairs = 0
    relevance_pairs = 0
    recommended_articles: set[str] = set()

    for customer_id in customer_ids:
        relevant = relevant_by_customer[customer_id]
        predicted = predictions_by_customer.get(customer_id, [])[:k]
        unique_predicted = set(predicted)
        hits = len(relevant.intersection(unique_predicted))

        average_precisions.append(average_precision_at_k(relevant, predicted, k))
        recalls.append(hits / len(relevant))
        customers_with_hit += int(hits > 0)
        matched_relevance_pairs += hits
        relevance_pairs += len(relevant)
        recommended_articles.update(unique_predicted)

    customer_count = len(customer_ids)
    return {
        "customers": customer_count,
        "customers_with_hit": customers_with_hit,
        "matched_relevance_pairs": matched_relevance_pairs,
        f"map_at_{k}": sum(average_precisions) / customer_count,
        f"recall_at_{k}": sum(recalls) / customer_count,
        f"micro_recall_at_{k}": matched_relevance_pairs / relevance_pairs,
        f"hit_rate_at_{k}": customers_with_hit / customer_count,
        f"catalog_coverage_at_{k}": len(recommended_articles) / catalog_size,
    }


# ---------------------------------------------------------------------------
# Retrieval metrics — order-insensitive candidate-set evaluation
# ---------------------------------------------------------------------------


def recall_at_k(relevant: set[str], candidates: Sequence[str], k: int) -> float:
    """Fraction of relevant items found in the first ``k`` candidates."""
    if not relevant:
        return 0.0
    return len(relevant.intersection(candidates[:k])) / len(relevant)


def retrieval_metrics_at_k(
    relevant_by_customer: Mapping[str, set[str]],
    candidates_by_customer: Mapping[str, Sequence[str]],
    catalog_size: int,
    k_values: Sequence[int] = (100, 300, 500),
) -> dict[str, float | int]:
    """Evaluate a candidate generator at multiple cutoffs.

    Unlike ``ranking_metrics_at_k``, this measures whether relevant items
    appear *anywhere* in the top-K candidates.  Order within the set does
    not matter — only set membership.
    """
    customer_ids = list(relevant_by_customer)
    if not customer_ids:
        raise ValueError("At least one evaluated customer is required")
    if catalog_size <= 0:
        raise ValueError("Catalog size must be positive")

    customer_count = len(customer_ids)
    relevance_pairs = sum(len(r) for r in relevant_by_customer.values())
    all_candidates: set[str] = set()

    per_k: dict[int, dict[str, float]] = {}
    for k in k_values:
        recalls: list[float] = []
        customers_with_hit = 0
        matched = 0

        for customer_id in customer_ids:
            relevant = relevant_by_customer[customer_id]
            candidates = candidates_by_customer.get(customer_id, [])
            top_k = set(candidates[:k])
            hits = len(relevant.intersection(top_k))

            recalls.append(hits / len(relevant))
            customers_with_hit += int(hits > 0)
            matched += hits
            all_candidates.update(top_k)

        per_k[k] = {
            f"recall_at_{k}": sum(recalls) / customer_count,
            f"micro_recall_at_{k}": matched / relevance_pairs if relevance_pairs else 0.0,
            f"hit_rate_at_{k}": customers_with_hit / customer_count,
            f"customers_with_hit_at_{k}": customers_with_hit,
            f"matched_pairs_at_{k}": matched,
        }

    result: dict[str, float | int] = {
        "customers": customer_count,
        "relevance_pairs": relevance_pairs,
        "catalog_size": catalog_size,
        "catalog_coverage": len(all_candidates) / catalog_size,
        "unique_candidates": len(all_candidates),
    }
    for k in k_values:
        result.update(per_k[k])

    return result
