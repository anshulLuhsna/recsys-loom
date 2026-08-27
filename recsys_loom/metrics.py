"""Ranking metrics used by RecSys Loom experiments."""

from collections.abc import Mapping


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
