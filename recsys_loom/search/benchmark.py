"""Synthetic search-benchmark helpers. Not real query-log evaluation."""

from __future__ import annotations

import math
from collections import defaultdict

from recsys_loom.search.catalog import Article


def ndcg(relevances: list[float], k: int) -> float:
    predicted = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]

    def dcg(values: list[float]) -> float:
        return sum(rel / math.log2(index + 2) for index, rel in enumerate(values))

    denom = dcg(ideal)
    return dcg(predicted) / denom if denom else 0.0


def generate_structured_queries(
    articles: dict[str, Article],
    limit: int = 80,
) -> list[str]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for article in articles.values():
        if article.colour_group_name and article.product_type_name:
            counts[(article.colour_group_name, article.product_type_name)] += 1
    popular = [
        pair
        for pair, count in sorted(counts.items(), key=lambda item: -item[1])
        if count >= 40
    ]
    return [
        f"{color.lower()} {product_type.lower()}"
        for color, product_type in popular[:limit]
    ]
