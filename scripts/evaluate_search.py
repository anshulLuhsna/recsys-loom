#!/usr/bin/env python3
"""Synthetic search benchmark. Not real H&M query/click evaluation."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.protocol import OVERNIGHT_DIR, ensure_directories
from recsys_loom.search.catalog import load_articles, section_family
from recsys_loom.search.intent import parse_query
from recsys_loom.search.lexical import BM25Index
from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.semantic import SemanticIndex
from recsys_loom.search.structured import StructuredIndex, _color_match

CURATED_QUERIES = [
    "black oversized hoodie",
    "linen summer shirt",
    "blue women's dress",
    "minimal black trousers",
    "casual green jacket",
    "white top for office",
    "red cardigan",
    "men's hoodie",
    "white cotton shirt",
    "black sportswear",
]


def ndcg(relevances: list[float], k: int) -> float:
    predicted = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]
    def dcg(values: list[float]) -> float:
        return sum(
            rel / math.log2(index + 2) for index, rel in enumerate(values)
        )
    denom = dcg(ideal)
    return dcg(predicted) / denom if denom else 0.0


def structured_relevance(article, intent) -> float:
    score = 0.0
    if intent.product_type and article.product_type_name == intent.product_type:
        score += 2.0
    if intent.color and _color_match(article, intent.color):
        score += 1.0
    if intent.section and section_family(article.section_name) == intent.section:
        score += 1.0
    return score


def generate_structured_queries(articles, limit: int = 80) -> list[str]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for article in articles.values():
        if article.colour_group_name and article.product_type_name:
            counts[(article.colour_group_name, article.product_type_name)] += 1
    popular = [
        pair
        for pair, count in sorted(counts.items(), key=lambda item: -item[1])
        if count >= 40
    ]
    queries = []
    for color, product_type in popular[:limit]:
        queries.append(f"{color.lower()} {product_type.lower()}")
    return queries


def evaluate_queries(engine: SearchEngine, queries: list[str], articles) -> dict[str, float]:
    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []
    for query in queries:
        intent = parse_query(query)
        relevant = {
            article_id
            for article_id, article in articles.items()
            if structured_relevance(article, intent) >= 3.0
        }
        if not relevant:
            continue
        payload = engine.search(query, limit=50)
        predicted = [row["article_id"] for row in payload["results"]]
        hits10 = len(set(predicted[:10]).intersection(relevant))
        hits50 = len(set(predicted[:50]).intersection(relevant))
        precisions.append(hits10 / 10)
        recalls.append(hits50 / min(len(relevant), 50))
        graded = [
            1.0 if article_id in relevant else 0.0 for article_id in predicted[:10]
        ]
        ndcgs.append(ndcg(graded, 10))
        rank = next(
            (
                index
                for index, article_id in enumerate(predicted, start=1)
                if article_id in relevant
            ),
            None,
        )
        mrrs.append(0.0 if rank is None else 1.0 / rank)
    n = len(precisions)
    return {
        "queries_scored": n,
        "precision_at_10": sum(precisions) / n if n else 0.0,
        "recall_at_50": sum(recalls) / n if n else 0.0,
        "ndcg_at_10": sum(ndcgs) / n if n else 0.0,
        "mrr": sum(mrrs) / n if n else 0.0,
    }


def main() -> None:
    ensure_directories()
    articles = load_articles()
    print(f"Loaded {len(articles):,} articles", flush=True)
    lexical = BM25Index()
    lexical.build(articles)
    structured = StructuredIndex(articles)
    semantic = SemanticIndex.load(load_encoder=True)
    engine = SearchEngine(articles, lexical, semantic, structured)
    structured_queries = generate_structured_queries(articles)
    holdout = structured_queries[::5]
    development = [query for query in structured_queries if query not in set(holdout)]
    report = {
        "label": "synthetic search benchmark",
        "not": "real user search quality",
        "structured_development": evaluate_queries(engine, development, articles),
        "structured_holdout": evaluate_queries(engine, holdout, articles),
        "curated_spotcheck": [
            {
                "query": query,
                "intent": parse_query(query).to_dict(),
                "top5": [
                    {
                        "name": row["name"],
                        "type": row["product_type"],
                        "color": row["color"],
                        "section": row["section"],
                    }
                    for row in engine.search(query, limit=5)["results"]
                ],
            }
            for query in CURATED_QUERIES
        ],
    }
    path = OVERNIGHT_DIR / "search_synthetic_eval.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "development": report["structured_development"],
        "holdout": report["structured_holdout"],
        "path": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
