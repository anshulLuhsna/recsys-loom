#!/usr/bin/env python3
"""Weakly supervised search LambdaRank on synthetic structured queries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.protocol import OVERNIGHT_DIR, ensure_directories
from recsys_loom.search.catalog import load_articles, section_family
from recsys_loom.search.intent import parse_query
from recsys_loom.search.lexical import BM25Index
from recsys_loom.search.pipeline import SearchEngine, fuse
from recsys_loom.search.semantic import SemanticIndex
from recsys_loom.search.benchmark import generate_structured_queries, ndcg
from recsys_loom.search.structured import StructuredIndex, _color_match, attribute_score


def relevance_label(article, intent) -> int:
    score = 0
    if intent.product_type and article.product_type_name == intent.product_type:
        score += 2
    if intent.color and _color_match(article, intent.color):
        score += 1
    if intent.section and section_family(article.section_name) == intent.section:
        score += 1
    return score


def pair_features(article, intent, signals: dict[str, float]) -> list[float]:
    return [
        signals.get("bm25", 0.0),
        signals.get("semantic", 0.0),
        signals.get("structured", 0.0),
        attribute_score(article, intent),
        1.0 if intent.product_type == article.product_type_name else 0.0,
        1.0 if intent.color and _color_match(article, intent.color) else 0.0,
        1.0 if intent.section and section_family(article.section_name) == intent.section else 0.0,
        len(set(article.search_text().lower().split()).intersection(intent.tokens)),
    ]


def main() -> None:
    ensure_directories()
    articles = load_articles()
    lexical = BM25Index()
    lexical.build(articles)
    structured = StructuredIndex(articles)
    load_semantic = "--lexical-only" not in sys.argv
    semantic = SemanticIndex.load(load_encoder=True) if load_semantic else None
    engine = SearchEngine(articles, lexical, semantic, structured)
    queries = generate_structured_queries(articles, limit=60)
    holdout = queries[::5]
    train_queries = [query for query in queries if query not in set(holdout)]

    features = []
    labels = []
    groups = []
    for query in train_queries:
        intent = parse_query(query)
        fused = fuse(
            intent,
            articles,
            lexical.search(query, 80),
            semantic.search(query, 80) if semantic is not None else [],
            structured.retrieve(intent, 80),
            {},
            limit=80,
        )
        if not fused:
            continue
        for result in fused:
            article = articles[result.article_id]
            features.append(pair_features(article, intent, result.signals))
            labels.append(relevance_label(article, intent))
        groups.append(len(fused))

    train = lgb.Dataset(
        np.asarray(features, dtype=np.float32),
        label=np.asarray(labels, dtype=np.float32),
        group=groups,
    )
    model = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "verbose": -1,
        },
        train,
        num_boost_round=80,
    )

    hybrid_ndcg = []
    model_ndcg = []
    for query in holdout:
        payload = engine.search(query, limit=10)
        intent = parse_query(query)
        hybrid_rel = [
            1.0 if relevance_label(articles[row["article_id"]], intent) >= 3 else 0.0
            for row in payload["results"]
        ]
        hybrid_ndcg.append(ndcg(hybrid_rel, 10))
        fused = fuse(
            intent,
            articles,
            lexical.search(query, 80),
            semantic.search(query, 80) if semantic is not None else [],
            structured.retrieve(intent, 80),
            {},
            limit=80,
        )
        scores = [
            float(
                model.predict(
                    np.asarray(
                        [pair_features(articles[result.article_id], intent, result.signals)],
                        dtype=np.float32,
                    )
                )[0]
            )
            for result in fused
        ]
        order = np.argsort(-np.asarray(scores))[:10]
        model_rel = [
            1.0 if relevance_label(articles[fused[int(index)].article_id], intent) >= 3 else 0.0
            for index in order
        ]
        model_ndcg.append(ndcg(model_rel, 10))

    hybrid_mean = float(np.mean(hybrid_ndcg)) if hybrid_ndcg else 0.0
    model_mean = float(np.mean(model_ndcg)) if model_ndcg else 0.0
    keep = model_mean > hybrid_mean + 0.01
    report = {
        "label": "synthetic weakly supervised search ranker",
        "not": "real user search quality",
        "hybrid_ndcg_at_10": hybrid_mean,
        "ranker_ndcg_at_10": model_mean,
        "selected": "search_lambdarank" if keep else "hybrid_fusion",
        "holdout_queries": len(holdout),
    }
    path = OVERNIGHT_DIR / "search_ranker.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
