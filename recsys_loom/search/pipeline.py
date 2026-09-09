"""Two-stage hybrid search: multi-source retrieval, then query-first fusion."""

from __future__ import annotations

from dataclasses import dataclass
import time

from recsys_loom.search.catalog import Article
from recsys_loom.search.intent import ParsedIntent, parse_query
from recsys_loom.search.lexical import BM25Index
from recsys_loom.search.semantic import SemanticIndex
from recsys_loom.search.structured import (
    StructuredIndex,
    attribute_score,
    passes_hard_filters,
)
from recsys_loom.search.visual import VisualIndex


@dataclass(slots=True)
class SearchResult:
    article_id: str
    score: float
    sources: list[str]
    signals: dict[str, float]


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    maximum = max(values.values())
    if maximum <= 0:
        return {key: 0.0 for key in values}
    return {key: value / maximum for key, value in values.items()}


class SearchEngine:
    def __init__(
        self,
        articles: dict[str, Article],
        lexical: BM25Index,
        semantic: SemanticIndex | None,
        structured: StructuredIndex,
        visual: VisualIndex | None = None,
    ):
        self.articles = articles
        self.lexical = lexical
        self.semantic = semantic
        self.structured = structured
        self.visual = visual

    def search(
        self,
        query: str,
        limit: int = 20,
        personalization: dict[str, float] | None = None,
    ) -> dict[str, object]:
        started = time.perf_counter()
        intent = parse_query(query)
        intent_ms = (time.perf_counter() - started) * 1000
        retrieve_started = time.perf_counter()
        bm25 = self.lexical.search(intent.free_text or query, k=200)
        semantic = (
            self.semantic.search(intent.free_text or query, k=200)
            if self.semantic is not None
            else []
        )
        visual = self.visual.search(query, k=200) if self.visual is not None else []
        structured = self.structured.retrieve(intent, k=400)
        retrieve_ms = (time.perf_counter() - retrieve_started) * 1000
        rank_started = time.perf_counter()
        results = fuse(
            intent,
            self.articles,
            bm25,
            semantic,
            structured,
            personalization or {},
            limit=limit,
            visual=visual,
        )
        rank_ms = (time.perf_counter() - rank_started) * 1000
        cards = []
        for result in results:
            article = self.articles[result.article_id]
            card = article.to_card()
            card["score"] = result.score
            card["sources"] = result.sources
            card["signals"] = result.signals
            cards.append(card)
        return {
            "query": query,
            "parsed_intent": intent.to_dict(),
            "results": cards,
            "latency_ms": {
                "intent": intent_ms,
                "retrieval": retrieve_ms,
                "ranking": rank_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        }


def fuse(
    intent: ParsedIntent,
    articles: dict[str, Article],
    bm25: list[tuple[str, float]],
    semantic: list[tuple[str, float]],
    structured: list[tuple[str, float]],
    personalization: dict[str, float],
    limit: int,
    visual: list[tuple[str, float]] | None = None,
) -> list[SearchResult]:
    visual = visual or []
    bm25_norm = _normalize({article_id: score for article_id, score in bm25})
    semantic_norm = _normalize({article_id: score for article_id, score in semantic})
    visual_norm = _normalize({article_id: score for article_id, score in visual})
    structured_norm = _normalize({article_id: score for article_id, score in structured})
    personal_norm = _normalize(personalization)
    ranks = {
        "bm25": {article_id: rank for rank, (article_id, _) in enumerate(bm25, start=1)},
        "semantic": {
            article_id: rank for rank, (article_id, _) in enumerate(semantic, start=1)
        },
        "visual": {
            article_id: rank for rank, (article_id, _) in enumerate(visual, start=1)
        },
        "structured": {
            article_id: rank for rank, (article_id, _) in enumerate(structured, start=1)
        },
    }
    candidates = (
        set(bm25_norm)
        | set(semantic_norm)
        | set(visual_norm)
        | set(structured_norm)
    )
    if not candidates and personal_norm:
        candidates = set(list(personal_norm)[:50])
    scored: list[SearchResult] = []
    for article_id in candidates:
        article = articles.get(article_id)
        if article is None:
            continue
        if not passes_hard_filters(article, intent):
            continue
        sources = [
            name
            for name, ranking in ranks.items()
            if article_id in ranking
        ]
        query_score = (
            0.45 * bm25_norm.get(article_id, 0.0)
            + 0.35 * semantic_norm.get(article_id, 0.0)
            + 0.20 * structured_norm.get(article_id, 0.0)
            + 0.25 * visual_norm.get(article_id, 0.0)
            + 0.15 * attribute_score(article, intent)
            + _rrf(ranks["bm25"].get(article_id, 10_000))
            + _rrf(ranks["semantic"].get(article_id, 10_000))
        )
        personal = 0.08 * personal_norm.get(article_id, 0.0)
        final = query_score + personal
        scored.append(
            SearchResult(
                article_id=article_id,
                score=final,
                sources=sources or ["personalization"],
                signals={
                    "bm25": bm25_norm.get(article_id, 0.0),
                    "semantic": semantic_norm.get(article_id, 0.0),
                    "visual": visual_norm.get(article_id, 0.0),
                    "structured": structured_norm.get(article_id, 0.0),
                    "attributes": attribute_score(article, intent),
                    "personalization": personal,
                    "query_relevance": query_score,
                },
            )
        )
    scored.sort(key=lambda item: (-item.score, item.article_id))
    return scored[:limit]
