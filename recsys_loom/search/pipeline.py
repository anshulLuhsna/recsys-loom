"""Two-stage hybrid search: multi-source retrieval, then query-first fusion."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

from recsys_loom.search.catalog import Article
from recsys_loom.search.intent import (
    ParsedIntent,
    STYLE_TERMS,
    TAXONOMY_ALIASES,
    merge_soft_intent,
    parse_query,
    tokenize,
)
from recsys_loom.search.lexical import BM25Index
from recsys_loom.search.llm import CatalogSearchLLM
from recsys_loom.search.structured import (
    StructuredIndex,
    attribute_score,
    passes_hard_filters,
)


class RetrievalIndex(Protocol):
    def search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        """Return article identifiers with source-native scores."""


LLM_FILLER_TOKENS = {
    "a",
    "an",
    "and",
    "can",
    "could",
    "do",
    "don",
    "for",
    "hey",
    "i",
    "isn",
    "like",
    "looking",
    "me",
    "need",
    "of",
    "or",
    "please",
    "show",
    "some",
    "something",
    "that",
    "the",
    "this",
    "t",
    "to",
    "too",
    "want",
    "would",
    "with",
    "you",
}


def needs_llm_enrichment(intent: ParsedIntent) -> bool:
    known = TAXONOMY_ALIASES | STYLE_TERMS | LLM_FILLER_TOKENS
    return any(token not in known for token in tokenize(intent.semantic_query))


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
        semantic: RetrievalIndex | None,
        structured: StructuredIndex,
        visual: RetrievalIndex | None = None,
        llm: CatalogSearchLLM | None = None,
        use_llm_intent: bool = False,
        use_llm_rerank: bool = True,
    ):
        self.articles = articles
        self.lexical = lexical
        self.semantic = semantic
        self.structured = structured
        self.visual = visual
        self.llm = llm
        self.use_llm_intent = use_llm_intent
        self.use_llm_rerank = use_llm_rerank

    def search(
        self,
        query: str,
        limit: int = 20,
        personalization: dict[str, float] | None = None,
    ) -> dict[str, object]:
        started = time.perf_counter()
        parse_started = time.perf_counter()
        deterministic_intent = parse_query(query)
        llm_enrichment_needed = needs_llm_enrichment(deterministic_intent)
        parse_ms = (time.perf_counter() - parse_started) * 1000
        pre_rank_limit = limit
        if (
            self.llm is not None
            and self.use_llm_rerank
            and llm_enrichment_needed
        ):
            pre_rank_limit = max(limit, self.llm.config.rerank_limit)

        raw_retrieve_started = time.perf_counter()
        bm25 = self.lexical.search(query, k=200)
        raw_semantic = (
            self.semantic.search(deterministic_intent.semantic_query or query, k=200)
            if self.semantic is not None
            else []
        )
        raw_visual = (
            self.visual.search(deterministic_intent.semantic_query or query, k=200)
            if self.visual is not None
            else []
        )
        structured = self.structured.retrieve(deterministic_intent, k=400)
        raw_pre_ranked = fuse(
            deterministic_intent,
            self.articles,
            bm25,
            raw_semantic,
            structured,
            {},
            limit=pre_rank_limit,
            visual=raw_visual,
        )
        retrieve_ms = (time.perf_counter() - raw_retrieve_started) * 1000

        intent_started = time.perf_counter()
        intent = deterministic_intent
        semantic_enrichment_applied = False
        intent_diagnostics = _inactive_llm_stage("not_configured")
        if (
            self.llm is not None
            and self.use_llm_intent
            and llm_enrichment_needed
        ):
            soft_intent, intent_diagnostics = self.llm.enrich_intent(
                deterministic_intent
            )
            if soft_intent is not None:
                intent = merge_soft_intent(deterministic_intent, soft_intent)
                raw_tokens = set(deterministic_intent.tokens)
                normalized_raw = " ".join(deterministic_intent.tokens)
                reformulation_changed = bool(
                    soft_intent.reformulated_query
                    and " ".join(tokenize(soft_intent.reformulated_query))
                    != normalized_raw
                )
                style_changed = any(
                    token not in raw_tokens
                    for term in soft_intent.style_terms
                    for token in tokenize(term)
                )
                semantic_enrichment_applied = reformulation_changed or style_changed
                if not semantic_enrichment_applied:
                    intent.semantic_query = deterministic_intent.semantic_query
        elif self.llm is not None and self.use_llm_intent:
            intent_diagnostics = _inactive_llm_stage(
                "deterministic_intent_complete"
            )
        elif self.llm is not None:
            intent_diagnostics = _inactive_llm_stage("stage_disabled")
        provider_failed = bool(
            intent_diagnostics["attempted"]
            and not intent_diagnostics["applied"]
        )
        intent_ms = parse_ms + (time.perf_counter() - intent_started) * 1000
        semantic_query = intent.semantic_query or query
        pre_ranked = raw_pre_ranked
        if semantic_enrichment_applied:
            enriched_retrieve_started = time.perf_counter()
            semantic = (
                self.semantic.search(semantic_query, k=200)
                if self.semantic is not None
                else []
            )
            visual = (
                self.visual.search(semantic_query, k=200)
                if self.visual is not None
                else []
            )
            pre_ranked = fuse(
                intent,
                self.articles,
                bm25,
                semantic,
                structured,
                {},
                limit=pre_rank_limit,
                visual=visual,
            )
            retrieve_ms += (
                time.perf_counter() - enriched_retrieve_started
            ) * 1000

        rank_started = time.perf_counter()
        rerank_diagnostics = _inactive_llm_stage(
            "not_configured",
            candidate_count=min(len(pre_ranked), pre_rank_limit),
        )
        reranked_ids = None
        if (
            self.llm is not None
            and self.use_llm_rerank
            and llm_enrichment_needed
            and not provider_failed
        ):
            reranked_ids, rerank_diagnostics = self.llm.rerank(
                query,
                intent,
                [result.article_id for result in pre_ranked],
                self.articles,
            )
        elif self.llm is not None and self.use_llm_rerank:
            rerank_diagnostics = _inactive_llm_stage(
                (
                    "prior_llm_failure"
                    if provider_failed
                    else "deterministic_intent_complete"
                ),
                candidate_count=min(
                    len(pre_ranked),
                    self.llm.config.rerank_limit,
                ),
            )
        elif self.llm is not None:
            rerank_diagnostics = _inactive_llm_stage(
                "stage_disabled",
                candidate_count=min(
                    len(pre_ranked),
                    self.llm.config.rerank_limit,
                ),
            )
        enrichment_succeeded = bool(intent_diagnostics["applied"])
        rerank_failed_after_enrichment = bool(
            enrichment_succeeded
            and self.llm is not None
            and self.use_llm_rerank
            and not rerank_diagnostics["applied"]
        )
        if rerank_failed_after_enrichment:
            intent = deterministic_intent
            semantic_enrichment_applied = False
            query_ordered = raw_pre_ranked
            reranked_ids = None
            fallback = "raw_query_pre_rank_after_llm_rerank_failure"
        else:
            query_ordered = _validated_llm_order(
                pre_ranked,
                reranked_ids,
                self.articles,
                intent,
                pre_rank_limit,
            )
            if provider_failed:
                fallback = "raw_query_pre_rank_after_llm_intent_failure"
            elif self.llm is None or not self.llm.available:
                fallback = "deterministic_raw_query_pre_rank"
            elif (
                self.use_llm_rerank
                and llm_enrichment_needed
                and not rerank_diagnostics["applied"]
            ):
                fallback = "raw_query_pre_rank_after_llm_rerank_failure"
            else:
                fallback = None
        intent_diagnostics["used_for_results"] = bool(
            enrichment_succeeded and not rerank_failed_after_enrichment
        )
        rerank_diagnostics["used_for_results"] = bool(reranked_ids)
        results = _apply_local_personalization(
            query_ordered,
            personalization or {},
            limit,
            llm_order_applied=bool(reranked_ids),
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
            "diagnostics": {
                "approach": "GenRec-inspired catalog-grounded search",
                "llm": (
                    self.llm.diagnostics()
                    if self.llm is not None
                    else {
                        "status": "not_configured",
                        "enabled": False,
                        "configured": False,
                        "customer_history_in_prompt": False,
                    }
                ),
                "intent_enrichment": intent_diagnostics,
                "candidate_rerank": rerank_diagnostics,
                "retrieval_queries": {
                    "bm25": "raw_query",
                    "semantic_visual": (
                        "llm_enriched"
                        if semantic_enrichment_applied
                        else (
                            "deterministic_normalized"
                            if deterministic_intent.semantic_query.lower()
                            != query.strip().lower()
                            else "raw_query"
                        )
                    ),
                },
                "hard_constraints_source": "deterministic_parser",
                "hard_filters_reapplied_after_llm": True,
                "personalization_stage": "local_after_llm",
                "fallback": fallback,
            },
            "latency_ms": {
                "intent": intent_ms,
                "retrieval": retrieve_ms,
                "ranking": rank_ms,
                "llm_intent": float(intent_diagnostics["latency_ms"]),
                "llm_rerank": float(rerank_diagnostics["latency_ms"]),
                "total": (time.perf_counter() - started) * 1000,
            },
        }


def _inactive_llm_stage(
    reason: str,
    candidate_count: int | None = None,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "attempted": False,
        "applied": False,
        "cache_hit": False,
        "fallback_reason": reason,
        "latency_ms": 0.0,
    }
    if candidate_count is not None:
        diagnostics.update(
            {
                "candidate_count": candidate_count,
                "returned_count": 0,
                "invalid_id_count": 0,
            }
        )
    return diagnostics


def _validated_llm_order(
    pre_ranked: list[SearchResult],
    reranked_ids: list[str] | None,
    articles: dict[str, Article],
    intent: ParsedIntent,
    limit: int,
) -> list[SearchResult]:
    by_id = {result.article_id: result for result in pre_ranked}
    ordered: list[SearchResult] = []
    seen: set[str] = set()
    for article_id in reranked_ids or []:
        result = by_id.get(article_id)
        article = articles.get(article_id)
        if result is None or article is None or not passes_hard_filters(article, intent):
            continue
        if article_id in seen:
            continue
        result.signals["llm_rerank"] = 1.0 / (len(ordered) + 1)
        result.sources = [*result.sources, "llm_rerank"]
        ordered.append(result)
        seen.add(article_id)
    for result in pre_ranked:
        article = articles.get(result.article_id)
        if (
            result.article_id in seen
            or article is None
            or not passes_hard_filters(article, intent)
        ):
            continue
        ordered.append(result)
        seen.add(result.article_id)
    return ordered[:limit]


def _apply_local_personalization(
    query_ordered: list[SearchResult],
    personalization: dict[str, float],
    limit: int,
    llm_order_applied: bool,
) -> list[SearchResult]:
    personal_norm = _normalize(personalization)
    if not personal_norm:
        return query_ordered[:limit]
    slate_size = max(len(query_ordered), 1)
    locally_scored: list[tuple[float, int, SearchResult]] = []
    for index, result in enumerate(query_ordered):
        boost = 0.08 * personal_norm.get(result.article_id, 0.0)
        result.signals["personalization"] = boost
        result.score += boost
        base_order = (
            1.0 - (index / slate_size)
            if llm_order_applied
            else result.signals["query_relevance"]
        )
        locally_scored.append((base_order + boost, index, result))
    locally_scored.sort(key=lambda row: (-row[0], row[1], row[2].article_id))
    return [row[2] for row in locally_scored[:limit]]


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
