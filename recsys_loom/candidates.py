"""Candidate provenance schema and utilities for retrieval evaluation.

Every candidate source must produce CandidateRecord objects so that
downstream merging, deduplication, and attribution work uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One article retrieved by one source for one customer."""

    customer_id: str
    article_id: str
    source_name: str
    source_rank: int
    source_score: float
    model_version: str = ""
    feature_cutoff: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "article_id": self.article_id,
            "source_name": self.source_name,
            "source_rank": self.source_rank,
            "source_score": self.source_score,
            "model_version": self.model_version,
            "feature_cutoff": self.feature_cutoff,
        }


@dataclass(slots=True)
class MergedCandidate:
    """An article after deduplication across sources, retaining all provenance."""

    customer_id: str
    article_id: str
    sources: list[CandidateRecord] = field(default_factory=list)

    @property
    def best_rank(self) -> int:
        return min(s.source_rank for s in self.sources)

    @property
    def source_names(self) -> set[str]:
        return {s.source_name for s in self.sources}


def merge_candidates(
    *source_lists: list[CandidateRecord],
) -> dict[str, list[MergedCandidate]]:
    """Merge candidate lists from multiple sources, deduplicating by article.

    Returns a dict mapping customer_id to a list of MergedCandidates,
    sorted by best source rank (lowest first).
    """
    by_customer: dict[str, dict[str, MergedCandidate]] = {}

    for source_list in source_lists:
        for record in source_list:
            customer_articles = by_customer.setdefault(record.customer_id, {})
            if record.article_id not in customer_articles:
                customer_articles[record.article_id] = MergedCandidate(
                    customer_id=record.customer_id,
                    article_id=record.article_id,
                )
            customer_articles[record.article_id].sources.append(record)

    result: dict[str, list[MergedCandidate]] = {}
    for customer_id, articles in by_customer.items():
        merged = sorted(articles.values(), key=lambda m: m.best_rank)
        result[customer_id] = merged

    return result


def candidates_to_article_lists(
    merged: dict[str, list[MergedCandidate]],
) -> dict[str, list[str]]:
    """Extract ordered article ID lists from merged candidates."""
    return {
        customer_id: [m.article_id for m in candidates]
        for customer_id, candidates in merged.items()
    }


def source_attribution(
    merged: dict[str, list[MergedCandidate]],
    source_names: list[str],
) -> dict[str, dict[str, int]]:
    """Count how many candidates each source contributed per customer.

    Returns {customer_id: {source_name: count}}.
    """
    result: dict[str, dict[str, int]] = {}
    for customer_id, candidates in merged.items():
        counts: dict[str, int] = {name: 0 for name in source_names}
        for candidate in candidates:
            for name in candidate.source_names:
                counts[name] = counts.get(name, 0) + 1
        result[customer_id] = counts
    return result
