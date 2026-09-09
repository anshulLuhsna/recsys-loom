"""Build a search engine with optional semantic and visual retrieval."""

from __future__ import annotations

from recsys_loom.search.catalog import load_articles
from recsys_loom.search.lexical import BM25Index
from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.semantic import SemanticIndex
from recsys_loom.search.structured import StructuredIndex
from recsys_loom.search.visual import VisualIndex


def build_search_engine(
    load_semantic: bool = False,
    load_visual: bool = False,
) -> SearchEngine:
    articles = load_articles()
    lexical = BM25Index()
    lexical.build(articles)
    semantic = SemanticIndex.load(load_encoder=True) if load_semantic else None
    visual = VisualIndex.load(load_encoder=True) if load_visual else None
    return SearchEngine(
        articles,
        lexical,
        semantic,
        StructuredIndex(articles),
        visual,
    )
