"""Attribute inverted lists for structured search retrieval."""

from __future__ import annotations

from collections import defaultdict

from recsys_loom.search.catalog import Article, section_family
from recsys_loom.search.intent import ParsedIntent


class StructuredIndex:
    def __init__(self, articles: dict[str, Article]):
        self.by_type: dict[str, list[str]] = defaultdict(list)
        self.by_color: dict[str, list[str]] = defaultdict(list)
        self.by_section_family: dict[str, list[str]] = defaultdict(list)
        self.articles = articles
        for article_id, article in articles.items():
            if article.product_type_name:
                self.by_type[article.product_type_name].append(article_id)
            if article.colour_group_name:
                self.by_color[article.colour_group_name].append(article_id)
            family = section_family(article.section_name)
            if family:
                self.by_section_family[family].append(article_id)

    def retrieve(self, intent: ParsedIntent, k: int = 400) -> list[tuple[str, float]]:
        pools: list[set[str]] = []
        if intent.product_type and intent.product_type in self.by_type:
            pools.append(set(self.by_type[intent.product_type]))
        if intent.color and intent.color in self.by_color:
            pools.append(set(self.by_color[intent.color]))
        if intent.section and intent.section in self.by_section_family:
            pools.append(set(self.by_section_family[intent.section]))
        if not pools:
            return []
        matched = set.intersection(*pools) if len(pools) > 1 else pools[0]
        scored = []
        for article_id in matched:
            article = self.articles[article_id]
            score = 0.0
            if intent.product_type and article.product_type_name == intent.product_type:
                score += 2.0
            if intent.color and _color_match(article, intent.color):
                score += 1.0
            if intent.section and section_family(article.section_name) == intent.section:
                score += 1.0
            scored.append((article_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]


def _color_match(article: Article, color: str) -> bool:
    return color.lower() in {
        article.colour_group_name.lower(),
        article.perceived_colour_master_name.lower(),
    }


def attribute_score(article: Article, intent: ParsedIntent) -> float:
    score = 0.0
    if intent.product_type and article.product_type_name == intent.product_type:
        score += 1.0
    if intent.color and _color_match(article, intent.color):
        score += 0.6
    if intent.section and section_family(article.section_name) == intent.section:
        score += 0.4
    return score


def passes_hard_filters(article: Article, intent: ParsedIntent) -> bool:
    if "product_type" in intent.hard_constraints:
        if article.product_type_name != intent.hard_constraints["product_type"]:
            return False
    if "color" in intent.hard_constraints and not _color_match(
        article, intent.hard_constraints["color"]
    ):
        return False
    if "section_family" in intent.hard_constraints:
        family = section_family(article.section_name)
        required = intent.hard_constraints["section_family"]
        if family != required:
            return False
    return True
