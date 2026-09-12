"""Inverted-index BM25 over article search text."""

from __future__ import annotations

import math
from collections import defaultdict

from recsys_loom.search.catalog import Article
from recsys_loom.search.intent import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.article_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avg_len = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}

    def build(self, articles: dict[str, Article]) -> None:
        self.article_ids = list(articles)
        self.doc_len = []
        tf_maps: list[dict[str, int]] = []
        df: dict[str, int] = defaultdict(int)
        for article_id in self.article_ids:
            counts: dict[str, int] = defaultdict(int)
            for token in tokenize(articles[article_id].search_text()):
                counts[token] += 1
            self.doc_len.append(sum(counts.values()))
            tf_maps.append(counts)
            for token in counts:
                df[token] += 1
        n_docs = max(len(self.article_ids), 1)
        self.avg_len = (sum(self.doc_len) / n_docs) if n_docs else 0.0
        self.idf = {
            token: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for token, count in df.items()
        }
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_index, counts in enumerate(tf_maps):
            for token, frequency in counts.items():
                postings[token].append((doc_index, frequency))
        self.postings = dict(postings)

    def search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for token in tokenize(query):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for doc_index, frequency in self.postings.get(token, []):
                denom = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc_index] / max(self.avg_len, 1e-6)
                )
                scores[doc_index] += idf * (frequency * (self.k1 + 1.0) / denom)
        ranked = sorted(scores.items(), key=lambda item: -item[1])[:k]
        return [(self.article_ids[doc_index], score) for doc_index, score in ranked]
