"""Query embedding search against the frozen MiniLM article matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from recsys_loom.overnight.protocol import ROOT

TEXT_DIR = ROOT / "artifacts" / "text_retrieval"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticIndex:
    def __init__(
        self,
        embeddings: NDArray[np.floating],
        article_ids: list[str],
        model: SentenceTransformer | None = None,
    ):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.article_ids = article_ids
        self.model = model

    @classmethod
    def load(
        cls,
        directory: Path | None = None,
        load_encoder: bool = True,
    ) -> "SemanticIndex":
        text_dir = directory or TEXT_DIR
        embeddings = np.load(text_dir / "text_embeddings.f32.npy", mmap_mode="r")
        article_ids = np.load(text_dir / "text_article_ids.npy").astype(str).tolist()
        model = SentenceTransformer(ENCODER_NAME) if load_encoder else None
        return cls(embeddings, article_ids, model)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        if self.model is None:
            self.model = SentenceTransformer(ENCODER_NAME)
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        query_vector = self.encode_query(query)
        scores = np.asarray(self.embeddings) @ query_vector
        take = min(k, len(scores))
        local = np.argpartition(scores, -take)[-take:]
        local = local[np.argsort(-scores[local])]
        return [(self.article_ids[int(index)], float(scores[int(index)])) for index in local]
