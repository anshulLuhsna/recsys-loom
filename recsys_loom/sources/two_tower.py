"""FAISS retrieval adapter for learned two-tower embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from numpy.typing import NDArray

from recsys_loom.candidates import CandidateRecord


@dataclass(slots=True)
class TwoTowerIndex:
    """Cosine-similarity IVF index over eligible article embeddings."""

    index: faiss.Index
    article_indices: NDArray[np.int64]
    article_ids: list[str]
    embedding_dim: int
    nlist: int
    nprobe: int

    @classmethod
    def build(
        cls,
        item_embeddings: NDArray[np.float32],
        eligible_item_indices: NDArray[np.int64],
        all_article_ids: list[str],
        nlist: int = 256,
        nprobe: int = 64,
    ) -> TwoTowerIndex:
        eligible_embeddings = np.ascontiguousarray(
            item_embeddings[eligible_item_indices],
            dtype=np.float32,
        )
        dimension = eligible_embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            min(nlist, len(eligible_embeddings)),
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(eligible_embeddings)
        index.add(eligible_embeddings)
        index.nprobe = min(nprobe, index.nlist)
        return cls(
            index=index,
            article_indices=eligible_item_indices.copy(),
            article_ids=all_article_ids,
            embedding_dim=dimension,
            nlist=int(index.nlist),
            nprobe=int(index.nprobe),
        )

    def search(
        self,
        user_embeddings: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        scores, local_indices = self.index.search(
            np.ascontiguousarray(user_embeddings, dtype=np.float32),
            min(k, self.index.ntotal),
        )
        catalog_indices = np.full_like(local_indices, fill_value=-1)
        valid = local_indices >= 0
        catalog_indices[valid] = self.article_indices[local_indices[valid]]
        return scores, catalog_indices

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)


def two_tower_candidates(
    customer_ids: list[str],
    user_embeddings: NDArray[np.float32],
    index: TwoTowerIndex,
    cutoff: str,
    k: int = 500,
    model_version: str = "two_tower_v1",
) -> list[CandidateRecord]:
    """Retrieve and format two-tower candidates with full provenance."""
    scores, article_indices = index.search(user_embeddings, k)
    records: list[CandidateRecord] = []
    for customer_index, customer_id in enumerate(customer_ids):
        for rank, (score, article_index) in enumerate(
            zip(scores[customer_index], article_indices[customer_index]),
            start=1,
        ):
            if article_index < 0:
                continue
            records.append(
                CandidateRecord(
                    customer_id=customer_id,
                    article_id=index.article_ids[int(article_index)],
                    source_name="two_tower",
                    source_rank=rank,
                    source_score=float(score),
                    model_version=model_version,
                    feature_cutoff=cutoff,
                )
            )
    return records


def ann_exact_diagnostics(
    index: TwoTowerIndex,
    user_embeddings: NDArray[np.float32],
    item_embeddings: NDArray[np.float32],
    k: int = 20,
    max_users: int = 50,
) -> dict[str, float]:
    """Compare HNSW with exact search, accounting for tied metadata embeddings."""
    users = user_embeddings[:max_users]
    ann_scores, ann_indices = index.search(users, k)

    eligible_embeddings = item_embeddings[index.article_indices]
    exact_scores = users @ eligible_embeddings.T
    exact_local = np.argpartition(-exact_scores, kth=k - 1, axis=1)[:, :k]
    exact_indices = index.article_indices[exact_local]
    exact_top_scores = np.take_along_axis(exact_scores, exact_local, axis=1)

    overlaps = [
        len(set(ann_indices[row]).intersection(exact_indices[row])) / k
        for row in range(len(users))
    ]
    exact_kth = np.min(exact_top_scores, axis=1)
    ann_kth = np.min(ann_scores, axis=1)
    score_regret = np.maximum(exact_kth - ann_kth, 0.0)
    return {
        "top_k_id_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_kth_score_regret": float(np.mean(score_regret)),
        "queries_with_kth_score_within_1e_4": float(
            np.mean(score_regret <= 1e-4)
        ),
    }
