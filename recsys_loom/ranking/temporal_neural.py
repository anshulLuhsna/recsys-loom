"""Memory-conscious neural ranking over temporal two-tower embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from recsys_loom.metrics import average_precision_at_k


@dataclass(slots=True)
class PairEmbeddingStore:
    """Resolve pair rows to snapshot-safe user and item vectors on demand."""

    user_embeddings: NDArray[np.float32]
    item_embeddings: NDArray[np.float32]
    pair_user_indices: NDArray[np.int32]
    pair_item_indices: NDArray[np.int32]

    @property
    def context_dim(self) -> int:
        return int(self.user_embeddings.shape[1] + self.item_embeddings.shape[1])

    def get(self, row_indices: NDArray[np.int64]) -> torch.Tensor:
        user = self.user_embeddings[self.pair_user_indices[row_indices]]
        item = self.item_embeddings[self.pair_item_indices[row_indices]]
        return torch.from_numpy(np.concatenate([user, item], axis=1)).float()


@dataclass
class TemporalTrainingLog:
    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    validation_maps: list[float] = field(default_factory=list)
    best_epoch: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "validation_losses": self.validation_losses,
            "validation_maps": self.validation_maps,
            "best_epoch": self.best_epoch,
        }


def build_group_bpr_pairs(
    labels: NDArray[np.int8],
    groups: NDArray[np.int32],
    features: NDArray[np.float32],
    feature_names: list[str],
    hard_scores: NDArray[np.float64] | None,
    neg_per_pos: int,
    hard_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Mix random negatives with top candidates from each retrieval/LambdaRank signal."""
    rng = np.random.default_rng(seed)
    score_columns = [
        index for index, name in enumerate(feature_names)
        if name.startswith("score_")
    ]
    hard_count = int(round(neg_per_pos * hard_fraction))
    random_count = neg_per_pos - hard_count
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    offset = 0

    for group_size in groups:
        end = offset + int(group_size)
        group_labels = labels[offset:end]
        positives = np.flatnonzero(group_labels == 1) + offset
        negatives = np.flatnonzero(group_labels == 0) + offset
        if len(positives) == 0 or len(negatives) == 0:
            offset = end
            continue

        hard_pool_parts: list[NDArray[np.int64]] = []
        per_signal = max(hard_count * 2, 20)
        for column in score_columns:
            values = np.nan_to_num(
                features[negatives, column],
                nan=-np.inf,
                neginf=-np.inf,
            )
            take = min(per_signal, len(negatives))
            selected = np.argpartition(values, -take)[-take:]
            hard_pool_parts.append(negatives[selected])
        if hard_scores is not None:
            values = hard_scores[negatives]
            take = min(per_signal, len(negatives))
            selected = np.argpartition(values, -take)[-take:]
            hard_pool_parts.append(negatives[selected])
        hard_pool = (
            np.unique(np.concatenate(hard_pool_parts))
            if hard_pool_parts
            else negatives
        )

        for positive in positives:
            sampled_hard = rng.choice(
                hard_pool,
                size=min(hard_count, len(hard_pool)),
                replace=False,
            )
            sampled_random = rng.choice(
                negatives,
                size=min(random_count, len(negatives)),
                replace=False,
            )
            selected_negatives = np.concatenate([sampled_hard, sampled_random])
            positive_rows.extend([int(positive)] * len(selected_negatives))
            negative_rows.extend(int(value) for value in selected_negatives)
        offset = end

    order = rng.permutation(len(positive_rows))
    return (
        np.asarray(positive_rows, dtype=np.int64)[order],
        np.asarray(negative_rows, dtype=np.int64)[order],
    )


def train_temporal_neural_ranker(
    model: nn.Module,
    train_features: NDArray[np.float32],
    train_categories: NDArray[np.int32],
    train_labels: NDArray[np.int8],
    train_groups: NDArray[np.int32],
    train_embeddings: PairEmbeddingStore,
    feature_names: list[str],
    validation_features: NDArray[np.float32],
    validation_categories: NDArray[np.int32],
    validation_labels: NDArray[np.int8],
    validation_groups: NDArray[np.int32],
    validation_embeddings: PairEmbeddingStore,
    validation_customer_ids: list[str],
    validation_article_indices: NDArray[np.int32],
    article_ids: list[str],
    validation_relevance: dict[str, set[str]],
    hard_scores: NDArray[np.float64] | None = None,
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    neg_per_pos: int = 20,
    hard_fraction: float = 0.5,
    patience: int = 6,
) -> tuple[nn.Module, TemporalTrainingLog]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    validation_positive, validation_negative = build_group_bpr_pairs(
        validation_labels,
        validation_groups,
        validation_features,
        feature_names,
        None,
        neg_per_pos=10,
        hard_fraction=0.5,
        seed=911,
    )
    log = TemporalTrainingLog()
    best_map = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    for epoch in range(epochs):
        positive_rows, negative_rows = build_group_bpr_pairs(
            train_labels,
            train_groups,
            train_features,
            feature_names,
            hard_scores,
            neg_per_pos,
            hard_fraction,
            seed=42 + epoch,
        )
        model.train()
        losses: list[float] = []
        for start in range(0, len(positive_rows), batch_size):
            positive = positive_rows[start : start + batch_size]
            negative = negative_rows[start : start + batch_size]
            positive_scores = _score_rows(
                model,
                train_features,
                train_categories,
                train_embeddings,
                positive,
            )
            negative_scores = _score_rows(
                model,
                train_features,
                train_categories,
                train_embeddings,
                negative,
            )
            loss = -F.logsigmoid(positive_scores - negative_scores).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        validation_loss = pairwise_loss(
            model,
            validation_features,
            validation_categories,
            validation_embeddings,
            validation_positive,
            validation_negative,
            batch_size,
        )
        validation_metrics = evaluate_temporal_ranker(
            model,
            validation_features,
            validation_categories,
            validation_groups,
            validation_embeddings,
            validation_customer_ids,
            validation_article_indices,
            article_ids,
            validation_relevance,
        )
        validation_map = validation_metrics["map_at_12"]
        log.epochs.append(epoch + 1)
        log.train_losses.append(float(np.mean(losses)))
        log.validation_losses.append(validation_loss)
        log.validation_maps.append(validation_map)
        print(
            f"    Epoch {epoch + 1:2d}: loss={log.train_losses[-1]:.4f} "
            f"val_loss={validation_loss:.4f} val_MAP@12={validation_map:.4f}",
            flush=True,
        )

        if validation_map > best_map:
            best_map = validation_map
            log.best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log


def fit_temporal_neural_fixed_epochs(
    model: nn.Module,
    features: NDArray[np.float32],
    categories: NDArray[np.int32],
    labels: NDArray[np.int8],
    groups: NDArray[np.int32],
    embeddings: PairEmbeddingStore,
    feature_names: list[str],
    epochs: int,
    hard_scores: NDArray[np.float64] | None = None,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    neg_per_pos: int = 20,
    hard_fraction: float = 0.5,
) -> tuple[nn.Module, list[float]]:
    """Refit on all safe training snapshots for a preselected epoch count."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        positive_rows, negative_rows = build_group_bpr_pairs(
            labels,
            groups,
            features,
            feature_names,
            hard_scores,
            neg_per_pos,
            hard_fraction,
            seed=1042 + epoch,
        )
        model.train()
        losses: list[float] = []
        for start in range(0, len(positive_rows), batch_size):
            positive = positive_rows[start : start + batch_size]
            negative = negative_rows[start : start + batch_size]
            difference = _score_rows(
                model, features, categories, embeddings, positive
            ) - _score_rows(
                model, features, categories, embeddings, negative
            )
            loss = -F.logsigmoid(difference).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses))
        epoch_losses.append(epoch_loss)
        print(
            f"    Refit epoch {epoch + 1:2d}/{epochs}: loss={epoch_loss:.4f}",
            flush=True,
        )
    return model, epoch_losses


def pairwise_loss(
    model: nn.Module,
    features: NDArray[np.float32],
    categories: NDArray[np.int32],
    embeddings: PairEmbeddingStore,
    positive_rows: NDArray[np.int64],
    negative_rows: NDArray[np.int64],
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(positive_rows), batch_size):
            positive = positive_rows[start : start + batch_size]
            negative = negative_rows[start : start + batch_size]
            difference = _score_rows(
                model, features, categories, embeddings, positive
            ) - _score_rows(
                model, features, categories, embeddings, negative
            )
            losses.append(float((-F.logsigmoid(difference).mean()).item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_temporal_ranker(
    model: nn.Module,
    features: NDArray[np.float32],
    categories: NDArray[np.int32],
    groups: NDArray[np.int32],
    embeddings: PairEmbeddingStore,
    customer_ids: list[str],
    pair_article_indices: NDArray[np.int32],
    article_ids: list[str],
    relevance: dict[str, set[str]],
    k: int = 12,
) -> dict[str, float | int]:
    scores = score_all(model, features, categories, embeddings)
    return evaluate_scores(
        scores,
        groups,
        customer_ids,
        pair_article_indices,
        article_ids,
        relevance,
        k,
    )


def evaluate_scores(
    scores: NDArray[np.floating],
    groups: NDArray[np.int32],
    customer_ids: list[str],
    pair_article_indices: NDArray[np.int32],
    article_ids: list[str],
    relevance: dict[str, set[str]],
    k: int = 12,
) -> dict[str, float | int]:
    """Evaluate precomputed pair scores without materializing pair ID tuples."""
    aps: list[float] = []
    hits = 0
    customers_with_hit = 0
    offset = 0
    for customer_id, group_size in zip(customer_ids, groups):
        end = offset + int(group_size)
        count = min(k, int(group_size))
        local = np.argpartition(scores[offset:end], -count)[-count:]
        local = local[np.argsort(-scores[offset:end][local])]
        predicted = [
            article_ids[int(pair_article_indices[offset + int(index)])]
            for index in local
        ]
        relevant = relevance.get(customer_id, set())
        customer_hits = len(set(predicted).intersection(relevant))
        hits += customer_hits
        customers_with_hit += int(customer_hits > 0)
        aps.append(average_precision_at_k(relevant, predicted, k))
        offset = end
    total_relevant = sum(len(items) for items in relevance.values())
    customer_count = len(customer_ids)
    return {
        "map_at_12": float(np.mean(aps)) if aps else 0.0,
        "recall_at_12": hits / total_relevant if total_relevant else 0.0,
        "hit_rate_at_12": (
            customers_with_hit / customer_count if customer_count else 0.0
        ),
        "customers": customer_count,
    }


def score_all(
    model: nn.Module,
    features: NDArray[np.float32],
    categories: NDArray[np.int32],
    embeddings: PairEmbeddingStore,
    batch_size: int = 50_000,
) -> NDArray[np.float32]:
    model.eval()
    outputs: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            rows = np.arange(
                start,
                min(start + batch_size, len(features)),
                dtype=np.int64,
            )
            outputs.append(
                _score_rows(
                    model,
                    features,
                    categories,
                    embeddings,
                    rows,
                ).numpy()
            )
    return np.concatenate(outputs).astype(np.float32)


def _score_rows(
    model: nn.Module,
    features: NDArray[np.float32],
    categories: NDArray[np.int32],
    embeddings: PairEmbeddingStore,
    rows: NDArray[np.int64],
) -> torch.Tensor:
    return model(
        torch.from_numpy(features[rows]).float(),
        torch.from_numpy(categories[rows]).long(),
        embeddings.get(rows),
    )
