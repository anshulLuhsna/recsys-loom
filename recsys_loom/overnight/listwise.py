"""Compact set-aware reranker used on a LambdaRank shortlist."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from numpy.typing import NDArray


class SetReranker(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=2)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(torch.relu(self.input(features)))
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        return self.output(hidden).squeeze(-1)


def listnet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scores = scores.masked_fill(~mask, -1e9)
    labels = labels.masked_fill(~mask, 0.0)
    label_mass = labels.sum(dim=1, keepdim=True).clamp(min=1.0)
    target = labels / label_mass
    log_prob = torch.log_softmax(scores, dim=1)
    return -(target * log_prob * mask).sum() / mask.any(dim=1).sum().clamp(min=1)


def shortlist_groups(
    scores: NDArray[np.floating],
    data: dict[str, NDArray],
    shortlist: int,
) -> list[dict[str, NDArray]]:
    groups: list[dict[str, NDArray]] = []
    offset = 0
    for group_size_value in data["groups"]:
        group_size = int(group_size_value)
        end = offset + group_size
        take = min(shortlist, group_size)
        local = np.argpartition(scores[offset:end], -take)[-take:]
        local = local[np.argsort(-scores[offset:end][local])]
        groups.append(
            {
                "features": data["features"][offset:end][local],
                "labels": data["labels"][offset:end][local],
                "article_indices": data["pair_article_indices"][offset:end][local],
            }
        )
        offset = end
    return groups


def _stack_batch(
    groups: list[dict[str, NDArray]],
    start: int,
    end: int,
    feature_mean: NDArray[np.floating],
    feature_std: NDArray[np.floating],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(group["labels"]) for group in groups[start:end])
    feature_dim = groups[0]["features"].shape[1]
    features = np.zeros((end - start, width, feature_dim), dtype=np.float32)
    labels = np.zeros((end - start, width), dtype=np.float32)
    mask = np.zeros((end - start, width), dtype=np.bool_)
    for row, group in enumerate(groups[start:end]):
        count = len(group["labels"])
        normalized = (
            (group["features"] - feature_mean) / np.maximum(feature_std, 1e-6)
        ).astype(np.float32)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        features[row, :count] = normalized
        labels[row, :count] = group["labels"]
        mask[row, :count] = True
    return (
        torch.from_numpy(features),
        torch.from_numpy(labels),
        torch.from_numpy(mask),
    )


def fit_reranker(
    groups: list[dict[str, NDArray]],
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> tuple[SetReranker, dict[str, list[float]]]:
    feature_dim = int(groups[0]["features"].shape[1])
    stacked = np.concatenate([group["features"] for group in groups], axis=0)
    feature_mean = np.nanmean(stacked, axis=0)
    feature_std = np.nanstd(stacked, axis=0)
    model = SetReranker(feature_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = {"loss": []}
    model.train()
    for _epoch in range(epochs):
        order = np.random.permutation(len(groups))
        epoch_loss = []
        for start in range(0, len(groups), batch_size):
            batch_index = order[start : start + batch_size]
            batch_groups = [groups[int(index)] for index in batch_index]
            features, labels, mask = _stack_batch(
                batch_groups,
                0,
                len(batch_groups),
                feature_mean,
                feature_std,
            )
            optimizer.zero_grad()
            scores = model(features, mask)
            loss = listnet_loss(scores, labels, mask)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.item()))
        history["loss"].append(float(np.mean(epoch_loss)))
    model.feature_mean = feature_mean
    model.feature_std = feature_std
    return model, history


def predict_shortlist(
    model: SetReranker,
    groups: list[dict[str, NDArray]],
    article_ids: list[str],
    top_k: int = 12,
) -> dict[int, list[str]]:
    model.eval()
    predictions: dict[int, list[str]] = {}
    with torch.no_grad():
        for customer_index, group in enumerate(groups):
            features, _labels, mask = _stack_batch(
                [group],
                0,
                1,
                model.feature_mean,
                model.feature_std,
            )
            scores = model(features, mask)[0].cpu().numpy()
            valid = int(mask[0].sum().item())
            order = np.argsort(-scores[:valid])
            predicted = [
                article_ids[int(group["article_indices"][int(index)])]
                for index in order[:top_k]
            ]
            predictions[customer_index] = predicted
    return predictions


def tiny_overfit_sanity() -> dict[str, float]:
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(4):
        features = rng.normal(size=(8, 6)).astype(np.float32)
        labels = np.zeros(8, dtype=np.int8)
        labels[:2] = 1
        features[:2] += 2.0
        groups.append(
            {
                "features": features,
                "labels": labels,
                "article_indices": np.arange(8),
            }
        )
    model, history = fit_reranker(groups, epochs=40, batch_size=4, learning_rate=5e-3)
    return {
        "start_loss": history["loss"][0],
        "end_loss": history["loss"][-1],
        "overfit_ok": float(history["loss"][-1] < history["loss"][0]),
    }
