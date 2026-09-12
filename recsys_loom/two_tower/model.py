"""Feature-built user and item towers for dot-product retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalEncoder(nn.Module):
    """Embed multiple categorical fields and concatenate their representations."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality + 1, embedding_dim, padding_idx=0)
            for cardinality in cardinalities
        )
        self.output_dim = len(cardinalities) * embedding_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = [
            embedding(values[..., field_index])
            for field_index, embedding in enumerate(self.embeddings)
        ]
        return torch.cat(encoded, dim=-1)


class ItemTower(nn.Module):
    """Construct article embeddings entirely from catalog metadata."""

    def __init__(
        self,
        article_cardinalities: Sequence[int],
        categorical_embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        text_embedding_dim: int = 0,
        text_projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.categorical = CategoricalEncoder(
            article_cardinalities,
            categorical_embedding_dim,
        )
        self.text_projection: nn.Module | None = None
        input_dim = self.categorical.output_dim
        if text_embedding_dim > 0:
            self.text_projection = nn.Sequential(
                nn.Linear(text_embedding_dim, text_projection_dim),
                nn.LayerNorm(text_projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            input_dim += text_projection_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        article_features: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = [self.categorical(article_features)]
        if self.text_projection is not None:
            if text_embeddings is None:
                raise ValueError("text_embeddings are required by this item tower")
            inputs.append(self.text_projection(text_embeddings))
        return F.normalize(
            self.projection(torch.cat(inputs, dim=-1)),
            dim=-1,
        )


class UserTower(nn.Module):
    """Construct users from recency-weighted history plus customer features."""

    def __init__(
        self,
        item_tower: ItemTower,
        customer_cardinalities: Sequence[int],
        customer_embedding_dim: int,
        numerical_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.item_tower = item_tower
        self.customer_categorical = CategoricalEncoder(
            customer_cardinalities,
            customer_embedding_dim,
        )
        input_dim = (
            output_dim
            + self.customer_categorical.output_dim
            + numerical_dim
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_weights: torch.Tensor,
        user_numerical: torch.Tensor,
        user_categorical: torch.Tensor,
        history_text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, history_length, field_count = history_features.shape
        history_items = self.item_tower(
            history_features.reshape(batch_size * history_length, field_count),
            (
                history_text_embeddings.reshape(
                    batch_size * history_length,
                    -1,
                )
                if history_text_embeddings is not None
                else None
            ),
        ).reshape(batch_size, history_length, -1)

        weighted_history = history_items * history_weights.unsqueeze(-1)
        weight_sum = history_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        history_embedding = weighted_history.sum(dim=1) / weight_sum
        has_history = (history_weights.sum(dim=1, keepdim=True) > 0).float()
        history_embedding = history_embedding * has_history

        customer_embedding = self.customer_categorical(user_categorical)
        inputs = torch.cat(
            [history_embedding, customer_embedding, user_numerical],
            dim=-1,
        )
        return F.normalize(self.projection(inputs), dim=-1)


class TwoTowerModel(nn.Module):
    """Two feature-built towers sharing the item encoder for user histories."""

    def __init__(
        self,
        article_cardinalities: Sequence[int],
        customer_cardinalities: Sequence[int],
        numerical_dim: int,
        embedding_dim: int = 64,
        categorical_embedding_dim: int = 8,
        customer_embedding_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        text_embedding_dim: int = 0,
        text_projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.item_tower = ItemTower(
            article_cardinalities=article_cardinalities,
            categorical_embedding_dim=categorical_embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
            text_embedding_dim=text_embedding_dim,
            text_projection_dim=text_projection_dim,
        )
        self.user_tower = UserTower(
            item_tower=self.item_tower,
            customer_cardinalities=customer_cardinalities,
            customer_embedding_dim=customer_embedding_dim,
            numerical_dim=numerical_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )

    def encode_items(
        self,
        article_features: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.item_tower(article_features, text_embeddings)

    def encode_users(
        self,
        history_features: torch.Tensor,
        history_weights: torch.Tensor,
        user_numerical: torch.Tensor,
        user_categorical: torch.Tensor,
        history_text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.user_tower(
            history_features,
            history_weights,
            user_numerical,
            user_categorical,
            history_text_embeddings,
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_weights: torch.Tensor,
        user_numerical: torch.Tensor,
        user_categorical: torch.Tensor,
        positive_article_features: torch.Tensor,
        history_text_embeddings: torch.Tensor | None = None,
        positive_text_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        users = self.encode_users(
            history_features,
            history_weights,
            user_numerical,
            user_categorical,
            history_text_embeddings,
        )
        items = self.encode_items(
            positive_article_features,
            positive_text_embeddings,
        )
        return users, items

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
