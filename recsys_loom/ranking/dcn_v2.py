"""DCN V2 ranking model with categorical embeddings and BPR pairwise loss.

Architecture (parallel, following Wang et al. 2020):

    Input = [categorical embeddings ; normalized numerical features]
                        |
           +------------+------------+
           |                         |
      Cross Network               MLP
    (explicit crosses)     (nonlinear patterns)
           |                         |
           +------------+------------+
                        |
                   Linear -> score

Feature normalization:
    - Count features (purchases, frequencies): log1p then standardize
    - Recency features (days_since_*): log1p(max(0, x)) then standardize
    - Already bounded features (affinity, rank_pct, is_*): standardize directly
    - Statistics (mean, std) fitted on training data only
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from recsys_loom.metrics import average_precision_at_k

LOG1P_FEATURES = {
    "item_purchases_7d", "item_purchases_30d", "item_unique_buyers_30d",
    "times_bought_item", "times_bought_product_type", "times_bought_garment_group",
    "days_since_bought_item", "days_since_bought_product_type",
    "days_since_bought_garment_group",
}

SKIP_NORMALIZE = {"num_sources"}


@dataclass
class FeatureNormalizer:
    """Fit on training data, transform both train and validation."""
    mean: NDArray[np.float32] = field(default_factory=lambda: np.array([]))
    std: NDArray[np.float32] = field(default_factory=lambda: np.array([]))
    log1p_mask: NDArray[np.bool_] = field(default_factory=lambda: np.array([]))
    skip_mask: NDArray[np.bool_] = field(default_factory=lambda: np.array([]))

    def fit(self, X: NDArray[np.float32], feature_names: list[str]) -> None:
        self.log1p_mask = np.array([f in LOG1P_FEATURES for f in feature_names])
        self.skip_mask = np.array([f in SKIP_NORMALIZE for f in feature_names])

        X_proc = X.copy()
        X_proc = np.nan_to_num(X_proc, nan=0.0)
        X_proc[:, self.log1p_mask] = np.log1p(np.maximum(X_proc[:, self.log1p_mask], 0))

        self.mean = np.nanmean(X_proc, axis=0).astype(np.float32)
        self.std = np.nanstd(X_proc, axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        self.mean[self.skip_mask] = 0.0
        self.std[self.skip_mask] = 1.0

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        X_proc = X.copy()
        X_proc = np.nan_to_num(X_proc, nan=0.0)
        X_proc[:, self.log1p_mask] = np.log1p(np.maximum(X_proc[:, self.log1p_mask], 0))
        X_proc = (X_proc - self.mean) / self.std
        return X_proc.astype(np.float32)


class CrossLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(dim, dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.weight(xl) + xl


class DCNV2(nn.Module):
    def __init__(
        self,
        num_numerical: int,
        categorical_cardinalities: list[int],
        embedding_dim: int = 8,
        cross_layers: int = 2,
        mlp_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        dense_context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embedding_dim, padding_idx=0)
            for card in categorical_cardinalities
        ])

        total_emb_dim = len(categorical_cardinalities) * embedding_dim
        self.dense_context_dim = dense_context_dim
        input_dim = num_numerical + total_emb_dim + dense_context_dim

        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(cross_layers)
        ])

        mlp_layers: list[nn.Module] = []
        prev = input_dim
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev = dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(input_dim + prev, 1)

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        dense_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb_list = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_concat = torch.cat(emb_list, dim=-1)
        inputs = [numerical, emb_concat]
        if self.dense_context_dim:
            if dense_context is None:
                raise ValueError("dense_context is required for this model")
            inputs.append(dense_context)
        x = torch.cat(inputs, dim=-1)

        cross_out = x
        for layer in self.cross_layers:
            cross_out = layer(x, cross_out)

        mlp_out = self.mlp(x)
        combined = torch.cat([cross_out, mlp_out], dim=-1)
        return self.output_layer(combined).squeeze(-1)


class SimpleMLPRanker(nn.Module):
    """MLP-only baseline for ablation against DCN."""
    def __init__(
        self,
        num_numerical: int,
        categorical_cardinalities: list[int],
        embedding_dim: int = 8,
        mlp_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        dense_context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embedding_dim, padding_idx=0)
            for card in categorical_cardinalities
        ])

        total_emb_dim = len(categorical_cardinalities) * embedding_dim
        self.dense_context_dim = dense_context_dim
        input_dim = num_numerical + total_emb_dim + dense_context_dim

        layers: list[nn.Module] = []
        prev = input_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        dense_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb_list = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_concat = torch.cat(emb_list, dim=-1)
        inputs = [numerical, emb_concat]
        if self.dense_context_dim:
            if dense_context is None:
                raise ValueError("dense_context is required for this model")
            inputs.append(dense_context)
        x = torch.cat(inputs, dim=-1)
        return self.net(x).squeeze(-1)


def build_bpr_pairs(
    pairs: Sequence[tuple[str, str]],
    labels: NDArray[np.int32],
    X_num: NDArray[np.float32],
    source_score_indices: dict[str, int] | None = None,
    neg_per_pos: int = 10,
    hard_neg_fraction: float = 0.5,
    rng_seed: int = 42,
) -> list[tuple[int, int]]:
    """Build (positive_idx, negative_idx) pairs with mixed random + hard negatives.

    Hard negatives: highest raw source scores among unpurchased candidates
    for the same customer (items the retrieval thought were good but weren't bought).
    """
    rng = np.random.default_rng(rng_seed)

    customer_positives: dict[str, list[int]] = {}
    customer_negatives: dict[str, list[int]] = {}

    for i, (cid, _) in enumerate(pairs):
        if labels[i] == 1:
            customer_positives.setdefault(cid, []).append(i)
        else:
            customer_negatives.setdefault(cid, []).append(i)

    n_hard = max(1, int(neg_per_pos * hard_neg_fraction))
    n_random = neg_per_pos - n_hard

    bpr_pairs: list[tuple[int, int]] = []
    for cid, pos_indices in customer_positives.items():
        neg_indices = customer_negatives.get(cid, [])
        if not neg_indices:
            continue

        neg_arr = np.array(neg_indices)

        hard_indices = neg_arr
        if source_score_indices and len(neg_arr) > n_hard:
            score_cols = list(source_score_indices.values())
            neg_scores = np.nanmax(
                np.nan_to_num(X_num[neg_arr][:, score_cols], nan=-999),
                axis=1,
            )
            top_hard = np.argsort(-neg_scores)[:n_hard * 3]
            hard_pool = neg_arr[top_hard]
        else:
            hard_pool = neg_arr

        for pos_idx in pos_indices:
            sampled_hard = rng.choice(
                hard_pool, size=min(n_hard, len(hard_pool)), replace=False,
            )
            sampled_random = rng.choice(
                neg_arr, size=min(n_random, len(neg_arr)), replace=False,
            )
            for neg_idx in sampled_hard:
                bpr_pairs.append((pos_idx, int(neg_idx)))
            for neg_idx in sampled_random:
                bpr_pairs.append((pos_idx, int(neg_idx)))

    rng.shuffle(bpr_pairs)
    return bpr_pairs


def get_source_score_indices(feature_names: list[str]) -> dict[str, int]:
    """Find column indices for source score features (for hard negative mining)."""
    result: dict[str, int] = {}
    for i, name in enumerate(feature_names):
        if name.startswith("score_"):
            result[name] = i
    return result


@dataclass
class TrainingLog:
    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_maps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_maps": self.val_maps,
        }


def train_neural_ranker(
    model: nn.Module,
    X_num: NDArray[np.float32],
    X_cat: NDArray[np.int64],
    labels: NDArray[np.int32],
    pairs: Sequence[tuple[str, str]],
    feature_names: list[str],
    X_val_num: NDArray[np.float32] | None = None,
    X_val_cat: NDArray[np.int64] | None = None,
    val_pairs: Sequence[tuple[str, str]] | None = None,
    val_relevant: dict[str, set[str]] | None = None,
    epochs: int = 40,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    neg_per_pos: int = 20,
    hard_neg_fraction: float = 0.5,
    patience: int = 6,
    verbose: bool = True,
) -> tuple[nn.Module, TrainingLog]:
    """Train a neural ranker (DCN or MLP) with BPR loss and early stopping."""
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_num_t = torch.from_numpy(X_num).float().to(device)
    X_cat_t = torch.from_numpy(X_cat).long().to(device)

    source_score_idx = get_source_score_indices(feature_names)

    has_validation = (
        X_val_num is not None and X_val_cat is not None
        and val_pairs is not None and val_relevant is not None
    )

    log = TrainingLog()
    best_map = -1.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()

        bpr_pairs_list = build_bpr_pairs(
            pairs, labels, X_num, source_score_idx,
            neg_per_pos, hard_neg_fraction, rng_seed=42 + epoch,
        )

        if not bpr_pairs_list:
            continue

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(bpr_pairs_list), batch_size):
            batch = bpr_pairs_list[start:start + batch_size]
            pos_idx = [p[0] for p in batch]
            neg_idx = [p[1] for p in batch]

            pos_scores = model(X_num_t[pos_idx], X_cat_t[pos_idx])
            neg_scores = model(X_num_t[neg_idx], X_cat_t[neg_idx])

            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        log.epochs.append(epoch + 1)
        log.train_losses.append(avg_loss)

        val_map = 0.0
        if has_validation:
            val_map = _evaluate_map(model, X_val_num, X_val_cat, val_pairs, val_relevant, device)
            log.val_maps.append(val_map)

            if verbose:
                print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}  val_MAP@12={val_map:.4f}",
                      flush=True)

            if val_map > best_map:
                best_map = val_map
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"    Early stop at epoch {epoch+1}, best MAP@12={best_map:.4f}",
                              flush=True)
                    break
        else:
            log.val_maps.append(0.0)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, log


def overfit_sanity_test(
    model: nn.Module,
    X_num: NDArray[np.float32],
    X_cat: NDArray[np.int64],
    labels: NDArray[np.int32],
    pairs: Sequence[tuple[str, str]],
    feature_names: list[str],
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> bool:
    """Verify the model can memorize a tiny dataset. Returns True if it can."""
    test_model = copy.deepcopy(model)
    device = torch.device("cpu")
    test_model = test_model.to(device)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=lr)

    X_num_t = torch.from_numpy(X_num).float().to(device)
    X_cat_t = torch.from_numpy(X_cat).long().to(device)

    source_score_idx = get_source_score_indices(feature_names)

    print("    Overfit sanity test:", flush=True)
    for epoch in range(n_epochs):
        test_model.train()
        bpr_list = build_bpr_pairs(
            pairs, labels, X_num, source_score_idx,
            neg_per_pos=5, hard_neg_fraction=0.0, rng_seed=epoch,
        )
        if not bpr_list:
            print("      No BPR pairs found!", flush=True)
            return False

        pos_idx = [p[0] for p in bpr_list]
        neg_idx = [p[1] for p in bpr_list]
        pos_scores = test_model(X_num_t[pos_idx], X_cat_t[pos_idx])
        neg_scores = test_model(X_num_t[neg_idx], X_cat_t[neg_idx])
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = ((pos_scores > neg_scores).float().mean().item())

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: loss={loss.item():.4f}  "
                  f"pair_accuracy={accuracy:.3f}", flush=True)

    final_acc = accuracy
    passed = final_acc > 0.9
    print(f"    Result: pair_accuracy={final_acc:.3f} -> {'PASS' if passed else 'FAIL'}",
          flush=True)
    return passed


def _evaluate_map(
    model: nn.Module,
    X_num: NDArray[np.float32],
    X_cat: NDArray[np.int64],
    pairs: Sequence[tuple[str, str]],
    relevant: dict[str, set[str]],
    device: torch.device,
    k: int = 12,
) -> float:
    model.eval()
    X_num_t = torch.from_numpy(X_num).float().to(device)
    X_cat_t = torch.from_numpy(X_cat).long().to(device)

    with torch.no_grad():
        scores_list = []
        for start in range(0, len(pairs), 50000):
            end = min(start + 50000, len(pairs))
            s = model(X_num_t[start:end], X_cat_t[start:end])
            scores_list.append(s.cpu().numpy())
        scores = np.concatenate(scores_list)

    by_customer: dict[str, list[tuple[float, str]]] = {}
    for i, (cid, aid) in enumerate(pairs):
        by_customer.setdefault(cid, []).append((scores[i], aid))

    aps: list[float] = []
    for cid, rel in relevant.items():
        items = by_customer.get(cid, [])
        items.sort(key=lambda x: -x[0])
        predicted = [aid for _, aid in items[:k]]
        aps.append(average_precision_at_k(rel, predicted, k))

    return sum(aps) / len(aps) if aps else 0.0


def predict_neural(
    model: nn.Module,
    X_num: NDArray[np.float32],
    X_cat: NDArray[np.int64],
    pairs: Sequence[tuple[str, str]],
    top_k: int = 12,
) -> dict[str, list[str]]:
    device = next(model.parameters()).device
    model.eval()

    X_num_t = torch.from_numpy(X_num).float().to(device)
    X_cat_t = torch.from_numpy(X_cat).long().to(device)

    with torch.no_grad():
        scores_list = []
        for start in range(0, len(pairs), 50000):
            end = min(start + 50000, len(pairs))
            s = model(X_num_t[start:end], X_cat_t[start:end])
            scores_list.append(s.cpu().numpy())
        scores = np.concatenate(scores_list)

    by_customer: dict[str, list[tuple[float, str]]] = {}
    for i, (cid, aid) in enumerate(pairs):
        by_customer.setdefault(cid, []).append((scores[i], aid))

    result: dict[str, list[str]] = {}
    for cid, items in by_customer.items():
        items.sort(key=lambda x: -x[0])
        result[cid] = [aid for _, aid in items[:top_k]]
    return result
