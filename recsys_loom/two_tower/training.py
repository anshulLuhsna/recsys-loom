"""Sampled-softmax training and diagnostics for two-tower retrieval."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from recsys_loom.two_tower.data import (
    Catalog,
    SnapshotData,
    UserNumericalNormalizer,
)
from recsys_loom.two_tower.model import TwoTowerModel


@dataclass(slots=True)
class TrainingLog:
    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    selected_epoch: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "validation_losses": self.validation_losses,
            "learning_rates": self.learning_rates,
            "selected_epoch": self.selected_epoch,
        }


@dataclass(frozen=True, slots=True)
class SimilarityDiagnostic:
    positive_mean: float
    random_negative_mean: float
    margin: float

    def to_dict(self) -> dict[str, float]:
        return {
            "positive_mean": self.positive_mean,
            "random_negative_mean": self.random_negative_mean,
            "margin": self.margin,
        }


def train_two_tower(
    model: TwoTowerModel,
    train_snapshots: list[SnapshotData],
    validation_snapshot: SnapshotData,
    catalog: Catalog,
    normalizer: UserNumericalNormalizer,
    text_embeddings: NDArray[np.float32] | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    random_negatives: int = 256,
    temperature: float = 0.07,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    seed: int = 42,
    verbose: bool = True,
    device_name: str = "cpu",
) -> tuple[TwoTowerModel, TrainingLog]:
    """Train with in-batch and sampled unpurchased negatives."""
    device = torch.device(device_name)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=learning_rate * 0.1,
    )
    rng = np.random.default_rng(seed)

    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    log = TrainingLog()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        snapshot_order = rng.permutation(len(train_snapshots))
        for snapshot_index in snapshot_order:
            snapshot = train_snapshots[int(snapshot_index)]
            losses.extend(
                _run_snapshot_batches(
                    model=model,
                    snapshot=snapshot,
                    catalog=catalog,
                    normalizer=normalizer,
                    batch_size=batch_size,
                    random_negatives=random_negatives,
                    temperature=temperature,
                    rng=rng,
                    optimizer=optimizer,
                    device=device,
                    text_embeddings=text_embeddings,
                )
            )

        validation_loss = evaluate_sampled_softmax_loss(
            model=model,
            snapshot=validation_snapshot,
            catalog=catalog,
            normalizer=normalizer,
            batch_size=batch_size,
            random_negatives=random_negatives,
            temperature=temperature,
            seed=seed + 10_000,
            text_embeddings=text_embeddings,
            device=device,
        )
        train_loss = float(np.mean(losses)) if losses else math.inf
        current_lr = float(optimizer.param_groups[0]["lr"])

        log.epochs.append(epoch)
        log.train_losses.append(train_loss)
        log.validation_losses.append(validation_loss)
        log.learning_rates.append(current_lr)

        if verbose:
            print(
                f"    epoch {epoch:02d}: train_loss={train_loss:.4f} "
                f"temporal_val_loss={validation_loss:.4f}",
                flush=True,
            )

        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            log.selected_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log


def fit_for_fixed_epochs(
    model: TwoTowerModel,
    snapshots: list[SnapshotData],
    catalog: Catalog,
    normalizer: UserNumericalNormalizer,
    epochs: int,
    text_embeddings: NDArray[np.float32] | None = None,
    batch_size: int = 256,
    random_negatives: int = 256,
    temperature: float = 0.07,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device_name: str = "cpu",
) -> tuple[TwoTowerModel, list[float]]:
    """Refit on all training snapshots for the selected number of epochs."""
    device = torch.device(device_name)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for snapshot_index in rng.permutation(len(snapshots)):
            losses.extend(
                _run_snapshot_batches(
                    model=model,
                    snapshot=snapshots[int(snapshot_index)],
                    catalog=catalog,
                    normalizer=normalizer,
                    batch_size=batch_size,
                    random_negatives=random_negatives,
                    temperature=temperature,
                    rng=rng,
                    optimizer=optimizer,
                    device=device,
                    text_embeddings=text_embeddings,
                )
            )
        mean_loss = float(np.mean(losses)) if losses else math.inf
        epoch_losses.append(mean_loss)
        print(f"    refit epoch {epoch:02d}: loss={mean_loss:.4f}", flush=True)

    return model, epoch_losses


def evaluate_sampled_softmax_loss(
    model: TwoTowerModel,
    snapshot: SnapshotData,
    catalog: Catalog,
    normalizer: UserNumericalNormalizer,
    batch_size: int,
    random_negatives: int,
    temperature: float,
    seed: int,
    text_embeddings: NDArray[np.float32] | None = None,
    device: torch.device | None = None,
) -> float:
    model.eval()
    evaluation_device = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        losses = _run_snapshot_batches(
            model=model,
            snapshot=snapshot,
            catalog=catalog,
            normalizer=normalizer,
            batch_size=batch_size,
            random_negatives=random_negatives,
            temperature=temperature,
            rng=rng,
            optimizer=None,
            device=evaluation_device,
            shuffle=False,
            text_embeddings=text_embeddings,
        )
    return float(np.mean(losses)) if losses else math.inf


def tiny_overfit_check(
    model: TwoTowerModel,
    snapshot: SnapshotData,
    catalog: Catalog,
    normalizer: UserNumericalNormalizer,
    text_embeddings: NDArray[np.float32] | None = None,
    max_examples: int = 96,
    epochs: int = 80,
    seed: int = 7,
    device_name: str = "cpu",
) -> dict[str, float | bool]:
    """Ensure feature-built towers can memorize a tiny set of future pairs."""
    examples = [
        (profile_index, int(items[0]))
        for profile_index, items in enumerate(snapshot.positive_item_indices)
        if len(items) > 0
    ]
    if not examples:
        return {"passed": False, "final_loss": math.inf, "top1_accuracy": 0.0}
    examples = examples[:max_examples]

    device = torch.device(device_name)
    test_model = copy.deepcopy(model).to(device)
    test_model.train()
    optimizer = torch.optim.Adam(test_model.parameters(), lr=5e-3)
    profile_indices = np.asarray([row[0] for row in examples], dtype=np.int64)
    item_indices = np.asarray([row[1] for row in examples], dtype=np.int64)
    final_loss = math.inf
    top1_accuracy = 0.0

    for _ in range(epochs):
        users, positives = _encode_example_batch(
            test_model,
            snapshot,
            catalog,
            normalizer,
            profile_indices,
            item_indices,
            device,
            text_embeddings,
        )
        logits = users @ positives.T / 0.07
        targets = torch.arange(len(examples), device=device)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        top1_accuracy = float(
            (logits.argmax(dim=1) == targets).float().mean().item()
        )

    passed = top1_accuracy >= 0.85 and final_loss < 0.5
    return {
        "passed": passed,
        "final_loss": final_loss,
        "top1_accuracy": top1_accuracy,
        "examples": len(examples),
    }


def similarity_diagnostic(
    model: TwoTowerModel,
    snapshot: SnapshotData,
    catalog: Catalog,
    normalizer: UserNumericalNormalizer,
    text_embeddings: NDArray[np.float32] | None = None,
    max_examples: int = 1000,
    seed: int = 42,
    device_name: str = "cpu",
) -> SimilarityDiagnostic:
    """Compare held-out positive similarity with random eligible negatives."""
    examples = _flatten_examples(snapshot)[:max_examples]
    if not examples:
        return SimilarityDiagnostic(0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    profiles = np.asarray([row[0] for row in examples], dtype=np.int64)
    positives = np.asarray([row[1] for row in examples], dtype=np.int64)
    negatives = rng.choice(
        snapshot.eligible_item_indices,
        size=len(examples),
        replace=True,
    ).astype(np.int64)

    model.eval()
    device = torch.device(device_name)
    model = model.to(device)
    with torch.no_grad():
        users, positive_items = _encode_example_batch(
            model,
            snapshot,
            catalog,
            normalizer,
            profiles,
            positives,
            device,
            text_embeddings,
        )
        negative_features = torch.from_numpy(
            catalog.article_features[negatives]
        ).long().to(device)
        negative_items = model.encode_items(
            negative_features,
            _item_text_tensor(text_embeddings, negatives, device),
        )
        positive_similarity = (users * positive_items).sum(dim=1)
        negative_similarity = (users * negative_items).sum(dim=1)

    positive_mean = float(positive_similarity.mean().item())
    negative_mean = float(negative_similarity.mean().item())
    return SimilarityDiagnostic(
        positive_mean=positive_mean,
        random_negative_mean=negative_mean,
        margin=positive_mean - negative_mean,
    )


def encode_all_items(
    model: TwoTowerModel,
    catalog: Catalog,
    text_embeddings: NDArray[np.float32] | None = None,
    batch_size: int = 4096,
) -> NDArray[np.float32]:
    model.eval()
    device = next(model.parameters()).device
    result: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(catalog.article_ids), batch_size):
            features = torch.from_numpy(
                catalog.article_features[start : start + batch_size]
            ).long().to(device)
            item_indices = np.arange(
                start,
                min(start + batch_size, len(catalog.article_ids)),
                dtype=np.int64,
            )
            result.append(
                model.encode_items(
                    features,
                    _item_text_tensor(
                        text_embeddings,
                        item_indices,
                        device,
                    ),
                ).cpu().numpy()
            )
    return np.concatenate(result).astype(np.float32)


def encode_snapshot_users(
    model: TwoTowerModel,
    snapshot: SnapshotData,
    normalizer: UserNumericalNormalizer,
    text_embeddings: NDArray[np.float32] | None = None,
    batch_size: int = 1024,
) -> NDArray[np.float32]:
    model.eval()
    device = next(model.parameters()).device
    normalized_numerical = normalizer.transform(snapshot.user_numerical)
    result: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(snapshot.customer_ids), batch_size):
            end = start + batch_size
            result.append(
                model.encode_users(
                    torch.from_numpy(snapshot.history_features[start:end]).long().to(device),
                    torch.from_numpy(snapshot.history_weights[start:end]).float().to(device),
                    torch.from_numpy(normalized_numerical[start:end]).float().to(device),
                    torch.from_numpy(snapshot.user_categorical[start:end]).long().to(device),
                    _history_text_tensor(
                        text_embeddings,
                        snapshot.history_article_indices[start:end],
                        device,
                    ),
                ).cpu().numpy()
            )
    return np.concatenate(result).astype(np.float32)


def _run_snapshot_batches(
    model: TwoTowerModel,
    snapshot: SnapshotData,
    catalog: Catalog,
    normalizer: UserNumericalNormalizer,
    batch_size: int,
    random_negatives: int,
    temperature: float,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    shuffle: bool = True,
    text_embeddings: NDArray[np.float32] | None = None,
) -> list[float]:
    examples = _flatten_examples(snapshot)
    if shuffle:
        rng.shuffle(examples)
    losses: list[float] = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        if len(batch) < 2:
            continue
        profiles = np.asarray([row[0] for row in batch], dtype=np.int64)
        positives = np.asarray([row[1] for row in batch], dtype=np.int64)
        users, positive_items = _encode_example_batch(
            model,
            snapshot,
            catalog,
            normalizer,
            profiles,
            positives,
            device,
            text_embeddings,
        )

        sampled = rng.choice(
            snapshot.eligible_item_indices,
            size=min(random_negatives, len(snapshot.eligible_item_indices)),
            replace=False,
        ).astype(np.int64)
        sampled_features = torch.from_numpy(
            catalog.article_features[sampled]
        ).long().to(device)
        sampled_items = model.encode_items(
            sampled_features,
            _item_text_tensor(text_embeddings, sampled, device),
        )

        in_batch_logits = users @ positive_items.T / temperature
        sampled_logits = users @ sampled_items.T / temperature
        _mask_known_items(
            in_batch_logits,
            sampled_logits,
            snapshot,
            profiles,
            positives,
            sampled,
        )
        logits = torch.cat([in_batch_logits, sampled_logits], dim=1)
        targets = torch.arange(len(batch), device=device)
        loss = F.cross_entropy(logits, targets)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        losses.append(float(loss.item()))

    return losses


def _encode_example_batch(
    model: TwoTowerModel,
    snapshot: SnapshotData,
    catalog: Catalog,
    normalizer: UserNumericalNormalizer,
    profile_indices: NDArray[np.int64],
    positive_indices: NDArray[np.int64],
    device: torch.device,
    text_embeddings: NDArray[np.float32] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerical = normalizer.transform(snapshot.user_numerical[profile_indices])
    return model(
        torch.from_numpy(snapshot.history_features[profile_indices]).long().to(device),
        torch.from_numpy(snapshot.history_weights[profile_indices]).float().to(device),
        torch.from_numpy(numerical).float().to(device),
        torch.from_numpy(snapshot.user_categorical[profile_indices]).long().to(device),
        torch.from_numpy(catalog.article_features[positive_indices]).long().to(device),
        _history_text_tensor(
            text_embeddings,
            snapshot.history_article_indices[profile_indices],
            device,
        ),
        _item_text_tensor(text_embeddings, positive_indices, device),
    )


def _item_text_tensor(
    text_embeddings: NDArray[np.float32] | None,
    item_indices: NDArray[np.int64],
    device: torch.device,
) -> torch.Tensor | None:
    if text_embeddings is None:
        return None
    values = np.asarray(text_embeddings[item_indices], dtype=np.float32)
    return torch.from_numpy(values).float().to(device)


def _history_text_tensor(
    text_embeddings: NDArray[np.float32] | None,
    history_item_indices: NDArray[np.int64],
    device: torch.device,
) -> torch.Tensor | None:
    if text_embeddings is None:
        return None
    valid = history_item_indices >= 0
    safe_indices = np.maximum(history_item_indices, 0)
    values = np.asarray(
        text_embeddings[safe_indices],
        dtype=np.float32,
    ).copy()
    values[~valid] = 0.0
    return torch.from_numpy(values).float().to(device)


def _flatten_examples(snapshot: SnapshotData) -> list[tuple[int, int]]:
    return [
        (profile_index, int(item_index))
        for profile_index, items in enumerate(snapshot.positive_item_indices)
        for item_index in items
    ]


def _mask_known_items(
    in_batch_logits: torch.Tensor,
    sampled_logits: torch.Tensor,
    snapshot: SnapshotData,
    profile_indices: NDArray[np.int64],
    positive_indices: NDArray[np.int64],
    sampled_indices: NDArray[np.int64],
) -> None:
    for row_index, profile_index in enumerate(profile_indices):
        known = snapshot.history_item_indices[int(profile_index)].union(
            int(item)
            for item in snapshot.positive_item_indices[int(profile_index)]
        )
        for column_index, item_index in enumerate(positive_indices):
            if column_index != row_index and int(item_index) in known:
                in_batch_logits[row_index, column_index] = -1e9
        for column_index, item_index in enumerate(sampled_indices):
            if int(item_index) in known:
                sampled_logits[row_index, column_index] = -1e9
