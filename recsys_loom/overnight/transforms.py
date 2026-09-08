"""Training-set transforms shared by experiments and BEST_SYSTEM serving."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

NEGATIVE_SCHEMES = {
    "all_candidates": None,
    "hard50_rand50": (50, 50),
    "hard100_rand50": (100, 50),
    "hard50_rand150": (50, 150),
}


def add_crosses(data: dict[str, NDArray]) -> dict[str, NDArray]:
    names = [str(value) for value in data["feature_names"]]
    index = {name: position for position, name in enumerate(names)}
    features = data["features"]
    als = np.nan_to_num(features[:, index["score_als"]], nan=0.0)
    two_tower = np.nan_to_num(features[:, index["score_two_tower"]], nan=0.0)
    sources = features[:, index["num_sources"]]
    popularity = np.nan_to_num(features[:, index["item_purchases_30d"]], nan=0.0)
    repeat = np.nan_to_num(features[:, index["score_repeat_purchase"]], nan=0.0)
    recency = np.nan_to_num(features[:, index["days_since_bought_item"]], nan=365.0)
    als_rank = np.nan_to_num(features[:, index["rank_als"]], nan=500.0)
    two_tower_rank = np.nan_to_num(features[:, index["rank_two_tower"]], nan=500.0)
    extras = np.column_stack(
        [
            als * two_tower,
            two_tower * sources,
            sources * np.log1p(popularity),
            repeat / (1.0 + np.maximum(recency, 0.0)),
            als_rank - two_tower_rank,
            np.minimum(als_rank, two_tower_rank),
        ]
    ).astype(np.float32)
    extra_names = np.asarray(
        [
            "cross_als_tt",
            "cross_tt_sources",
            "cross_sources_logpop",
            "cross_repeat_recency",
            "als_rank_minus_tt_rank",
            "min_als_tt_rank",
        ]
    )
    updated = dict(data)
    updated["features"] = np.concatenate([features, extras], axis=1)
    updated["feature_names"] = np.concatenate([data["feature_names"], extra_names])
    return updated


def downsample_group(
    features: np.ndarray,
    labels: np.ndarray,
    feature_index: dict[str, int],
    hard: int,
    random: int,
    rng: np.random.Generator,
) -> np.ndarray:
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels == 0)
    if len(negative) == 0:
        return np.ones(len(labels), dtype=np.bool_)
    hardness = (
        features[negative, feature_index["num_sources"]]
        + np.nan_to_num(features[negative, feature_index["score_als"]], nan=0.0)
        + np.nan_to_num(
            features[negative, feature_index["score_two_tower"]], nan=0.0
        )
    )
    hard_take = min(hard, len(negative))
    hard_local = np.argpartition(-hardness, hard_take - 1)[:hard_take]
    remaining = np.setdiff1d(np.arange(len(negative)), hard_local, assume_unique=False)
    random_take = min(random, len(remaining))
    random_local = (
        rng.choice(remaining, size=random_take, replace=False)
        if random_take
        else np.asarray([], dtype=np.int64)
    )
    keep = np.zeros(len(labels), dtype=np.bool_)
    keep[positive] = True
    keep[negative[hard_local]] = True
    keep[negative[random_local]] = True
    return keep


def downsample_snapshot(
    data: dict[str, np.ndarray],
    hard: int,
    random: int,
    seed: int,
) -> dict[str, np.ndarray]:
    feature_index = {
        str(name): index for index, name in enumerate(data["feature_names"])
    }
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(data["labels"]), dtype=np.bool_)
    group_sizes: list[int] = []
    offset = 0
    for group_size_value in data["groups"]:
        group_size = int(group_size_value)
        end = offset + group_size
        group_keep = downsample_group(
            data["features"][offset:end],
            data["labels"][offset:end],
            feature_index,
            hard,
            random,
            rng,
        )
        keep[offset:end] = group_keep
        group_sizes.append(int(group_keep.sum()))
        offset = end
    return {
        "features": data["features"][keep],
        "labels": data["labels"][keep],
        "groups": np.asarray(group_sizes, dtype=np.int32),
        "pair_article_indices": data["pair_article_indices"][keep],
        "customer_ids": data["customer_ids"],
        "history_buckets": data["history_buckets"],
        "feature_names": data["feature_names"],
    }


def weighted_labels(data: dict[str, np.ndarray]) -> np.ndarray:
    names = [str(value) for value in data["feature_names"]]
    popularity = np.nan_to_num(
        data["features"][:, names.index("item_purchases_30d")],
        nan=0.0,
    )
    weights = np.ones(len(data["labels"]), dtype=np.float32)
    positive = data["labels"] > 0
    weights[positive] = 1.0 / np.sqrt(np.maximum(popularity[positive], 1.0))
    weights = np.clip(weights, 0.25, 4.0)
    return data["labels"].astype(np.float32) * weights
