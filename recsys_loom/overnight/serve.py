"""Train and transform snapshots from a frozen BEST_SYSTEM spec."""

from __future__ import annotations

from typing import Any

from recsys_loom.overnight.ranking import combine_training, fit_ranker
from recsys_loom.overnight.transforms import (
    NEGATIVE_SCHEMES,
    add_crosses,
    downsample_snapshot,
    weighted_labels,
)
from recsys_loom.ranking.ranker import train_ranker


def uses_extra_training(spec: dict[str, Any]) -> bool:
    snapshots = spec.get("architecture", {}).get("training_snapshots", [])
    return "2020-08-03" in snapshots or "2020-08-10" in snapshots


def transform_snapshot(
    data: dict[str, Any],
    spec: dict[str, Any],
    *,
    training: bool,
    seed: int = 17,
) -> dict[str, Any]:
    architecture = spec.get("architecture", {})
    transformed = data
    if architecture.get("feature_set") == "six_source_lambdarank_v1_crosses":
        transformed = add_crosses(transformed)
    scheme = architecture.get("negative_sampling")
    if training and scheme and scheme != "all_candidates":
        hard, random = NEGATIVE_SCHEMES[scheme]
        transformed = downsample_snapshot(transformed, hard, random, seed)
    return transformed


def fit_from_spec(snapshots: list[dict[str, Any]], spec: dict[str, Any]) -> Any:
    architecture = spec.get("architecture", {})
    if architecture.get("reranker"):
        raise ValueError("listwise rerank is not wired into holdout/demo serving")
    ranker = architecture.get("ranker", {})
    family = ranker.get("family", "lightgbm_lambdarank")
    if family != "lightgbm_lambdarank":
        raise ValueError(f"holdout/demo serving does not implement {family}")
    prepared = [
        transform_snapshot(snapshot, spec, training=True, seed=17 + index)
        for index, snapshot in enumerate(snapshots)
    ]
    params_update = ranker.get("params_update") or None
    if architecture.get("label_weighting") == "inv_sqrt_popularity":
        train = combine_training(prepared)
        return train_ranker(
            train["features"],
            weighted_labels(train),
            train["groups"],
            [str(value) for value in train["feature_names"]],
            verbose=0,
            n_estimators=int(ranker.get("n_estimators", 300)),
            learning_rate=float(ranker.get("learning_rate", 0.05)),
            num_leaves=int(ranker.get("num_leaves", 63)),
            min_child_samples=int(ranker.get("min_child_samples", 50)),
            params_update=params_update,
        )
    return fit_ranker(
        prepared,
        n_estimators=int(ranker.get("n_estimators", 300)),
        learning_rate=float(ranker.get("learning_rate", 0.05)),
        num_leaves=int(ranker.get("num_leaves", 63)),
        min_child_samples=int(ranker.get("min_child_samples", 50)),
        params_update=params_update,
    )
