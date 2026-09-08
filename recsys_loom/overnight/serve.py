"""Train and score snapshots from a frozen BEST_SYSTEM spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from catboost import CatBoostRanker, Pool
from numpy.typing import NDArray

from recsys_loom.overnight.listwise import (
    _stack_batch,
    fit_reranker,
    predict_shortlist,
    shortlist_groups,
)
from recsys_loom.overnight.ranking import combine_training, fit_ranker
from recsys_loom.overnight.transforms import (
    NEGATIVE_SCHEMES,
    add_crosses,
    downsample_snapshot,
    row_weights_for_scheme,
    weighted_labels,
)
from recsys_loom.ranking.ranker import train_ranker


@dataclass
class FittedSystem:
    family: str
    ranker: Any
    reranker: Any | None = None
    shortlist: int | None = None


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


def _ranker_kwargs(ranker: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": int(ranker.get("n_estimators", 300)),
        "learning_rate": float(ranker.get("learning_rate", 0.05)),
        "num_leaves": int(ranker.get("num_leaves", 63)),
        "min_child_samples": int(ranker.get("min_child_samples", 50)),
        "params_update": ranker.get("params_update") or None,
    }


def _fit_lightgbm(
    prepared: list[dict[str, Any]],
    spec: dict[str, Any],
) -> Any:
    architecture = spec.get("architecture", {})
    ranker = architecture.get("ranker", {})
    kwargs = _ranker_kwargs(ranker)
    uses_label_weights = architecture.get("label_weighting") == "inv_sqrt_popularity"
    group_scheme = architecture.get("group_weighting")
    if uses_label_weights or group_scheme:
        train = combine_training(prepared)
        return train_ranker(
            train["features"],
            weighted_labels(train) if uses_label_weights else train["labels"],
            train["groups"],
            [str(value) for value in train["feature_names"]],
            verbose=0,
            weight=row_weights_for_scheme(train, group_scheme),
            **kwargs,
        )
    return fit_ranker(prepared, **kwargs)


def _fit_catboost(prepared: list[dict[str, Any]], spec: dict[str, Any]) -> Any:
    architecture = spec.get("architecture", {})
    ranker = architecture.get("ranker", {})
    train = combine_training(prepared)
    labels = (
        weighted_labels(train)
        if architecture.get("label_weighting") == "inv_sqrt_popularity"
        else train["labels"]
    )
    group_id = np.repeat(np.arange(len(train["groups"])), train["groups"])
    pool = Pool(
        data=np.nan_to_num(train["features"], nan=0.0),
        label=labels,
        group_id=group_id,
        weight=row_weights_for_scheme(train, architecture.get("group_weighting")),
    )
    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=int(ranker.get("n_estimators", 300)),
        learning_rate=float(ranker.get("learning_rate", 0.05)),
        depth=6,
        random_seed=42,
        verbose=False,
    )
    model.fit(pool)
    return model


def _raw_scores(ranker: Any, features: NDArray) -> NDArray[np.float32]:
    matrix = np.nan_to_num(np.asarray(features), nan=0.0)
    return np.asarray(ranker.predict(matrix), dtype=np.float32)


def predict_scores(fitted: FittedSystem, validation: dict[str, Any]) -> NDArray[np.float32]:
    raw = _raw_scores(fitted.ranker, validation["features"])
    if fitted.reranker is None or fitted.shortlist is None:
        return raw
    scores = np.full(len(raw), -1.0e9, dtype=np.float32)
    groups = shortlist_groups(raw, validation, fitted.shortlist)
    fitted.reranker.eval()
    with torch.no_grad():
        for group in groups:
            features, _labels, mask = _stack_batch(
                [group],
                0,
                1,
                fitted.reranker.feature_mean,
                fitted.reranker.feature_std,
            )
            reranked = fitted.reranker(features, mask)[0].cpu().numpy()
            valid = int(mask[0].sum().item())
            scores[group["global_indices"]] = reranked[:valid]
    return scores


def predict_slate(
    fitted: FittedSystem,
    validation: dict[str, Any],
    article_ids: list[str],
) -> dict[int, list[str]]:
    raw = _raw_scores(fitted.ranker, validation["features"])
    if fitted.reranker is None or fitted.shortlist is None:
        predictions: dict[int, list[str]] = {}
        offset = 0
        for customer_index, group_size_value in enumerate(validation["groups"]):
            group_size = int(group_size_value)
            end = offset + group_size
            take = min(12, group_size)
            local = np.argpartition(raw[offset:end], -take)[-take:]
            local = local[np.argsort(-raw[offset:end][local])]
            predictions[customer_index] = [
                article_ids[int(validation["pair_article_indices"][offset + int(index)])]
                for index in local
            ]
            offset = end
        return predictions
    groups = shortlist_groups(raw, validation, fitted.shortlist)
    return predict_shortlist(fitted.reranker, groups, article_ids)


def fit_from_spec(snapshots: list[dict[str, Any]], spec: dict[str, Any]) -> FittedSystem:
    architecture = spec.get("architecture", {})
    ranker = architecture.get("ranker", {})
    family = ranker.get("family", "lightgbm_lambdarank")
    prepared = [
        transform_snapshot(snapshot, spec, training=True, seed=17 + index)
        for index, snapshot in enumerate(snapshots)
    ]
    if family == "catboost_yetirank":
        return FittedSystem(family=family, ranker=_fit_catboost(prepared, spec))
    if family != "lightgbm_lambdarank":
        raise ValueError(f"holdout/demo serving does not implement {family}")
    booster = _fit_lightgbm(prepared, spec)
    reranker_spec = architecture.get("reranker") or {}
    selected = str(reranker_spec.get("selected") or "")
    if selected.startswith("shortlist_"):
        shortlist = int(selected.split("_")[-1])
        train_groups = []
        for snapshot in prepared:
            scores = _raw_scores(booster, snapshot["features"])
            train_groups.extend(shortlist_groups(scores, snapshot, shortlist))
        reranker, _history = fit_reranker(train_groups)
        return FittedSystem(
            family=family,
            ranker=booster,
            reranker=reranker,
            shortlist=shortlist,
        )
    return FittedSystem(family=family, ranker=booster)
