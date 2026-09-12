"""LightGBM LambdaRank ranker.

Trains a grouped ranking model where each customer is a query group.
The model learns to order candidates within a customer so that
purchased items rank higher than non-purchased ones.

Training objective: LambdaRank (listwise, optimizes NDCG)
Evaluation metric: MAP@12 on the final sorted slate
"""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from recsys_loom.metrics import average_precision_at_k


def build_labels(
    pairs: Sequence[tuple[str, str]],
    relevant_by_customer: dict[str, set[str]],
) -> NDArray[np.int32]:
    """Binary labels: 1 if the customer bought this article in the target period."""
    labels = np.zeros(len(pairs), dtype=np.int32)
    for i, (cid, aid) in enumerate(pairs):
        if aid in relevant_by_customer.get(cid, set()):
            labels[i] = 1
    return labels


def build_groups(
    pairs: Sequence[tuple[str, str]],
) -> tuple[NDArray[np.int32], list[str]]:
    """Group sizes for LambdaRank: one group per customer.

    Returns:
        groups: array of group sizes (number of candidates per customer)
        group_customers: ordered customer IDs matching each group
    """
    groups: list[int] = []
    group_customers: list[str] = []
    current_cid = None
    count = 0

    for cid, _ in pairs:
        if cid != current_cid:
            if current_cid is not None:
                groups.append(count)
                group_customers.append(current_cid)
            current_cid = cid
            count = 0
        count += 1

    if current_cid is not None:
        groups.append(count)
        group_customers.append(current_cid)

    return np.array(groups, dtype=np.int32), group_customers


def train_ranker(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int32],
    groups_train: NDArray[np.int32],
    feature_names: list[str],
    X_valid: NDArray[np.float32] | None = None,
    y_valid: NDArray[np.int32] | None = None,
    groups_valid: NDArray[np.int32] | None = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_child_samples: int = 50,
    verbose: int = 25,
    params_update: dict | None = None,
    weight: NDArray[np.float32] | None = None,
) -> lgb.Booster:
    """Train a LambdaRank model."""
    train_data = lgb.Dataset(
        X_train, label=y_train, group=groups_train,
        weight=weight,
        feature_name=feature_names,
        free_raw_data=False,
    )

    valid_sets = [train_data]
    valid_names = ["train"]
    if X_valid is not None and y_valid is not None and groups_valid is not None:
        valid_data = lgb.Dataset(
            X_valid, label=y_valid, group=groups_valid,
            feature_name=feature_names,
            free_raw_data=False,
        )
        valid_sets.append(valid_data)
        valid_names.append("valid")

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [12],
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    if params_update:
        params.update(params_update)

    callbacks = []
    if verbose > 0:
        callbacks.append(lgb.log_evaluation(period=verbose))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    return model


def predict_and_rank(
    model: lgb.Booster,
    X: NDArray[np.float32],
    pairs: Sequence[tuple[str, str]],
    top_k: int = 12,
) -> dict[str, list[str]]:
    """Score all candidates and return top-K per customer."""
    scores = model.predict(X)

    by_customer: dict[str, list[tuple[float, str]]] = {}
    for i, (cid, aid) in enumerate(pairs):
        by_customer.setdefault(cid, []).append((scores[i], aid))

    result: dict[str, list[str]] = {}
    for cid, items in by_customer.items():
        items.sort(key=lambda x: -x[0])
        result[cid] = [aid for _, aid in items[:top_k]]

    return result


def evaluate_ranking(
    predictions: dict[str, list[str]],
    relevant_by_customer: dict[str, set[str]],
    k: int = 12,
) -> dict[str, float]:
    """Compute MAP@K and supporting metrics over predictions."""
    aps: list[float] = []
    hits = 0
    total_relevant = 0

    for cid, relevant in relevant_by_customer.items():
        predicted = predictions.get(cid, [])
        ap = average_precision_at_k(relevant, predicted, k)
        aps.append(ap)
        hits += len(set(predicted[:k]).intersection(relevant))
        total_relevant += len(relevant)

    n = len(relevant_by_customer)
    return {
        f"map_at_{k}": sum(aps) / n if n else 0.0,
        f"recall_at_{k}": hits / total_relevant if total_relevant else 0.0,
        f"hit_rate_at_{k}": sum(1 for ap in aps if ap > 0) / n if n else 0.0,
        "customers": n,
    }
