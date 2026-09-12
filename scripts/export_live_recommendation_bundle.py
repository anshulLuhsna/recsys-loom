#!/usr/bin/env python3
"""Export BEST_SYSTEM model and per-customer features for live inference."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.live_recommender import SERVING_DIR
from recsys_loom.overnight.best_system import load_spec
from recsys_loom.overnight.db import catalog_article_ids, connect
from recsys_loom.overnight.protocol import DEV_CUSTOMERS, ensure_directories
from recsys_loom.overnight.ranking import (
    apply_baseline_budget,
    development_specs,
    load_or_build_snapshot,
)
from recsys_loom.overnight.serve import (
    fit_from_spec,
    transform_snapshot,
    uses_extra_training,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.monotonic()
    ensure_directories()
    connection = connect()
    article_ids = catalog_article_ids(connection)
    article_to_index = {
        article_id: index for index, article_id in enumerate(article_ids)
    }
    spec = load_spec()
    specifications = development_specs(
        DEV_CUSTOMERS,
        include_extra_training=uses_extra_training(spec),
    )
    training_specifications = [
        specification
        for specification in specifications
        if specification.cutoff != spec["architecture"]["serving_cutoff"]
    ]
    train = []
    for specification in training_specifications:
        data, _relevance = load_or_build_snapshot(
            connection,
            specification,
            article_to_index,
        )
        train.append(apply_baseline_budget(data))
    del data
    gc.collect()
    serving_specification = next(
        specification
        for specification in specifications
        if specification.cutoff == spec["architecture"]["serving_cutoff"]
    )
    print("Training frozen BEST_SYSTEM for serving...", flush=True)
    fitted = fit_from_spec(train, spec)
    if fitted.family != "catboost_yetirank" or fitted.reranker is not None:
        raise RuntimeError("Live export currently requires plain CatBoost YetiRank")
    del train
    gc.collect()

    serving_data, _relevance = load_or_build_snapshot(
        connection,
        serving_specification,
        article_to_index,
    )
    serving_raw = apply_baseline_budget(serving_data)
    del serving_data
    gc.collect()

    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SERVING_DIR / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    customer_dir = SERVING_DIR / "customer_features"
    if customer_dir.exists():
        shutil.rmtree(customer_dir)
    customer_dir.mkdir()
    model_path = SERVING_DIR / "BEST_SYSTEM.cbm"
    fitted.ranker.save_model(str(model_path), format="cbm")
    np.save(SERVING_DIR / "article_ids.npy", np.asarray(article_ids))

    serving = transform_snapshot(serving_raw, spec, training=False)
    feature_names = [str(value) for value in serving["feature_names"]]
    customer_ids = [str(value) for value in serving["customer_ids"]]
    offset = 0
    candidate_counts = []
    for customer_index, group_size_value in enumerate(serving["groups"]):
        group_size = int(group_size_value)
        end = offset + group_size
        customer_id = customer_ids[customer_index]
        np.savez_compressed(
            customer_dir / f"{customer_id}.npz",
            features=np.asarray(serving["features"][offset:end], dtype=np.float32),
            article_indices=np.asarray(
                serving["pair_article_indices"][offset:end],
                dtype=np.int32,
            ),
        )
        candidate_counts.append(group_size)
        offset = end
        if (customer_index + 1) % 250 == 0:
            print(
                f"  exported {customer_index + 1:,}/{len(customer_ids):,} customers",
                flush=True,
            )

    manifest = {
        "model_version": str(spec.get("name", "BEST_SYSTEM")),
        "ranker_family": fitted.family,
        "training_snapshots": spec["architecture"]["training_snapshots"],
        "serving_cutoff": spec["architecture"]["serving_cutoff"],
        "feature_names": feature_names,
        "customer_ids": customer_ids,
        "customer_count": len(customer_ids),
        "candidate_rows": int(sum(candidate_counts)),
        "candidate_count_min": int(min(candidate_counts)),
        "candidate_count_mean": float(np.mean(candidate_counts)),
        "candidate_count_max": int(max(candidate_counts)),
        "model_sha256": sha256(model_path),
        "runtime_seconds": time.monotonic() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
