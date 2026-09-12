#!/usr/bin/env python3
"""Export temporal-safe TT embeddings for dense candidate scoring."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.ranking.cached_data import load_candidate_pool
from recsys_loom.two_tower.data import (
    Catalog,
    UserNumericalNormalizer,
    build_catalog,
    build_snapshot,
)
from recsys_loom.two_tower.training import (
    encode_all_items,
    encode_snapshot_users,
)
from scripts.run_two_tower_text_ablation import make_model, setup_connection
from scripts.run_two_tower_text_ranking import FINAL_FOLD, make_specs

TEXT_DIR = ROOT / "artifacts" / "text_retrieval"
ABLATION_DIR = TEXT_DIR / "two_tower_ablation"
OUTPUT_DIR = TEXT_DIR / "dense_scores"


def load_model(
    arm: str,
    cutoff: str,
    customer_count: int,
    catalog: Catalog,
    device: str,
) -> tuple[torch.nn.Module, UserNumericalNormalizer]:
    checkpoint_path = (
        ABLATION_DIR
        / arm
        / f"model_{cutoff}_{customer_count}.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    text_dim = int(checkpoint["text_embedding_dim"])
    model = make_model(catalog, text_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(torch.device(device))
    model.eval()
    normalizer = UserNumericalNormalizer(
        mean=np.asarray(checkpoint["normalizer_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["normalizer_std"], dtype=np.float32),
    )
    return model, normalizer


def main() -> None:
    training_customers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    final_customers = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    con = duckdb.connect()
    setup_connection(con)
    catalog = build_catalog(con)
    specifications = make_specs(
        "metadata",
        training_customers,
        final_customers,
    )
    text_article_ids = np.load(TEXT_DIR / "text_article_ids.npy").astype(str)
    if text_article_ids.tolist() != catalog.article_ids:
        raise RuntimeError("Text embedding article IDs do not align with catalog")
    text_embeddings = np.load(
        TEXT_DIR / "text_embeddings.f32.npy",
        mmap_mode="r",
    )
    np.save(OUTPUT_DIR / "article_ids.npy", np.asarray(catalog.article_ids))
    report: dict[str, object] = {
        "device": device,
        "snapshots": {},
    }

    for index, specification in enumerate(specifications):
        print(f"Scoring snapshot {specification.cutoff}...", flush=True)
        pool = load_candidate_pool(specification)
        snapshot = build_snapshot(
            con,
            catalog,
            specification.cutoff,
            specification.target_start,
            specification.target_end,
            pool.customer_ids,
            observed_items_only=True,
        )
        snapshot_dir = OUTPUT_DIR / specification.cutoff
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            snapshot_dir / "customer_ids.npy",
            np.asarray(pool.customer_ids),
        )
        snapshot_report: dict[str, object] = {
            "customers": len(pool.customer_ids),
            "arms": {},
        }
        for arm in ("metadata", "metadata_text"):
            model, normalizer = load_model(
                arm,
                specification.cutoff,
                final_customers if index == FINAL_FOLD else training_customers,
                catalog,
                device,
            )
            arm_text = text_embeddings if arm == "metadata_text" else None
            item_embeddings = encode_all_items(
                model,
                catalog,
                text_embeddings=arm_text,
            )
            user_embeddings = encode_snapshot_users(
                model,
                snapshot,
                normalizer,
                text_embeddings=arm_text,
            )
            np.save(
                snapshot_dir / f"{arm}_item_embeddings.npy",
                item_embeddings,
            )
            np.save(
                snapshot_dir / f"{arm}_user_embeddings.npy",
                user_embeddings,
            )
            snapshot_report["arms"][arm] = {
                "item_shape": list(item_embeddings.shape),
                "user_shape": list(user_embeddings.shape),
                "item_norm_mean": float(
                    np.linalg.norm(item_embeddings, axis=1).mean()
                ),
                "user_norm_mean": float(
                    np.linalg.norm(user_embeddings, axis=1).mean()
                ),
            }
            del model, normalizer, item_embeddings, user_embeddings
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        report["snapshots"][specification.cutoff] = snapshot_report
        del pool, snapshot
        gc.collect()

    report_path = OUTPUT_DIR / (
        f"dense_embedding_export_{training_customers}_{final_customers}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
