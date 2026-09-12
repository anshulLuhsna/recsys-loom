#!/usr/bin/env python3
"""Build and query an image ANN index outside the PyTorch process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.sources.two_tower import TwoTowerIndex, ann_exact_diagnostics


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: run_image_ann.py EMBEDDING_DIR QUERY_NPZ RESULT_NPZ METRICS_JSON"
        )
    embedding_directory = Path(sys.argv[1])
    query_path = Path(sys.argv[2])
    result_path = Path(sys.argv[3])
    metrics_path = Path(sys.argv[4])
    item_embeddings = np.load(
        embedding_directory / "image_embeddings.f32.npy",
        mmap_mode="r",
    )
    with np.load(query_path) as arrays:
        user_embeddings = arrays["user_embeddings"].astype(np.float32)
        eligible_item_indices = arrays["eligible_item_indices"].astype(np.int64)

    index = TwoTowerIndex.build(
        item_embeddings=item_embeddings,
        eligible_item_indices=eligible_item_indices,
        all_article_ids=[],
    )
    diagnostics = ann_exact_diagnostics(
        index,
        user_embeddings,
        item_embeddings,
        k=20,
        max_users=50,
    )
    scores, article_indices = index.search(user_embeddings, 500)
    np.savez_compressed(
        result_path,
        scores=scores,
        article_indices=article_indices,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "index_type": "IVFFlat",
                "indexed_articles": int(index.index.ntotal),
                "nlist": index.nlist,
                "nprobe": index.nprobe,
                "ann_exact_top20": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
