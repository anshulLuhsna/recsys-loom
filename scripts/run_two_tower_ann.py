#!/usr/bin/env python3
"""Build/search the FAISS index in an isolated process.

PyTorch and FAISS load conflicting OpenMP runtimes on this macOS environment,
so learned embedding generation and ANN search must not share a process.
"""

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
            "Usage: run_two_tower_ann.py EMBEDDINGS_NPZ RESULTS_NPZ INDEX_PATH METRICS_JSON"
        )

    embeddings_path = Path(sys.argv[1])
    results_path = Path(sys.argv[2])
    index_path = Path(sys.argv[3])
    metrics_path = Path(sys.argv[4])

    arrays = np.load(embeddings_path)
    item_embeddings = arrays["item_embeddings"].astype(np.float32)
    user_embeddings = arrays["user_embeddings"].astype(np.float32)
    eligible_item_indices = arrays["eligible_item_indices"].astype(np.int64)

    index = TwoTowerIndex.build(
        item_embeddings=item_embeddings,
        eligible_item_indices=eligible_item_indices,
        all_article_ids=[],
    )
    diagnostics = ann_exact_diagnostics(
        index=index,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        k=20,
        max_users=50,
    )
    scores, article_indices = index.search(user_embeddings, 500)
    index.save(str(index_path))
    np.savez_compressed(
        results_path,
        scores=scores,
        article_indices=article_indices,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "ann_exact_top20": diagnostics,
                "index_type": "IVFFlat",
                "nlist": index.nlist,
                "nprobe": index.nprobe,
                "indexed_articles": int(index.index.ntotal),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
