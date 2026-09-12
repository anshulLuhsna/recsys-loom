#!/usr/bin/env python3
"""Precompute fixed, normalized article text embeddings."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = ROOT / "artifacts" / "text_retrieval"


def load_articles() -> tuple[list[str], list[str]]:
    con = duckdb.connect()
    rows = con.sql(f"""
        SELECT article_id, prod_name, detail_desc
        FROM read_csv(
            '{ROOT / "articles.csv"}',
            header=true,
            all_varchar=true
        )
        ORDER BY article_id
    """).fetchall()
    con.close()
    article_ids = [row[0] for row in rows]
    texts = [
        (
            f"Product: {(row[1] or '').strip()}. "
            f"Description: {(row[2] or '').strip()}"
        )
        for row in rows
    ]
    return article_ids, texts


def initialize_arrays(
    article_ids: list[str],
    embedding_dim: int,
) -> tuple[np.memmap, np.memmap]:
    embedding_path = OUTPUT_DIR / "text_embeddings.f32.npy"
    status_path = OUTPUT_DIR / "text_embedding_status.u8.npy"
    ids_path = OUTPUT_DIR / "text_article_ids.npy"
    if embedding_path.exists() and status_path.exists() and ids_path.exists():
        saved_ids = np.load(ids_path)
        if list(saved_ids.astype(str)) != article_ids:
            raise RuntimeError("Article ordering changed; refuse to reuse text cache")
        embeddings = np.load(embedding_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        if embeddings.shape != (len(article_ids), embedding_dim):
            raise RuntimeError("Existing text embedding shape does not match model")
        return embeddings, status

    np.save(ids_path, np.asarray(article_ids))
    embeddings = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(article_ids), embedding_dim),
    )
    status = np.lib.format.open_memmap(
        status_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(article_ids),),
    )
    embeddings[:] = 0.0
    status[:] = 0
    embeddings.flush()
    status.flush()
    return embeddings, status


def main() -> None:
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.monotonic()
    print(f"Loading {MODEL_NAME} on {device}...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)
    embedding_dim = int(model.get_sentence_embedding_dimension())
    article_ids, texts = load_articles()
    embeddings, status = initialize_arrays(article_ids, embedding_dim)
    pending = np.flatnonzero(status == 0)
    print(
        f"Articles={len(article_ids):,}, pending={len(pending):,}, "
        f"already_encoded={int(np.sum(status == 1)):,}",
        flush=True,
    )

    encoded = 0
    for start in range(0, len(pending), chunk_size):
        indices = pending[start : start + chunk_size]
        vectors = model.encode(
            [texts[int(index)] for index in indices],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        embeddings[indices] = vectors.astype(np.float32)
        status[indices] = 1
        embeddings.flush()
        status.flush()
        encoded += len(indices)
        elapsed = time.monotonic() - started
        print(
            f"  {encoded:,}/{len(pending):,} encoded "
            f"({encoded / elapsed:.1f} articles/s)",
            flush=True,
        )

    ids_digest = hashlib.sha256(
        "\n".join(article_ids).encode("utf-8")
    ).hexdigest()
    norms = np.linalg.norm(embeddings, axis=1)
    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dim": embedding_dim,
        "device": device,
        "article_count": len(article_ids),
        "encoded_count": int(np.sum(status == 1)),
        "coverage": float(np.mean(status == 1)),
        "article_id_sha256": ids_digest,
        "alignment_verified": list(np.load(OUTPUT_DIR / "text_article_ids.npy").astype(str))
        == article_ids,
        "embedding_norm": {
            "minimum": float(norms.min()),
            "mean": float(norms.mean()),
            "maximum": float(norms.max()),
        },
        "storage_bytes": int(embeddings.nbytes + status.nbytes),
        "text_format": "Product: {prod_name}. Description: {detail_desc}",
        "runtime_seconds": time.monotonic() - started,
    }
    (OUTPUT_DIR / "text_embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Metadata: {OUTPUT_DIR / 'text_embedding_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
