#!/usr/bin/env python3
"""Encode H&M article images once with a frozen DINOv2 image tower."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "facebook/dinov2-small"
IMAGE_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "artifacts" / "image_retrieval"


def image_path(article_id: str) -> Path:
    return IMAGE_DIR / article_id[:3] / f"{article_id}.jpg"


def load_article_ids() -> list[str]:
    con = duckdb.connect()
    rows = con.sql(f"""
        SELECT article_id
        FROM read_csv(
            '{ROOT / "articles.csv"}',
            header=true,
            all_varchar=true
        )
        ORDER BY article_id
    """).fetchall()
    con.close()
    return [row[0] for row in rows]


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_arrays(
    article_ids: list[str],
    embedding_dim: int,
) -> tuple[np.memmap, np.memmap]:
    embeddings_path = OUTPUT_DIR / "image_embeddings.f32.npy"
    status_path = OUTPUT_DIR / "image_embedding_status.u8.npy"
    article_ids_path = OUTPUT_DIR / "article_ids.npy"
    if embeddings_path.exists() and status_path.exists() and article_ids_path.exists():
        saved_ids = np.load(article_ids_path)
        if list(saved_ids.astype(str)) != article_ids:
            raise RuntimeError("Article ordering changed; refuse to reuse embedding cache")
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        if embeddings.shape != (len(article_ids), embedding_dim):
            raise RuntimeError("Existing image embedding shape does not match model")
        return embeddings, status

    np.save(article_ids_path, np.asarray(article_ids))
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
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
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = select_device()
    print(f"Loading {MODEL_NAME} on {device}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, backend="pil")
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    embedding_dim = int(model.config.hidden_size)

    article_ids = load_article_ids()
    embeddings, status = initialize_arrays(article_ids, embedding_dim)
    missing_paths = 0
    for index, article_id in enumerate(article_ids):
        if status[index] == 0 and not image_path(article_id).exists():
            status[index] = 2
            missing_paths += 1
    status.flush()
    pending = np.flatnonzero(status == 0)
    print(
        f"Articles={len(article_ids):,}, pending={len(pending):,}, "
        f"already_encoded={int(np.sum(status == 1)):,}, "
        f"missing={int(np.sum(status == 2)):,}",
        flush=True,
    )

    encoded_this_run = 0
    failed_this_run = 0
    for batch_start in range(0, len(pending), batch_size):
        batch_indices = pending[batch_start : batch_start + batch_size]
        valid_indices: list[int] = []
        images: list[Image.Image] = []
        for index_value in batch_indices:
            index = int(index_value)
            try:
                with Image.open(image_path(article_ids[index])) as image:
                    images.append(image.convert("RGB"))
                valid_indices.append(index)
            except (OSError, UnidentifiedImageError):
                status[index] = 2
                failed_this_run += 1
        if images:
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                outputs = model(pixel_values=pixel_values)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    pooled = outputs.last_hidden_state[:, 0]
                normalized = F.normalize(pooled.float(), dim=1)
            embeddings[valid_indices] = normalized.cpu().numpy()
            status[valid_indices] = 1
            encoded_this_run += len(valid_indices)

        completed = batch_start + len(batch_indices)
        if completed % (batch_size * 20) == 0 or completed == len(pending):
            embeddings.flush()
            status.flush()
            elapsed = time.monotonic() - started
            rate = encoded_this_run / elapsed if elapsed else 0.0
            print(
                f"  {completed:,}/{len(pending):,} pending processed "
                f"({rate:.1f} images/s)",
                flush=True,
            )

    metadata = {
        "model_name": MODEL_NAME,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "embedding_dim": embedding_dim,
        "device": str(device),
        "article_count": len(article_ids),
        "encoded_count": int(np.sum(status == 1)),
        "missing_or_failed_count": int(np.sum(status == 2)),
        "missing_paths_found_this_run": missing_paths,
        "failed_images_this_run": failed_this_run,
        "embedding_norm_mean": float(
            np.linalg.norm(embeddings[status == 1], axis=1).mean()
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    (OUTPUT_DIR / "image_embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Metadata: {OUTPUT_DIR / 'image_embedding_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
