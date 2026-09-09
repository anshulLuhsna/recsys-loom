#!/usr/bin/env python3
"""Encode catalog images with CLIP for text-to-image search."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.search.catalog import load_articles
from recsys_loom.search.visual import ENCODER_NAME, VISUAL_DIR, select_device

IMAGE_DIR = ROOT / "images"


def image_path(article_id: str) -> Path:
    return IMAGE_DIR / article_id[:3] / f"{article_id}.jpg"


def initialize_arrays(
    article_ids: list[str],
    embedding_dim: int,
) -> tuple[np.memmap, np.memmap]:
    embeddings_path = VISUAL_DIR / "visual_embeddings.f32.npy"
    status_path = VISUAL_DIR / "visual_embedding_status.u8.npy"
    article_ids_path = VISUAL_DIR / "visual_article_ids.npy"
    if embeddings_path.exists() and status_path.exists() and article_ids_path.exists():
        saved_ids = np.load(article_ids_path)
        if list(saved_ids.astype(str)) != article_ids:
            raise RuntimeError("Article ordering changed; refusing to reuse CLIP cache")
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        if embeddings.shape != (len(article_ids), embedding_dim):
            raise RuntimeError("Existing CLIP embedding shape does not match model")
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
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = select_device()
    print(f"Loading {ENCODER_NAME} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(ENCODER_NAME)
    model = CLIPModel.from_pretrained(ENCODER_NAME).to(device)
    model.eval()
    embedding_dim = int(model.config.projection_dim)

    article_ids = sorted(load_articles())
    embeddings, status = initialize_arrays(article_ids, embedding_dim)
    for index, article_id in enumerate(article_ids):
        if status[index] == 0 and not image_path(article_id).exists():
            status[index] = 2
    status.flush()
    pending = np.flatnonzero(status == 0)
    print(
        f"Articles={len(article_ids):,}, pending={len(pending):,}, "
        f"encoded={int(np.sum(status == 1)):,}, "
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
                features = model.get_image_features(
                    pixel_values=pixel_values
                ).pooler_output
                features = F.normalize(features.float(), dim=1)
            embeddings[valid_indices] = features.cpu().numpy()
            status[valid_indices] = 1
            encoded_this_run += len(valid_indices)

        completed = batch_start + len(batch_indices)
        if completed % (batch_size * 20) == 0 or completed == len(pending):
            embeddings.flush()
            status.flush()
            elapsed = time.monotonic() - started
            rate = encoded_this_run / elapsed if elapsed else 0.0
            print(
                f"  {completed:,}/{len(pending):,} images "
                f"({rate:.1f} images/s)",
                flush=True,
            )

    metadata = {
        "model_name": ENCODER_NAME,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "embedding_dim": embedding_dim,
        "device": str(device),
        "article_count": len(article_ids),
        "encoded_count": int(np.sum(status == 1)),
        "missing_or_failed_count": int(np.sum(status == 2)),
        "failed_images_this_run": failed_this_run,
        "runtime_seconds": time.monotonic() - started,
    }
    metadata_path = VISUAL_DIR / "visual_embedding_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
