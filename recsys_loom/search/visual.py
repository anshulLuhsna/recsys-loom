"""Text-to-image retrieval using a frozen CLIP image matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from transformers import AutoProcessor, CLIPModel

from recsys_loom.overnight.protocol import ROOT

VISUAL_DIR = ROOT / "artifacts" / "visual_search"
ENCODER_NAME = "openai/clip-vit-base-patch32"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class VisualIndex:
    def __init__(
        self,
        embeddings: NDArray[np.floating],
        article_ids: list[str],
        available: NDArray[np.bool_],
        processor: Any | None = None,
        model: CLIPModel | None = None,
        device: torch.device | None = None,
    ):
        self.embeddings = embeddings
        self.article_ids = article_ids
        self.available = available
        self.processor = processor
        self.model = model
        self.device = device or select_device()

    @classmethod
    def load(
        cls,
        directory: Path | None = None,
        load_encoder: bool = True,
    ) -> "VisualIndex":
        visual_dir = directory or VISUAL_DIR
        metadata = json.loads(
            (visual_dir / "visual_embedding_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        model_name = str(metadata["model_name"])
        if model_name != ENCODER_NAME:
            raise RuntimeError(
                f"Visual embeddings use {model_name}, expected {ENCODER_NAME}"
            )
        embeddings = np.load(
            visual_dir / "visual_embeddings.f32.npy",
            mmap_mode="r",
        )
        article_ids = (
            np.load(visual_dir / "visual_article_ids.npy").astype(str).tolist()
        )
        status = np.load(
            visual_dir / "visual_embedding_status.u8.npy",
            mmap_mode="r",
        )
        device = select_device()
        processor = None
        model = None
        if load_encoder:
            processor = AutoProcessor.from_pretrained(ENCODER_NAME)
            model = CLIPModel.from_pretrained(ENCODER_NAME).to(device)
            model.eval()
        return cls(
            embeddings=embeddings,
            article_ids=article_ids,
            available=np.asarray(status == 1),
            processor=processor,
            model=model,
            device=device,
        )

    def _load_encoder(self) -> None:
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(ENCODER_NAME)
        if self.model is None:
            self.model = CLIPModel.from_pretrained(ENCODER_NAME).to(self.device)
            self.model.eval()

    def encode_query(self, query: str) -> NDArray[np.float32]:
        self._load_encoder()
        assert self.processor is not None
        assert self.model is not None
        inputs = self.processor(
            text=[f"a product photo of {query}"],
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs).pooler_output
            features = F.normalize(features.float(), dim=1)
        return np.asarray(features[0].cpu().numpy(), dtype=np.float32)

    def search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        query_vector = self.encode_query(query)
        scores = np.asarray(self.embeddings @ query_vector, dtype=np.float32)
        scores[~self.available] = -np.inf
        take = min(k, int(np.sum(self.available)))
        if take == 0:
            return []
        local = np.argpartition(scores, -take)[-take:]
        local = local[np.argsort(-scores[local])]
        return [
            (self.article_ids[int(index)], float(scores[int(index)]))
            for index in local
        ]
