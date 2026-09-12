"""Lazy CatBoost inference over exported per-customer candidate features."""

from __future__ import annotations

import hashlib
import json
import time
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostRanker

from recsys_loom.overnight.protocol import ROOT
from recsys_loom.search.catalog import Article

SERVING_DIR = ROOT / "artifacts" / "serving"
SOURCE_NAMES = (
    "recent_7d_pop",
    "repeat_purchase",
    "cooccurrence",
    "als",
    "content",
    "two_tower",
)


class LiveRecommendationStore:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or SERVING_DIR
        manifest_path = self.directory / "manifest.json"
        self.manifest_error: str | None = None
        self.manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    self.manifest = value
                else:
                    self.manifest_error = "manifest root must be an object"
            except (OSError, json.JSONDecodeError) as exc:
                self.manifest_error = f"{type(exc).__name__}: {exc}"
        self.supported_customers = set(
            self.manifest.get("customer_ids", []) if self.manifest else []
        )

    @property
    def configured(self) -> bool:
        return self.directory.exists() and any(self.directory.iterdir())

    @property
    def available(self) -> bool:
        return self.manifest is not None and not self.validation_errors

    @cached_property
    def validation_errors(self) -> list[str]:
        if not self.configured:
            return []
        errors = []
        if self.manifest_error:
            errors.append(self.manifest_error)
        if self.manifest is None:
            errors.append("manifest.json is missing or invalid")
        else:
            for key in ("model_version", "feature_names", "customer_ids"):
                if not self.manifest.get(key):
                    errors.append(f"manifest field {key} is missing or empty")
        model_path = self.directory / "BEST_SYSTEM.cbm"
        article_ids_path = self.directory / "article_ids.npy"
        customer_dir = self.directory / "customer_features"
        if not model_path.is_file():
            errors.append("BEST_SYSTEM.cbm is missing")
        if not article_ids_path.is_file():
            errors.append("article_ids.npy is missing")
        if not customer_dir.is_dir():
            errors.append("customer_features directory is missing")
        if (
            self.manifest
            and model_path.is_file()
            and self.manifest.get("model_sha256")
        ):
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if digest != self.manifest["model_sha256"]:
                errors.append("BEST_SYSTEM.cbm checksum does not match manifest")
        return errors

    @cached_property
    def model(self) -> CatBoostRanker:
        model = CatBoostRanker()
        model.load_model(str(self.directory / "BEST_SYSTEM.cbm"))
        return model

    @cached_property
    def article_ids(self) -> list[str]:
        return np.load(self.directory / "article_ids.npy").astype(str).tolist()

    @cached_property
    def feature_names(self) -> list[str]:
        if self.manifest is None:
            return []
        return [str(value) for value in self.manifest["feature_names"]]

    def _sources(self, row: np.ndarray) -> list[str]:
        sources = []
        for source in SOURCE_NAMES:
            column_name = f"is_{source}"
            if column_name in self.feature_names:
                column = self.feature_names.index(column_name)
                if row[column] == 1.0:
                    sources.append(source)
        return sources

    @lru_cache(maxsize=128)
    def _customer_features(
        self,
        customer_id: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        path = self.directory / "customer_features" / f"{customer_id}.npz"
        if not path.exists():
            return None
        with np.load(path) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float32)
            article_indices = np.asarray(arrays["article_indices"], dtype=np.int32)
        return features, article_indices

    def recommend(
        self,
        customer_id: str,
        catalog: dict[str, Article],
    ) -> dict[str, object] | None:
        if not self.available or customer_id not in self.supported_customers:
            return None
        started = time.perf_counter()
        candidate_data = self._customer_features(customer_id)
        if candidate_data is None:
            return None
        features, article_indices = candidate_data
        scores = np.asarray(
            self.model.predict(np.nan_to_num(features, nan=0.0)),
            dtype=np.float32,
        )
        take = min(12, len(scores))
        if take == 0:
            return None
        local = np.argpartition(scores, -take)[-take:]
        local = local[np.argsort(-scores[local])]
        recommendations = []
        for index_value in local:
            index = int(index_value)
            article_id = self.article_ids[int(article_indices[index])]
            article = catalog.get(article_id)
            if article is None:
                continue
            card = article.to_card()
            card["score"] = float(scores[index])
            card["sources"] = self._sources(features[index])
            recommendations.append(card)
        return {
            "customer_id": customer_id,
            "model_version": str(self.manifest["model_version"]),
            "serving_mode": "live_catboost",
            "candidate_count": len(features),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "recommendations": recommendations,
        }
