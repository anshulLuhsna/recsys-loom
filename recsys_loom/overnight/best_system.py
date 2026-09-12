"""Frozen BEST_SYSTEM specification written after development selection."""

from __future__ import annotations

import json
from pathlib import Path

from recsys_loom.overnight.protocol import (
    BASELINE_RANKER,
    OVERNIGHT_DIR,
    TT_BUDGET,
)

BEST_SYSTEM_PATH = OVERNIGHT_DIR / "BEST_SYSTEM.json"


def default_spec() -> dict[str, object]:
    return {
        "name": "BEST_SYSTEM",
        "status": "provisional_baseline",
        "architecture": {
            "candidate_sources": BASELINE_RANKER["sources"],
            "two_tower_k": TT_BUDGET,
            "ranker": BASELINE_RANKER,
            "reranker": None,
            "feature_set": "six_source_lambdarank_v1",
            "training_snapshots": [
                "2020-08-17",
                "2020-08-24",
                "2020-08-31",
                "2020-09-07",
            ],
            "serving_cutoff": "2020-09-15",
            "group_weighting": None,
        },
        "selection": {
            "development_mean_map_at_12": 0.027068315668018046,
            "protocol": "customer_holdout_final_evaluation",
        },
    }


def write_spec(spec: dict[str, object]) -> Path:
    BEST_SYSTEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_SYSTEM_PATH.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return BEST_SYSTEM_PATH


def load_spec() -> dict[str, object]:
    if BEST_SYSTEM_PATH.exists():
        return json.loads(BEST_SYSTEM_PATH.read_text(encoding="utf-8"))
    return default_spec()
