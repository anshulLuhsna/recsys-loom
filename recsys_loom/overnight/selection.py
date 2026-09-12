"""Read the selected LightGBM trial for later overnight stages."""

from __future__ import annotations

import json
from typing import Any

from recsys_loom.overnight.protocol import BASELINE_RANKER, OVERNIGHT_DIR

DEFAULT_KWARGS = {
    "n_estimators": BASELINE_RANKER["n_estimators"],
    "learning_rate": BASELINE_RANKER["learning_rate"],
    "num_leaves": BASELINE_RANKER["num_leaves"],
    "min_child_samples": BASELINE_RANKER["min_child_samples"],
    "params_update": None,
}


def load_lgbm_tune() -> dict[str, Any]:
    path = OVERNIGHT_DIR / "lgbm_tune.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_trial(report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = report if report is not None else load_lgbm_tune()
    name = report.get("selected")
    for row in report.get("trials", []):
        if row.get("name") == name:
            return row
    return {}


def selected_lightgbm_kwargs() -> dict[str, Any]:
    trial = selected_trial()
    params = trial.get("params") or {}
    if not params:
        return dict(DEFAULT_KWARGS)
    update = params.get("params_update") or None
    return {
        "n_estimators": int(params.get("n_estimators", DEFAULT_KWARGS["n_estimators"])),
        "learning_rate": float(params.get("learning_rate", DEFAULT_KWARGS["learning_rate"])),
        "num_leaves": int(params.get("num_leaves", DEFAULT_KWARGS["num_leaves"])),
        "min_child_samples": int(
            params.get("min_child_samples", DEFAULT_KWARGS["min_child_samples"])
        ),
        "params_update": update if update else None,
    }


def selected_lightgbm_mean() -> float | None:
    report = load_lgbm_tune()
    value = report.get("selected_mean_map_at_12")
    return float(value) if value is not None else None
