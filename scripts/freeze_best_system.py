#!/usr/bin/env python3
"""Freeze BEST_SYSTEM from overnight development reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.best_system import default_spec, write_spec
from recsys_loom.overnight.protocol import MAP_TOLERANCE, OVERNIGHT_DIR
from recsys_loom.overnight.selection import selected_lightgbm_kwargs, selected_trial


def _load(name: str) -> dict:
    path = OVERNIGHT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = _load("baseline_ranker.json")
    tune = _load("lgbm_tune.json")
    negatives = _load("hard_negatives.json")
    crosses = _load("cross_features.json")
    scaled = _load("scaled_supervision.json")
    catboost = _load("catboost_ranker.json")
    listwise = _load("listwise_reranker.json")
    weights = _load("longtail_weights.json")
    groups = _load("group_weights.json")
    spec = default_spec()
    architecture = spec["architecture"]
    decisions = []

    if tune.get("selected") and tune["selected"] != "baseline":
        architecture["ranker"].update(selected_lightgbm_kwargs())
        architecture["ranker"]["selected_trial"] = tune["selected"]
        chosen = selected_trial(tune)
        if chosen:
            architecture["ranker"]["params_update"] = chosen.get("params", {}).get(
                "params_update"
            )
        decisions.append(f"lightgbm:{tune['selected']}")
    else:
        decisions.append("lightgbm:baseline")
    lightgbm_mean = tune.get("selected_mean_map_at_12") or baseline.get(
        "development_mean_map_at_12"
    )

    if scaled.get("selected") == "scaled":
        architecture["training_snapshots"] = [
            "2020-08-03",
            "2020-08-10",
            "2020-08-17",
            "2020-08-24",
            "2020-08-31",
            "2020-09-07",
        ]
        decisions.append("supervision:scaled")
    else:
        decisions.append("supervision:current")

    if negatives.get("selected") and negatives["selected"] != "all_candidates":
        architecture["negative_sampling"] = negatives["selected"]
        decisions.append(f"negatives:{negatives['selected']}")
    else:
        architecture["negative_sampling"] = "all_candidates"
        decisions.append("negatives:all_candidates")

    if crosses.get("selected") == "targeted_crosses":
        architecture["feature_set"] = "six_source_lambdarank_v1_crosses"
        decisions.append("features:crosses")
    else:
        decisions.append("features:baseline")

    if weights.get("selected") == "inv_sqrt_popularity":
        architecture["label_weighting"] = "inv_sqrt_popularity"
        decisions.append("weights:inv_sqrt_popularity")
    else:
        architecture["label_weighting"] = None
        decisions.append("weights:none")

    group_selected = groups.get("selected")
    if group_selected and group_selected != "unweighted":
        architecture["group_weighting"] = group_selected
        decisions.append(f"group_weights:{group_selected}")
    else:
        architecture["group_weighting"] = None
        decisions.append("group_weights:none")

    cat_mean = (catboost.get("catboost") or {}).get("mean_map_at_12")
    if (
        catboost.get("selected") == "catboost_yetirank"
        and cat_mean is not None
        and lightgbm_mean is not None
        and cat_mean > float(lightgbm_mean) + MAP_TOLERANCE
    ):
        architecture["ranker"]["family"] = "catboost_yetirank"
        decisions.append("ranker:catboost")
    else:
        decisions.append("ranker:lightgbm")

    listwise_mean = None
    selected_listwise = listwise.get("selected")
    if selected_listwise and selected_listwise != "lambda_only":
        listwise_mean = (listwise.get("arms") or {}).get(selected_listwise, {}).get(
            "mean_map_at_12"
        )
    if (
        selected_listwise
        and selected_listwise != "lambda_only"
        and listwise_mean is not None
        and lightgbm_mean is not None
        and listwise_mean > float(lightgbm_mean) + MAP_TOLERANCE
    ):
        architecture["reranker"] = {
            "family": "set_transformer_listnet",
            "selected": selected_listwise,
        }
        decisions.append(f"reranker:{selected_listwise}")
    else:
        architecture["reranker"] = None
        decisions.append("reranker:none")

    spec["status"] = "frozen_for_holdout"
    spec["decisions"] = decisions
    spec["development_reports"] = {
        "baseline_mean": baseline.get("development_mean_map_at_12"),
        "tune_selected": tune.get("selected"),
        "tune_mean": tune.get("selected_mean_map_at_12"),
        "scaled_selected": scaled.get("selected"),
        "catboost_selected": catboost.get("selected"),
        "listwise_selected": listwise.get("selected"),
        "group_selected": groups.get("selected"),
    }
    path = write_spec(spec)
    print(json.dumps(spec, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
