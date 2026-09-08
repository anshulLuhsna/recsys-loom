#!/usr/bin/env python3
"""Fill OVERNIGHT_RESULTS.md and print the morning summary from reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.protocol import OVERNIGHT_DIR


def _load(name: str) -> dict:
    path = OVERNIGHT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_log() -> str:
    completed = subprocess.run(
        ["git", "log", "--oneline", "c4683da^..HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def main() -> None:
    baseline = _load("baseline_ranker.json")
    spec = _load("BEST_SYSTEM.json")
    holdout = _load("holdout_evaluation.json")
    search = _load("search_synthetic_eval.json")
    search_ranker = _load("search_ranker.json")
    tune = _load("lgbm_tune.json")
    negatives = _load("hard_negatives.json")
    crosses = _load("cross_features.json")
    scaled = _load("scaled_supervision.json")
    catboost = _load("catboost_ranker.json")
    listwise = _load("listwise_reranker.json")
    weights = _load("longtail_weights.json")
    baseline_map = baseline.get("development_mean_map_at_12")
    holdout_map = holdout.get("mean_map_at_12")
    selected = (spec.get("architecture") or {}).get("ranker") or {}
    decisions = spec.get("decisions") or []
    lift = None
    if baseline_map and holdout_map:
        lift = (holdout_map - baseline_map) / baseline_map
    wins = [
        item
        for item in (
            f"scaled supervision" if scaled.get("selected") == "scaled" else None,
            f"LightGBM {tune.get('selected')}" if tune.get("selected") not in {None, "baseline"} else None,
            f"hard negatives {negatives.get('selected')}"
            if negatives.get("selected") not in {None, "all_candidates"}
            else None,
            "targeted crosses" if crosses.get("selected") == "targeted_crosses" else None,
            "CatBoost YetiRank" if catboost.get("selected") == "catboost_yetirank" else None,
            f"listwise {listwise.get('selected')}"
            if listwise.get("selected") not in {None, "lambda_only"}
            else None,
            "long-tail weights" if weights.get("selected") == "inv_sqrt_popularity" else None,
        )
        if item
    ]
    fails = [
        item
        for item in (
            "LambdaRank Top-12 truncation",
            "DCN V2 / MLP",
            "image-only DINOv2",
            "text candidate expansion",
            None
            if negatives.get("selected") not in {None, "all_candidates"}
            else "hard-negative downsampling (kept all candidates)",
            None
            if crosses.get("selected") == "targeted_crosses"
            else "explicit cross features",
            None
            if catboost.get("selected") == "catboost_yetirank"
            else "CatBoost YetiRank",
            None
            if listwise.get("selected") not in {None, "lambda_only"}
            else "listwise reranker",
        )
        if item
    ]
    block = [
        "## Best system",
        "",
        f"Status: {spec.get('status', 'not frozen')}",
        "",
        "```text",
        json.dumps(spec.get("architecture", {}), indent=2),
        "```",
        "",
        f"- Development baseline mean MAP@12: {baseline_map}",
        f"- Selected LightGBM trial: {tune.get('selected')} "
        f"({tune.get('selected_mean_map_at_12')})",
        f"- Customer-holdout MAP@12: {holdout_map}",
        f"- Holdout customers: {holdout.get('customers')}",
        f"- Decisions: {', '.join(decisions) if decisions else 'none'}",
        "",
        "Holdout is a customer-holdout evaluation on 2020-09-16..22, not a "
        "pristine unseen week.",
        "",
        "## Search product",
        "",
        f"- Synthetic structured holdout: {search.get('structured_holdout')}",
        f"- Style curated: {search.get('style_curated')}",
        f"- Search ranker: {search_ranker.get('selected', 'not run')}",
        "",
        "## Git checkpoints",
        "",
        "```text",
        _git_log(),
        "```",
        "",
    ]
    results = ROOT / "OVERNIGHT_RESULTS.md"
    text = results.read_text(encoding="utf-8")
    marker = "## Best system"
    if marker in text:
        text = text[: text.index(marker)] + "\n".join(block)
        results.write_text(text, encoding="utf-8")
    print("1. Best architecture:")
    print(json.dumps(spec.get("architecture", {}), indent=2))
    print(f"2. Final MAP@12 (customer holdout): {holdout_map}")
    print(f"3. Baseline MAP@12 (development mean): {baseline_map}")
    print(f"4. Relative lift vs development baseline: {lift}")
    print("5. Top things that improved performance:")
    print("\n".join(f"   - {item}" for item in (wins or ["none beyond the frozen baseline"])))
    print("6. Top things that failed:")
    print("\n".join(f"   - {item}" for item in fails[:6]))
    print(
        "7. Remaining bottleneck: complementary-retriever positives still "
        "look weakly separable from high-ranked implicit negatives."
    )
    print(
        "8. Project complete only if holdout, demo Top-12, search, and docs "
        f"all exist. holdout={bool(holdout_map)} demo="
        f"{(OVERNIGHT_DIR / 'demo_recommendations.json').exists()} "
        f"search={bool(search)}"
    )
    print("9. Open first: OVERNIGHT_RESULTS.md, README.md, artifacts/overnight/")
    print("10. Git:")
    print(_git_log())


if __name__ == "__main__":
    main()
