#!/usr/bin/env python3
"""Compare frozen lexical and MiniLM search benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERNIGHT_DIR = ROOT / "artifacts" / "overnight"
MIN_NDCG_LIFT = 0.01


def main() -> None:
    lexical_path = OVERNIGHT_DIR / "search_synthetic_eval_lexical.json"
    hybrid_path = OVERNIGHT_DIR / "search_synthetic_eval_hybrid.json"
    lexical = json.loads(lexical_path.read_text(encoding="utf-8"))
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))

    lexical_style = lexical["style_curated"]
    hybrid_style = hybrid["style_curated"]
    exact_preserved = (
        hybrid["structured_holdout"]["ndcg_at_10"]
        >= lexical["structured_holdout"]["ndcg_at_10"]
    )
    ndcg_delta = (
        hybrid_style["ndcg_at_10"] - lexical_style["ndcg_at_10"]
    )
    keep_semantic = exact_preserved and ndcg_delta >= MIN_NDCG_LIFT
    selected = "bm25+semantic+structured" if keep_semantic else "bm25+structured"
    report = {
        "hypothesis": (
            "MiniLM semantic retrieval improves top-ranked style relevance while "
            "preserving exact color and product-type intent."
        ),
        "benchmark_warning": (
            "Labels are synthetic metadata/token judgments, not user query logs."
        ),
        "acceptance_rule": {
            "minimum_style_ndcg_at_10_lift": MIN_NDCG_LIFT,
            "must_preserve_structured_holdout_ndcg_at_10": True,
        },
        "lexical_structured": {
            "structured_holdout": lexical["structured_holdout"],
            "style_curated": lexical_style,
        },
        "with_minilm": {
            "structured_holdout": hybrid["structured_holdout"],
            "style_curated": hybrid_style,
        },
        "deltas": {
            metric: hybrid_style[metric] - lexical_style[metric]
            for metric in (
                "precision_at_10",
                "recall_at_50",
                "ndcg_at_10",
                "mrr",
            )
        },
        "selected": selected,
        "decision": (
            "keep MiniLM enabled"
            if keep_semantic
            else "keep MiniLM optional; default to lexical plus structured"
        ),
    }
    output_path = OVERNIGHT_DIR / "search_semantic_ablation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "deltas": report["deltas"]}, indent=2))


if __name__ == "__main__":
    main()
