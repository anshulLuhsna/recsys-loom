#!/usr/bin/env python3
"""Prepare blinded visual judgments and evaluate CLIP search fusion."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.search.benchmark import generate_structured_queries
from recsys_loom.search.catalog import Article, load_articles
from recsys_loom.search.intent import parse_query
from recsys_loom.search.lexical import BM25Index
from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.structured import StructuredIndex, passes_hard_filters
from recsys_loom.search.visual import VISUAL_DIR, VisualIndex

VISUAL_QUERIES = [
    "floral summer dress",
    "chunky cream cardigan",
    "distressed blue jeans",
    "minimal black slip dress",
    "striped oversized shirt",
]
JUDGMENTS_PATH = VISUAL_DIR / "visual_judgments.json"


def build_engines() -> tuple[
    dict[str, Article],
    SearchEngine,
    SearchEngine,
]:
    articles = load_articles()
    lexical = BM25Index()
    lexical.build(articles)
    structured = StructuredIndex(articles)
    baseline = SearchEngine(articles, lexical, None, structured)
    with_visual = SearchEngine(
        articles,
        lexical,
        None,
        structured,
        VisualIndex.load(load_encoder=True),
    )
    return articles, baseline, with_visual


def image_path(article_id: str) -> Path:
    return ROOT / "images" / article_id[:3] / f"{article_id}.jpg"


def make_sheet(
    query: str,
    candidates: list[dict[str, object]],
    articles: dict[str, Article],
    output_path: Path,
) -> None:
    columns = 4
    tile_width = 240
    tile_height = 300
    header_height = 50
    rows = math.ceil(len(candidates) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"Query: {query}", fill="black")
    for index, candidate in enumerate(candidates):
        article_id = str(candidate["article_id"])
        x = (index % columns) * tile_width
        y = header_height + (index // columns) * tile_height
        path = image_path(article_id)
        if path.exists():
            with Image.open(path) as image:
                fitted = ImageOps.contain(image.convert("RGB"), (220, 225))
            canvas.paste(fitted, (x + 10, y + 5))
        article = articles[article_id]
        draw.text((x + 10, y + 235), f"{index + 1}. {article_id}", fill="black")
        draw.text(
            (x + 10, y + 251),
            f"{article.product_type_name[:22]} | {article.colour_group_name[:16]}",
            fill="black",
        )
    canvas.save(output_path, quality=90)


def prepare_candidates(
    articles: dict[str, Article],
    baseline: SearchEngine,
    with_visual: SearchEngine,
) -> None:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    sheets_dir = VISUAL_DIR / "judging_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": (
            "Judge whether each image matches the query without using catalog "
            "metadata or retrieval score. Candidate order is a stable union."
        ),
        "queries": [],
    }
    for query_index, query in enumerate(VISUAL_QUERIES, start=1):
        baseline_rows = baseline.search(query, limit=10)["results"]
        visual_rows = with_visual.search(query, limit=10)["results"]
        baseline_rank = {
            row["article_id"]: rank for rank, row in enumerate(baseline_rows, start=1)
        }
        visual_rank = {
            row["article_id"]: rank for rank, row in enumerate(visual_rows, start=1)
        }
        article_ids = sorted(
            set(baseline_rank) | set(visual_rank),
            key=lambda article_id: (
                min(
                    baseline_rank.get(article_id, 10_000),
                    visual_rank.get(article_id, 10_000),
                ),
                article_id,
            ),
        )
        candidates = [
            {
                "article_id": article_id,
                "channels": [
                    name
                    for name, ranking in (
                        ("baseline", baseline_rank),
                        ("visual", visual_rank),
                    )
                    if article_id in ranking
                ],
                "baseline_rank": baseline_rank.get(article_id),
                "visual_rank": visual_rank.get(article_id),
            }
            for article_id in article_ids
        ]
        sheet_path = sheets_dir / f"{query_index}_{query.replace(' ', '_')}.jpg"
        make_sheet(query, candidates, articles, sheet_path)
        report["queries"].append(
            {
                "query": query,
                "candidates": candidates,
                "judging_sheet": str(sheet_path),
            }
        )
    candidates_path = VISUAL_DIR / "visual_judging_candidates.json"
    candidates_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidates": str(candidates_path),
                "sheets": str(sheets_dir),
                "next": f"write image-only relevance labels to {JUDGMENTS_PATH}",
            },
            indent=2,
        )
    )


def metric_row(
    rows: list[dict[str, object]],
    relevant: set[str],
) -> dict[str, float]:
    article_ids = [str(row["article_id"]) for row in rows[:10]]
    gains = [1.0 if article_id in relevant else 0.0 for article_id in article_ids]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_count = min(len(relevant), 10)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    first = next((rank for rank, gain in enumerate(gains, start=1) if gain), None)
    return {
        "precision_at_10": sum(gains) / 10.0,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "mrr": 1.0 / first if first else 0.0,
    }


def evaluate(
    articles: dict[str, Article],
    baseline: SearchEngine,
    with_visual: SearchEngine,
) -> None:
    judgments_payload = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
    judgments = {
        row["query"]: set(row["relevant_article_ids"])
        for row in judgments_payload["queries"]
    }
    per_query = []
    for query in VISUAL_QUERIES:
        relevant = judgments[query]
        baseline_metrics = metric_row(
            baseline.search(query, limit=10)["results"],
            relevant,
        )
        visual_metrics = metric_row(
            with_visual.search(query, limit=10)["results"],
            relevant,
        )
        per_query.append(
            {
                "query": query,
                "relevant_candidates": len(relevant),
                "baseline": baseline_metrics,
                "with_visual": visual_metrics,
                "ndcg_delta": (
                    visual_metrics["ndcg_at_10"]
                    - baseline_metrics["ndcg_at_10"]
                ),
            }
        )

    exact_violations = 0
    structured_queries = generate_structured_queries(articles)
    for query in structured_queries:
        intent = parse_query(query)
        rows = with_visual.search(query, limit=10)["results"]
        exact_violations += sum(
            not passes_hard_filters(articles[str(row["article_id"])], intent)
            for row in rows
        )
    baseline_ndcg = sum(row["baseline"]["ndcg_at_10"] for row in per_query) / len(
        per_query
    )
    visual_ndcg = sum(row["with_visual"]["ndcg_at_10"] for row in per_query) / len(
        per_query
    )
    improved_queries = sum(row["ndcg_delta"] > 0 for row in per_query)
    keep_visual = (
        visual_ndcg >= baseline_ndcg + 0.05
        and improved_queries >= 3
        and exact_violations == 0
    )
    report = {
        "label": "manual image-only visual-search evaluation",
        "not": "production search quality or purchase prediction",
        "model": "openai/clip-vit-base-patch32",
        "acceptance_rule": {
            "minimum_mean_ndcg_at_10_lift": 0.05,
            "minimum_improved_queries": 3,
            "maximum_exact_constraint_violations": 0,
        },
        "per_query": per_query,
        "summary": {
            "baseline_mean_ndcg_at_10": baseline_ndcg,
            "with_visual_mean_ndcg_at_10": visual_ndcg,
            "ndcg_at_10_delta": visual_ndcg - baseline_ndcg,
            "improved_queries": improved_queries,
            "exact_constraint_violations": exact_violations,
        },
        "selected": "with_visual" if keep_visual else "baseline_without_visual",
    }
    output_path = VISUAL_DIR / "visual_search_eval.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"] | {"selected": report["selected"]}, indent=2))


def main() -> None:
    articles, baseline, with_visual = build_engines()
    if "--prepare" in sys.argv:
        prepare_candidates(articles, baseline, with_visual)
        return
    if not JUDGMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {JUDGMENTS_PATH}; run with --prepare and label the sheets first"
        )
    evaluate(articles, baseline, with_visual)


if __name__ == "__main__":
    main()
