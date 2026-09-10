#!/usr/bin/env python3
"""Prepare and score blinded human judgments for GenRec-inspired search."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.overnight.protocol import ROOT as PROJECT_ROOT
from recsys_loom.search.catalog import Article
from recsys_loom.search.intent import parse_query
from recsys_loom.search.llm import SearchLLMConfig
from recsys_loom.search.pipeline import SearchEngine
from recsys_loom.search.service import build_search_engine
from recsys_loom.search.structured import passes_hard_filters

QUERIES = [
    "black oversized hoodie for a relaxed weekend",
    "women's linen shirt for a hot office",
    "minimal black dress for an evening event",
    "men's blue shirt that feels polished but casual",
    "warm cream cardigan with a soft textured look",
    "kids red jacket for rainy school days",
    "striped oversized shirt for a breezy summer holiday",
    "comfortable black trousers for work",
]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "llm_search"
RANKINGS_PATH = OUTPUT_DIR / "llm_search_frozen_rankings.json"
JUDGMENTS_PATH = OUTPUT_DIR / "llm_search_blinded_judgments.json"
REPORT_PATH = OUTPUT_DIR / "llm_search_human_eval.json"
SHEETS_DIR = OUTPUT_DIR / "judging_sheets"
VARIANT_NAMES = (
    "bm25_structured_clip",
    "hybrid_minilm",
    "llm_intent_only",
    "llm_rerank_only",
    "full_llm",
)


def _require_configuration() -> None:
    config = SearchLLMConfig.from_env()
    if not config.configured:
        raise RuntimeError(
            "LLM search credentials are required. Set SEARCH_LLM=1 and "
            "SEARCH_LLM_API_KEY (or GROQ_API_KEY), then configure the optional "
            "base URL/model if the Groq defaults are not appropriate."
        )


def _build_variants() -> dict[str, SearchEngine]:
    full = build_search_engine(
        load_semantic=True,
        load_visual=True,
        load_llm=True,
    )

    def variant(use_intent: bool, use_rerank: bool) -> SearchEngine:
        return SearchEngine(
            full.articles,
            full.lexical,
            full.semantic,
            full.structured,
            full.visual,
            full.llm,
            use_intent,
            use_rerank,
        )

    return {
        "bm25_structured_clip": SearchEngine(
            full.articles,
            full.lexical,
            None,
            full.structured,
            full.visual,
            full.llm,
            False,
            False,
        ),
        "hybrid_minilm": variant(False, False),
        "llm_intent_only": variant(True, False),
        "llm_rerank_only": variant(False, True),
        "full_llm": variant(True, True),
    }


def _blind_key(query: str, article_id: str) -> str:
    return hashlib.sha256(f"{query}:{article_id}".encode("utf-8")).hexdigest()


def _candidate_code(query: str, article_id: str) -> str:
    digest = hashlib.sha256(
        f"candidate:{query}:{article_id}".encode("utf-8")
    ).hexdigest()
    return digest[:10].upper()


def _image_path(article_id: str) -> Path:
    return PROJECT_ROOT / "images" / article_id[:3] / f"{article_id}.jpg"


def _make_judging_sheet(
    query: str,
    candidate_ids: list[str],
    articles: dict[str, Article],
    output_path: Path,
) -> None:
    columns = 4
    tile_width = 260
    tile_height = 310
    header_height = 55
    rows = math.ceil(len(candidate_ids) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 15), f"Query: {query}", fill="black")
    for position, article_id in enumerate(candidate_ids):
        article = articles[article_id]
        x = (position % columns) * tile_width
        y = header_height + (position // columns) * tile_height
        image_path = _image_path(article_id)
        if image_path.exists():
            with Image.open(image_path) as image:
                fitted = ImageOps.contain(
                    image.convert("RGB"),
                    (240, 225),
                    Image.Resampling.LANCZOS,
                )
            canvas.paste(fitted, (x + 10, y + 5))
        else:
            draw.rectangle((x + 10, y + 5, x + 250, y + 230), fill="#eeeeee")
            draw.text((x + 70, y + 110), "image unavailable", fill="#555555")
        draw.text(
            (x + 10, y + 238),
            f"Code: {_candidate_code(query, article_id)}",
            fill="black",
        )
        draw.text((x + 10, y + 256), article.prod_name[:36], fill="black")
        draw.text(
            (x + 10, y + 274),
            (
                f"{article.product_type_name[:20]} | "
                f"{article.colour_group_name[:16]}"
            ),
            fill="black",
        )
    canvas.save(output_path, quality=90)


def prepare() -> None:
    if RANKINGS_PATH.exists() or JUDGMENTS_PATH.exists():
        raise FileExistsError(
            "Frozen LLM-search comparison already exists; move it explicitly "
            "before preparing a new candidate set."
        )
    variants = _build_variants()
    articles = next(iter(variants.values())).articles
    frozen_queries = []
    blinded_queries = []
    sheet_jobs: list[tuple[str, list[str], Path]] = []
    for query_index, query in enumerate(QUERIES, start=1):
        rankings: dict[str, list[str]] = {}
        diagnostics: dict[str, object] = {}
        union: set[str] = set()
        for name, engine in variants.items():
            payload = engine.search(query, limit=20)
            rows = payload["results"]
            variant_diagnostics = payload["diagnostics"]
            diagnostics[name] = variant_diagnostics
            expected_stages = []
            if name in {"llm_intent_only", "full_llm"}:
                expected_stages.append("intent_enrichment")
            if name in {"llm_rerank_only", "full_llm"}:
                expected_stages.append("candidate_rerank")
            for stage in expected_stages:
                stage_diagnostics = variant_diagnostics[stage]
                if not stage_diagnostics["applied"]:
                    raise RuntimeError(
                        f"{name} could not apply {stage} for {query!r}: "
                        f"{stage_diagnostics['fallback_reason']}"
                    )
            article_ids = [str(row["article_id"]) for row in rows]
            rankings[name] = article_ids
            union.update(article_ids)
        blinded_ids = sorted(union, key=lambda item: _blind_key(query, item))
        frozen_queries.append(
            {
                "query": query,
                "rankings": rankings,
                "diagnostics": diagnostics,
            }
        )
        sheet_path = SHEETS_DIR / f"query_{query_index:02d}.jpg"
        sheet_jobs.append((query, blinded_ids, sheet_path))
        blinded_queries.append(
            {
                "query": query,
                "instruction": (
                    "Assign relevance from 0 (irrelevant) to 3 (excellent match) "
                    "using the blinded image sheet and minimal product text. Do not "
                    "inspect the frozen rankings file while judging."
                ),
                "judging_sheet": str(sheet_path),
                "candidates": [
                    {
                        "candidate_code": _candidate_code(query, article_id),
                        "article_id": article_id,
                        "relevance": None,
                    }
                    for article_id in blinded_ids
                ],
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    for query, candidate_ids, sheet_path in sheet_jobs:
        _make_judging_sheet(query, candidate_ids, articles, sheet_path)
    RANKINGS_PATH.write_text(
        json.dumps(
            {
                "label": "frozen GenRec-inspired search variant rankings",
                "not": "an automated relevance evaluation",
                "variants": list(VARIANT_NAMES),
                "queries": frozen_queries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    JUDGMENTS_PATH.write_text(
        json.dumps(
            {
                "label": "blinded human catalog-search judgments",
                "not": "LLM judgments or token-overlap labels",
                "queries": blinded_queries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rankings": str(RANKINGS_PATH),
                "judgments": str(JUDGMENTS_PATH),
                "next": "fill every relevance field, then rerun without --prepare",
            },
            indent=2,
        )
    )


def _ndcg(article_ids: list[str], relevance: dict[str, int], k: int = 10) -> float:
    gains = [relevance.get(article_id, 0) for article_id in article_ids[:k]]
    dcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _index_query_rows(
    payload: object,
    label: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError(f"{label} must contain a queries array")
    indexed: dict[str, dict[str, object]] = {}
    for row in payload["queries"]:
        if not isinstance(row, dict) or not isinstance(row.get("query"), str):
            raise ValueError(f"{label} contains an invalid query row")
        query = row["query"]
        if query in indexed:
            raise ValueError(f"{label} contains duplicate query {query!r}")
        indexed[query] = row
    return indexed


def _validate_frozen_rankings(
    query: str,
    query_row: dict[str, object],
) -> tuple[dict[str, list[str]], set[str]]:
    rankings = query_row.get("rankings")
    if not isinstance(rankings, dict) or set(rankings) != set(VARIANT_NAMES):
        raise ValueError(f"Frozen rankings have invalid variants for {query!r}")
    validated: dict[str, list[str]] = {}
    candidate_union: set[str] = set()
    for name in VARIANT_NAMES:
        values = rankings[name]
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"Frozen ranking {name} is invalid for {query!r}")
        article_ids = list(values)
        if len(article_ids) != len(set(article_ids)):
            raise ValueError(f"Frozen ranking {name} has duplicate IDs for {query!r}")
        validated[name] = article_ids
        candidate_union.update(article_ids)
    return validated, candidate_union


def _validate_judgments(
    query: str,
    query_row: dict[str, object],
    expected_ids: set[str],
) -> dict[str, int]:
    candidates = query_row.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Judgments have no candidate array for {query!r}")
    relevance: dict[str, int] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(
            candidate.get("article_id"),
            str,
        ):
            raise ValueError(f"Judgments contain an invalid candidate for {query!r}")
        article_id = candidate["article_id"]
        if article_id in relevance:
            raise ValueError(f"Judgments contain duplicate ID {article_id} for {query!r}")
        if candidate.get("candidate_code") != _candidate_code(query, article_id):
            raise ValueError(
                f"Judgments contain an invalid candidate code for {query!r}"
            )
        value = candidate.get("relevance")
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
            raise ValueError(
                f"Every relevance must be an integer from 0 to 3; incomplete: "
                f"{query} / {article_id}"
            )
        relevance[article_id] = value
    judged_ids = set(relevance)
    if judged_ids != expected_ids:
        missing = sorted(expected_ids - judged_ids)
        extra = sorted(judged_ids - expected_ids)
        raise ValueError(
            f"Judgment candidate mismatch for {query!r}; "
            f"missing={missing}, extra={extra}"
        )
    return relevance


def evaluate() -> None:
    if not RANKINGS_PATH.exists() or not JUDGMENTS_PATH.exists():
        raise FileNotFoundError(
            "Missing frozen comparison files; run with --prepare first."
        )
    rankings_payload = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
    judgments_payload = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
    frozen_rows = _index_query_rows(rankings_payload, "Frozen rankings")
    judgment_rows = _index_query_rows(judgments_payload, "Judgments")
    frozen_queries = set(frozen_rows)
    if frozen_queries != set(QUERIES):
        raise ValueError("Frozen rankings do not match the declared query set")
    if set(judgment_rows) != frozen_queries:
        missing = sorted(frozen_queries - set(judgment_rows))
        extra = sorted(set(judgment_rows) - frozen_queries)
        raise ValueError(
            f"Judgment query mismatch; missing={missing}, extra={extra}"
        )
    validated_rankings: dict[str, dict[str, list[str]]] = {}
    judgments: dict[str, dict[str, int]] = {}
    for query, frozen_row in frozen_rows.items():
        rankings, candidate_union = _validate_frozen_rankings(query, frozen_row)
        validated_rankings[query] = rankings
        judgments[query] = _validate_judgments(
            query,
            judgment_rows[query],
            candidate_union,
        )
    articles = build_search_engine(
        load_semantic=False,
        load_visual=False,
        load_llm=False,
    ).articles
    per_variant: dict[str, list[float]] = {name: [] for name in VARIANT_NAMES}
    hard_filter_violations = {name: 0 for name in VARIANT_NAMES}
    for query in QUERIES:
        intent = parse_query(query)
        for name in VARIANT_NAMES:
            article_ids = validated_rankings[query][name]
            per_variant[name].append(_ndcg(article_ids, judgments[query]))
            hard_filter_violations[name] += sum(
                article_id not in articles
                or not passes_hard_filters(articles[article_id], intent)
                for article_id in article_ids
            )
    report = {
        "label": "human-judged GenRec-inspired search comparison",
        "not": "Netflix GenRec reproduction, LLM judging, or production relevance",
        "query_count": len(validated_rankings),
        "variants": {
            name: {
                "mean_ndcg_at_10": sum(scores) / len(scores) if scores else 0.0,
                "hard_filter_violations": hard_filter_violations[name],
            }
            for name, scores in per_variant.items()
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    if "--prepare" in sys.argv:
        _require_configuration()
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
