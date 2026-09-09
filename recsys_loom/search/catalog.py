"""In-memory article catalog used by search and the demo API."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path

from recsys_loom.overnight.protocol import ROOT

ARTICLE_FIELDS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]


@dataclass(slots=True)
class Article:
    article_id: str
    prod_name: str
    product_type_name: str
    product_group_name: str
    colour_group_name: str
    perceived_colour_master_name: str
    department_name: str
    index_name: str
    index_group_name: str
    section_name: str
    garment_group_name: str
    detail_desc: str

    def search_text(self) -> str:
        return " ".join(
            part
            for part in (
                self.prod_name,
                self.product_type_name,
                self.product_group_name,
                self.colour_group_name,
                self.section_name,
                self.garment_group_name,
                self.department_name,
                self.detail_desc,
            )
            if part
        )

    def image_relpath(self) -> str:
        return f"images/{self.article_id[:3]}/{self.article_id}.jpg"

    def to_card(self) -> dict[str, str]:
        return {
            "article_id": self.article_id,
            "name": self.prod_name,
            "product_type": self.product_type_name,
            "color": self.colour_group_name,
            "section": self.section_name,
            "garment_group": self.garment_group_name,
            "department": self.department_name,
            "description": self.detail_desc,
            "image_path": self.image_relpath(),
        }


def load_articles(path: Path | None = None) -> dict[str, Article]:
    configured_path = os.environ.get("CATALOG_PATH")
    csv_path = path or (
        Path(configured_path) if configured_path else ROOT / "articles.csv"
    )
    articles: dict[str, Article] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            article_id = row["article_id"]
            articles[article_id] = Article(
                article_id=article_id,
                prod_name=row.get("prod_name") or "",
                product_type_name=row.get("product_type_name") or "",
                product_group_name=row.get("product_group_name") or "",
                colour_group_name=row.get("colour_group_name") or "",
                perceived_colour_master_name=row.get("perceived_colour_master_name") or "",
                department_name=row.get("department_name") or "",
                index_name=row.get("index_name") or "",
                index_group_name=row.get("index_group_name") or "",
                section_name=row.get("section_name") or "",
                garment_group_name=row.get("garment_group_name") or "",
                detail_desc=row.get("detail_desc") or "",
            )
    return articles


def section_family(section_name: str) -> str:
    lowered = section_name.lower()
    if "men" in lowered and "women" not in lowered:
        return "Menswear"
    if "ladies" in lowered or "women" in lowered or "girl" in lowered:
        return "Womenswear"
    if "baby" in lowered:
        return "Baby"
    if "kid" in lowered or "boy" in lowered or "children" in lowered:
        return "Kids"
    if "divided" in lowered:
        return "Divided"
    return ""
