"""Deterministic query-intent parser over the H&M taxonomy."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

COLOR_ALIASES = {
    "black": "Black",
    "white": "White",
    "blue": "Blue",
    "navy": "Dark Blue",
    "red": "Red",
    "green": "Green",
    "pink": "Pink",
    "beige": "Beige",
    "grey": "Grey",
    "gray": "Grey",
    "brown": "Brown",
    "yellow": "Yellow",
    "orange": "Orange",
    "purple": "Purple",
    "gold": "Gold",
    "silver": "Silver",
    "khaki": "Khaki green",
    "cream": "White",
    "ivory": "White",
}

TYPE_ALIASES = {
    "hoodie": "Hoodie",
    "hoodies": "Hoodie",
    "sweatshirt": "Sweater",
    "sweater": "Sweater",
    "jumper": "Sweater",
    "dress": "Dress",
    "dresses": "Dress",
    "trouser": "Trousers",
    "trousers": "Trousers",
    "pants": "Trousers",
    "jean": "Trousers",
    "jeans": "Trousers",
    "shirt": "Shirt",
    "shirts": "Shirt",
    "blouse": "Blouse",
    "top": "Top",
    "tops": "Top",
    "tee": "T-shirt",
    "t-shirt": "T-shirt",
    "tshirt": "T-shirt",
    "jacket": "Jacket",
    "jackets": "Jacket",
    "coat": "Coat",
    "cardigan": "Cardigan",
    "skirt": "Skirt",
    "shorts": "Shorts",
    "legging": "Leggings/Tights",
    "leggings": "Leggings/Tights",
    "tights": "Leggings/Tights",
    "sock": "Socks",
    "socks": "Socks",
    "shoe": "Shoes",
    "shoes": "Shoes",
    "sneaker": "Sneakers",
    "sneakers": "Sneakers",
    "boot": "Boots",
    "boots": "Boots",
    "bag": "Bag",
    "hat": "Hat/beanie",
    "beanie": "Hat/beanie",
    "scarf": "Scarf",
    "swim": "Swimwear",
    "swimsuit": "Swimwear",
}

SECTION_ALIASES = {
    "women": "Womenswear",
    "woman": "Womenswear",
    "womens": "Womenswear",
    "ladies": "Womenswear",
    "men": "Menswear",
    "man": "Menswear",
    "mens": "Menswear",
    "male": "Menswear",
    "kids": "Kids",
    "kid": "Kids",
    "children": "Kids",
    "baby": "Baby",
    "divided": "Divided",
}

STYLE_TERMS = {
    "oversized",
    "linen",
    "summer",
    "winter",
    "minimal",
    "casual",
    "formal",
    "office",
    "sport",
    "sportswear",
    "streetwear",
    "cotton",
    "denim",
    "knit",
    "printed",
    "plain",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


@dataclass(slots=True)
class ParsedIntent:
    raw_query: str
    tokens: list[str]
    color: str | None = None
    product_type: str | None = None
    section: str | None = None
    style_terms: list[str] = field(default_factory=list)
    hard_constraints: dict[str, str] = field(default_factory=dict)
    soft_preferences: dict[str, str] = field(default_factory=dict)
    free_text: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_query": self.raw_query,
            "tokens": self.tokens,
            "color": self.color,
            "product_type": self.product_type,
            "section": self.section,
            "style_terms": self.style_terms,
            "hard_constraints": self.hard_constraints,
            "soft_preferences": self.soft_preferences,
            "free_text": self.free_text,
        }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def parse_query(query: str) -> ParsedIntent:
    tokens = tokenize(query)
    color = next((COLOR_ALIASES[token] for token in tokens if token in COLOR_ALIASES), None)
    product_type = next(
        (TYPE_ALIASES[token] for token in tokens if token in TYPE_ALIASES),
        None,
    )
    section = next(
        (SECTION_ALIASES[token] for token in tokens if token in SECTION_ALIASES),
        None,
    )
    style_terms = [token for token in tokens if token in STYLE_TERMS]
    hard: dict[str, str] = {}
    soft: dict[str, str] = {}
    if product_type:
        hard["product_type"] = product_type
    if section:
        soft["section"] = section
        if section in {"Womenswear", "Menswear", "Kids", "Baby"}:
            hard["section_family"] = section
    if color and product_type:
        hard["color"] = color
    elif color:
        soft["color"] = color
    return ParsedIntent(
        raw_query=query.strip(),
        tokens=tokens,
        color=color,
        product_type=product_type,
        section=section,
        style_terms=style_terms,
        hard_constraints=hard,
        soft_preferences=soft,
        free_text=query.strip(),
    )
