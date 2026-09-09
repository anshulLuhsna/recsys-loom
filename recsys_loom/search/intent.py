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
ARTICLE_ID_RE = re.compile(r"\b\d{6,}\b")
MAX_SOFT_TERMS = 8
MAX_SOFT_TEXT_LENGTH = 160


@dataclass(frozen=True, slots=True)
class SoftIntent:
    reformulated_query: str | None = None
    style_terms: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "reformulated_query": self.reformulated_query,
            "style_terms": list(self.style_terms),
            "exclusions": list(self.exclusions),
        }


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
    semantic_query: str = ""
    soft_exclusions: list[str] = field(default_factory=list)

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
            "semantic_query": self.semantic_query,
            "soft_exclusions": self.soft_exclusions,
        }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _validate_soft_text(
    value: object,
    field_name: str,
    allowed_tokens: set[str],
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = " ".join(value.strip().split())
    if not cleaned or len(cleaned) > MAX_SOFT_TEXT_LENGTH:
        raise ValueError(f"{field_name} has an invalid length")
    if ARTICLE_ID_RE.search(cleaned):
        raise ValueError(f"{field_name} may not contain catalog IDs")
    tokens = tokenize(cleaned)
    if not tokens or any(token not in allowed_tokens for token in tokens):
        raise ValueError(f"{field_name} must use the request allowlist")
    rebuilt = " ".join(tokens)
    if rebuilt != cleaned.lower():
        raise ValueError(f"{field_name} contains unsupported characters")
    return rebuilt


def validate_soft_intent(
    payload: object,
    allowed_soft_tokens: set[str],
    query_tokens: list[str],
) -> SoftIntent:
    if not isinstance(payload, dict):
        raise ValueError("soft intent must be a JSON object")
    expected = {"reformulated_query", "style_terms", "exclusions"}
    if set(payload) != expected:
        raise ValueError("soft intent has unexpected or missing fields")
    allowed_tokens = allowed_soft_tokens | set(query_tokens)
    deterministic_aliases = (
        set(COLOR_ALIASES) | set(TYPE_ALIASES) | set(SECTION_ALIASES)
    )
    allowed_tokens -= deterministic_aliases - set(query_tokens)
    reformulated_value = payload["reformulated_query"]
    reformulated = None
    if reformulated_value is not None:
        reformulated = _validate_soft_text(
            reformulated_value,
            "reformulated_query",
            allowed_tokens,
        )
    validated_lists: dict[str, tuple[str, ...]] = {}
    for field_name in ("style_terms", "exclusions"):
        values = payload[field_name]
        if not isinstance(values, list) or len(values) > MAX_SOFT_TERMS:
            raise ValueError(f"{field_name} must be a bounded JSON array")
        cleaned_values: list[str] = []
        for value in values:
            cleaned = _validate_soft_text(value, field_name, allowed_tokens)
            if cleaned.lower() not in {item.lower() for item in cleaned_values}:
                cleaned_values.append(cleaned)
        validated_lists[field_name] = tuple(cleaned_values)
    return SoftIntent(
        reformulated_query=reformulated,
        style_terms=validated_lists["style_terms"],
        exclusions=validated_lists["exclusions"],
    )


def merge_soft_intent(intent: ParsedIntent, soft: SoftIntent) -> ParsedIntent:
    original_hard_constraints = dict(intent.hard_constraints)
    style_terms = list(intent.style_terms)
    known_styles = {term.lower() for term in style_terms}
    for term in soft.style_terms:
        if term.lower() not in known_styles:
            style_terms.append(term)
            known_styles.add(term.lower())
    enriched_parts = [intent.raw_query]
    if soft.reformulated_query:
        enriched_parts.append(soft.reformulated_query)
    enriched_parts.extend(soft.style_terms)
    soft_preferences = dict(intent.soft_preferences)
    if soft.reformulated_query:
        soft_preferences["llm_reformulation"] = soft.reformulated_query
    if soft.style_terms:
        soft_preferences["llm_style_terms"] = ", ".join(soft.style_terms)
    if soft.exclusions:
        soft_preferences["llm_exclusions"] = ", ".join(soft.exclusions)
    merged = ParsedIntent(
        raw_query=intent.raw_query,
        tokens=list(intent.tokens),
        color=intent.color,
        product_type=intent.product_type,
        section=intent.section,
        style_terms=style_terms,
        hard_constraints=dict(intent.hard_constraints),
        soft_preferences=soft_preferences,
        free_text=intent.free_text,
        semantic_query=" ".join(part for part in enriched_parts if part),
        soft_exclusions=list(soft.exclusions),
    )
    if merged.hard_constraints != original_hard_constraints:
        raise RuntimeError("soft intent attempted to change deterministic constraints")
    return merged


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
        semantic_query=query.strip(),
    )
