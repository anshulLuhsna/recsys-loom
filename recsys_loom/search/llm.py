"""Optional GenRec-inspired, catalog-grounded search intent and reranking."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import os
from threading import Lock
import time
from typing import Callable

import httpx

from recsys_loom.search.catalog import Article
from recsys_loom.search.intent import (
    ParsedIntent,
    SoftIntent,
    STYLE_TERMS,
    validate_soft_intent,
)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RERANK_LIMIT = 15
MAX_RERANK_LIMIT = 50
DEFAULT_MAX_CALLS_PER_MINUTE = 30
MAX_CALLS_PER_MINUTE = 600
DEFAULT_CACHE_SIZE = 128
FASHION_VOCABULARY_HINTS = {
    "classic",
    "comfortable",
    "cropped",
    "elegant",
    "everyday",
    "fitted",
    "floral",
    "loose",
    "modern",
    "party",
    "relaxed",
    "ribbed",
    "romantic",
    "slim",
    "soft",
    "striped",
    "tailored",
    "textured",
    "warm",
    "workwear",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_float(raw: str | None, default: float) -> float:
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        return default
    return min(max(value, 0.2), 30.0)


def _bounded_int(raw: str | None, default: int, maximum: int) -> int:
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        return default
    return min(max(value, 1), maximum)


@dataclass(frozen=True, slots=True)
class SearchLLMConfig:
    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    rerank_limit: int
    max_calls_per_minute: int
    cache_size: int = DEFAULT_CACHE_SIZE

    @classmethod
    def from_env(cls, enabled: bool | None = None) -> "SearchLLMConfig":
        return cls(
            enabled=_env_flag("SEARCH_LLM") if enabled is None else enabled,
            base_url=(
                os.environ.get("SEARCH_LLM_BASE_URL", DEFAULT_BASE_URL).strip()
                or DEFAULT_BASE_URL
            ),
            api_key=(
                os.environ.get("SEARCH_LLM_API_KEY")
                or os.environ.get("GROQ_API_KEY")
                or ""
            ).strip(),
            model=(
                os.environ.get("SEARCH_LLM_MODEL", DEFAULT_MODEL).strip()
                or DEFAULT_MODEL
            ),
            timeout_seconds=_bounded_float(
                os.environ.get("SEARCH_LLM_TIMEOUT_SECONDS"),
                DEFAULT_TIMEOUT_SECONDS,
            ),
            rerank_limit=_bounded_int(
                os.environ.get("SEARCH_LLM_RERANK_LIMIT"),
                DEFAULT_RERANK_LIMIT,
                MAX_RERANK_LIMIT,
            ),
            max_calls_per_minute=_bounded_int(
                os.environ.get("SEARCH_LLM_MAX_CALLS_PER_MINUTE"),
                DEFAULT_MAX_CALLS_PER_MINUTE,
                MAX_CALLS_PER_MINUTE,
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.base_url and self.api_key and self.model)

    def diagnostics(self) -> dict[str, object]:
        if not self.enabled:
            status = "disabled"
        elif not self.configured:
            status = "missing_configuration"
        else:
            status = "configured"
        return {
            "approach": "GenRec-inspired catalog-grounded search",
            "status": status,
            "enabled": self.enabled,
            "configured": self.configured,
            "model": self.model if self.enabled else None,
            "timeout_seconds": self.timeout_seconds,
            "rerank_limit": self.rerank_limit,
            "max_calls_per_minute": self.max_calls_per_minute,
            "customer_history_in_prompt": False,
        }


class BoundedCache:
    def __init__(self, max_size: int):
        self.max_size = max(1, max_size)
        self._values: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> dict[str, object] | None:
        with self._lock:
            value = self._values.get(key)
            if value is None:
                return None
            self._values.move_to_end(key)
            return dict(value)

    def put(self, key: str, value: dict[str, object]) -> None:
        with self._lock:
            self._values[key] = dict(value)
            self._values.move_to_end(key)
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)


class ProviderRateLimitExceeded(RuntimeError):
    """Raised before an external call when the process quota is exhausted."""


class ProcessCallRateLimiter:
    def __init__(self) -> None:
        self._call_times: deque[float] = deque()
        self._lock = Lock()

    def acquire(self, max_calls_per_minute: int) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            while self._call_times and self._call_times[0] <= cutoff:
                self._call_times.popleft()
            if len(self._call_times) >= max_calls_per_minute:
                return False
            self._call_times.append(now)
            return True


PROCESS_CALL_RATE_LIMITER = ProcessCallRateLimiter()


class CatalogSearchLLM:
    def __init__(
        self,
        config: SearchLLMConfig,
        catalog_vocabulary: set[str],
        post: Callable[..., httpx.Response] | None = None,
    ):
        self.config = config
        self.catalog_vocabulary = {token.lower() for token in catalog_vocabulary}
        self.cache = BoundedCache(config.cache_size)
        self._post = post or httpx.post

    @property
    def available(self) -> bool:
        return self.config.configured

    def diagnostics(self) -> dict[str, object]:
        return self.config.diagnostics()

    def enrich_intent(
        self,
        intent: ParsedIntent,
    ) -> tuple[SoftIntent | None, dict[str, object]]:
        if not self.available:
            return None, self._unavailable_stage()
        started = time.perf_counter()
        allowed_terms = sorted(
            (STYLE_TERMS | FASHION_VOCABULARY_HINTS)
            & self.catalog_vocabulary
        )
        allowed_soft_tokens = set(allowed_terms)
        prompt = (
            "You add only clearly supported soft catalog-search intent. "
            "Do not fill fields merely because values appear in the allowlist. "
            "If the query already expresses only a clear color, product type, or "
            "section, return null and empty arrays. Deterministic constraints are "
            f"locked and cannot be changed: {json.dumps(intent.hard_constraints)}. "
            f"Raw query: {json.dumps(intent.raw_query)}. "
            "Do not emit article IDs, customer history, colors, product types, or "
            "sections that are not already in the raw query. Use only words already "
            "present in the raw query or from this exact catalog allowlist: "
            f"{json.dumps(allowed_terms)}. "
            "Return exactly one JSON object with keys reformulated_query, "
            "style_terms, exclusions. reformulated_query is a string or null; "
            "style_terms and exclusions are arrays of at most 8 short strings. "
            "Use [] when no defensible addition exists. No markdown."
        )

        def validate_enrichment(payload: object) -> dict[str, object]:
            return validate_soft_intent(
                payload,
                allowed_soft_tokens,
                intent.tokens,
            ).to_dict()

        try:
            payload, cache_hit = self._request_validated_json(
                "intent",
                prompt,
                validate_enrichment,
            )
            soft_intent = validate_soft_intent(
                payload,
                allowed_soft_tokens,
                intent.tokens,
            )
        except httpx.TimeoutException:
            return None, self._failed_stage(started, "timeout")
        except ProviderRateLimitExceeded:
            return None, self._failed_stage(started, "rate_limited")
        except httpx.HTTPStatusError as error:
            return None, self._failed_stage(
                started,
                self._http_failure_reason(error),
            )
        except httpx.HTTPError:
            return None, self._failed_stage(started, "provider_connection_error")
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None, self._failed_stage(started, "invalid_response")
        return soft_intent, {
            "attempted": True,
            "applied": True,
            "cache_hit": cache_hit,
            "fallback_reason": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "soft_intent": soft_intent.to_dict(),
        }

    def rerank(
        self,
        query: str,
        intent: ParsedIntent,
        candidate_ids: list[str],
        articles: dict[str, Article],
    ) -> tuple[list[str] | None, dict[str, object]]:
        bounded_ids = candidate_ids[: self.config.rerank_limit]
        if not self.available:
            return None, self._unavailable_stage(candidate_count=len(bounded_ids))
        if not bounded_ids:
            return None, {
                "attempted": False,
                "applied": False,
                "cache_hit": False,
                "fallback_reason": "no_candidates",
                "latency_ms": 0.0,
                "candidate_count": 0,
                "returned_count": 0,
                "invalid_id_count": 0,
            }
        started = time.perf_counter()
        candidates = [
            {
                "article_id": article_id,
                "name": articles[article_id].prod_name,
                "product_type": articles[article_id].product_type_name,
                "color": articles[article_id].colour_group_name,
                "section": articles[article_id].section_name,
                "description": articles[article_id].detail_desc[:240],
            }
            for article_id in bounded_ids
            if article_id in articles
        ]
        prompt = (
            "Rerank only the supplied catalog candidates for the raw query. "
            f"Raw query: {json.dumps(query)}. "
            f"Locked deterministic constraints: {json.dumps(intent.hard_constraints)}. "
            f"Soft style terms: {json.dumps(intent.style_terms)}. "
            f"Soft exclusions: {json.dumps(intent.soft_exclusions)}. "
            f"Candidates: {json.dumps(candidates, ensure_ascii=True)}. "
            "Return exactly one JSON object with key article_ids. Its value is an "
            "ordered array containing only article_id strings copied from the "
            "candidate list. You may omit uncertain candidates. Never invent an ID, "
            "change a constraint, or use customer history. No scores or markdown."
        )

        def validate_rerank(payload: object) -> dict[str, object]:
            if not isinstance(payload, dict) or set(payload) != {"article_ids"}:
                raise ValueError("rerank response has unexpected fields")
            returned = payload["article_ids"]
            if not isinstance(returned, list) or len(returned) > len(bounded_ids):
                raise ValueError("article_ids must be a bounded array")
            allowed = set(bounded_ids)
            validated_ids: list[str] = []
            invalid_ids = 0
            seen: set[str] = set()
            for value in returned:
                if not isinstance(value, str) or value not in allowed:
                    invalid_ids += 1
                    continue
                if value not in seen:
                    validated_ids.append(value)
                    seen.add(value)
            return {
                "article_ids": validated_ids,
                "invalid_id_count": invalid_ids,
            }

        try:
            payload, cache_hit = self._request_validated_json(
                "rerank",
                prompt,
                validate_rerank,
            )
            valid_ids = list(payload["article_ids"])
            invalid_count = int(payload["invalid_id_count"])
        except httpx.TimeoutException:
            return None, self._failed_stage(
                started,
                "timeout",
                candidate_count=len(bounded_ids),
            )
        except ProviderRateLimitExceeded:
            return None, self._failed_stage(
                started,
                "rate_limited",
                candidate_count=len(bounded_ids),
            )
        except httpx.HTTPStatusError as error:
            return None, self._failed_stage(
                started,
                self._http_failure_reason(error),
                candidate_count=len(bounded_ids),
            )
        except httpx.HTTPError:
            return None, self._failed_stage(
                started,
                "provider_connection_error",
                candidate_count=len(bounded_ids),
            )
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None, self._failed_stage(
                started,
                "invalid_response",
                candidate_count=len(bounded_ids),
            )
        return valid_ids, {
            "attempted": True,
            "applied": bool(valid_ids),
            "cache_hit": cache_hit,
            "fallback_reason": None if valid_ids else "empty_valid_order",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "candidate_count": len(bounded_ids),
            "returned_count": len(valid_ids),
            "invalid_id_count": invalid_count,
        }

    def _request_validated_json(
        self,
        stage: str,
        prompt: str,
        validator: Callable[[object], dict[str, object]],
    ) -> tuple[dict[str, object], bool]:
        request_payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a bounded catalog-search component. Output strict "
                        "JSON only and follow catalog constraints exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": self._response_format(stage),
        }
        cache_key = f"{stage}:{json.dumps(request_payload, sort_keys=True)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached, True
        response = None
        for attempt in range(2):
            if not PROCESS_CALL_RATE_LIMITER.acquire(
                self.config.max_calls_per_minute
            ):
                raise ProviderRateLimitExceeded
            try:
                response = self._post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=self.config.timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 0:
                    time.sleep(0.25)
                    continue
                raise
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if attempt == 0 and (status in {400, 429} or status >= 500):
                    time.sleep(0.25)
                    continue
                raise
            break
        if response is None:
            raise RuntimeError("provider request was not attempted")
        envelope = response.json()
        content = envelope["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("provider content must be a JSON string")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("provider content must be a JSON object")
        validated = validator(parsed)
        self.cache.put(cache_key, validated)
        return validated, False

    def _response_format(self, stage: str) -> dict[str, object]:
        if stage == "intent":
            properties: dict[str, object] = {
                "reformulated_query": {
                    "type": ["string", "null"],
                },
                "style_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
                "exclusions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
            }
            required = ["reformulated_query", "style_terms", "exclusions"]
        elif stage == "rerank":
            properties = {
                "article_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": self.config.rerank_limit,
                }
            }
            required = ["article_ids"]
        else:
            raise ValueError(f"unsupported LLM stage: {stage}")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"catalog_search_{stage}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _http_failure_reason(error: httpx.HTTPStatusError) -> str:
        status = error.response.status_code
        if status == 429:
            return "provider_rate_limited"
        if status >= 500:
            return "provider_unavailable"
        return "provider_request_rejected"

    def _unavailable_stage(self, candidate_count: int | None = None) -> dict[str, object]:
        reason = "disabled" if not self.config.enabled else "missing_configuration"
        diagnostics: dict[str, object] = {
            "attempted": False,
            "applied": False,
            "cache_hit": False,
            "fallback_reason": reason,
            "latency_ms": 0.0,
        }
        if candidate_count is not None:
            diagnostics.update(
                {
                    "candidate_count": candidate_count,
                    "returned_count": 0,
                    "invalid_id_count": 0,
                }
            )
        return diagnostics

    @staticmethod
    def _failed_stage(
        started: float,
        reason: str,
        candidate_count: int | None = None,
    ) -> dict[str, object]:
        diagnostics: dict[str, object] = {
            "attempted": True,
            "applied": False,
            "cache_hit": False,
            "fallback_reason": reason,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        if candidate_count is not None:
            diagnostics.update(
                {
                    "candidate_count": candidate_count,
                    "returned_count": 0,
                    "invalid_id_count": 0,
                }
            )
        return diagnostics


def search_llm_health() -> dict[str, object]:
    return SearchLLMConfig.from_env().diagnostics()
