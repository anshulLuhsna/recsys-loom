"""Operational HTTP concerns shared by the serving entrypoints."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

LOGGER = logging.getLogger("recsys_loom.http")


def _route_template(path: str) -> str:
    if path.startswith("/api/recommendations/"):
        return "/api/recommendations/{customer_id}"
    if path.startswith("/api/articles/"):
        return "/api/articles/{article_id}"
    return path


def _allowed_origins() -> list[str]:
    configured = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000",
    )
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    LOGGER.propagate = False


def configure_api(app: FastAPI, service_name: str) -> None:
    """Attach strict CORS and structured request logging."""

    _configure_logging()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_log(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            LOGGER.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "service": service_name,
                        "request_id": request_id,
                        "method": request.method,
                        "path": _route_template(request.url.path),
                        "status": status_code,
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                        "image_digest": os.environ.get("IMAGE_DIGEST", "local"),
                        "model_version": os.environ.get("MODEL_VERSION", "unknown"),
                    },
                    separators=(",", ":"),
                )
            )


def readiness_payload(
    service_name: str,
    checks: dict[str, bool],
    details: dict[str, object] | None = None,
) -> tuple[dict[str, object], int]:
    ready = all(checks.values())
    payload: dict[str, object] = {
        "status": "ready" if ready else "not_ready",
        "service": service_name,
        "checks": checks,
    }
    if details:
        payload["details"] = details
    return payload, 200 if ready else 503
