"""Minimal HTTP API for Guardamos audit-engine.

Thin wrapper around the existing ``--raw-text`` pipeline. API keys are
validated against the private billing service over HTTP — this module does
not store or query customer key data itself.
"""

from __future__ import annotations

import os
import secrets
from time import perf_counter
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .pipeline import run_raw_text_pipeline
from .request_log import (
    build_request_log_record,
    emit_request_log,
    hash_api_key,
    ms_since,
    resolve_ruleset_ids,
)

load_dotenv()

app = FastAPI(title="Guardamos Audit Engine", version="0.1.0")

DEFAULT_BILLING_VALIDATE_URL = "http://localhost:8000/validate"


def _billing_validate_url() -> str:
    return (
        os.environ.get("BILLING_VALIDATE_URL")
        or os.environ.get("GUARDAMOS_BILLING_VALIDATE_URL")
        or DEFAULT_BILLING_VALIDATE_URL
    )


def _diagnostics_secret() -> str:
    return (os.environ.get("GUARDAMOS_DIAGNOSTICS_SECRET") or "").strip()


def _require_diagnostics_auth(x_diagnostics_secret: str | None) -> None:
    """Gate internal diagnostics behind a dedicated admin secret (not customer API keys)."""
    expected = _diagnostics_secret()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Diagnostics endpoint is not configured",
        )
    provided = (x_diagnostics_secret or "").strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _openai_api_key_status() -> dict[str, Any]:
    """Diagnostics for OPENAI_API_KEY without revealing the secret value."""
    raw = os.environ.get("OPENAI_API_KEY")
    if raw is None:
        return {
            "openai_api_key_set": False,
            "openai_api_key_nonempty_after_strip": False,
            "openai_api_key_starts_with_sk": False,
            "openai_api_key_had_leading_or_trailing_whitespace": False,
        }
    stripped = raw.strip()
    return {
        "openai_api_key_set": True,
        "openai_api_key_nonempty_after_strip": bool(stripped),
        "openai_api_key_starts_with_sk": stripped.startswith("sk-"),
        "openai_api_key_had_leading_or_trailing_whitespace": raw != stripped,
    }


def validate_api_key_via_billing(api_key: str) -> bool:
    """Ask billing whether ``api_key`` is active (HTTP, no local key DB)."""
    if not api_key:
        return False
    url = _billing_validate_url()
    try:
        response = httpx.get(url, params={"key": api_key}, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="Billing key validation service unavailable",
        ) from None
    return bool(payload.get("valid"))


def _client_facing_payload(pipeline_result: dict[str, Any]) -> dict[str, Any]:
    """Drop internal underscore-prefixed keys (e.g. timing) from the audit body."""
    audit = dict(pipeline_result.get("audit") or {})
    audit = {k: v for k, v in audit.items() if not str(k).startswith("_")}
    return {
        "extraction": pipeline_result.get("extraction"),
        "audit": audit,
    }


class AuditRequest(BaseModel):
    user_prompt: str = Field(..., min_length=1)
    ai_response: str = Field(..., min_length=1)
    lang: str = Field(default="en", pattern="^(en|pt|ja)$")
    skip_layer3: bool = Field(
        default=True,
        description=(
            "When true (the API default), skip the Layer3 LLM summary call even "
            "on pass. Deterministic checked_facts are still returned on pass. "
            "Set false to also generate layer3_response (extra OpenAI call / latency)."
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Public liveness probe — no configuration details."""
    return {"status": "ok", "service": "guardamos-audit-engine"}


@app.get("/internal/diagnostics")
def internal_diagnostics(
    x_diagnostics_secret: str | None = Header(default=None, alias="X-Diagnostics-Secret"),
) -> dict[str, Any]:
    """Operator-only env diagnostics (requires ``GUARDAMOS_DIAGNOSTICS_SECRET``)."""
    _require_diagnostics_auth(x_diagnostics_secret)
    return {
        "service": "guardamos-audit-engine",
        **_openai_api_key_status(),
        "billing_validate_url_configured": bool(
            os.environ.get("BILLING_VALIDATE_URL")
            or os.environ.get("GUARDAMOS_BILLING_VALIDATE_URL")
        ),
    }


@app.post("/audit")
def audit(
    body: AuditRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Run the same pipeline as ``main.py --raw-text``.

    Requires a valid ``X-API-Key`` issued by the billing service.

    ``skip_layer3`` defaults to **true**: Layer3 (LLM narrative on pass) is not
    called unless the client sets ``skip_layer3: false``. Pass responses still
    include deterministic ``checked_facts``; only ``layer3_response`` is omitted
    when skipped.
    """
    total_t0 = perf_counter()
    billing_ms: int | None = None
    pipeline_result: dict[str, Any] | None = None
    key_hash = hash_api_key(x_api_key) if x_api_key else hash_api_key("")

    try:
        billing_t0 = perf_counter()
        valid = bool(x_api_key) and validate_api_key_via_billing(x_api_key)
        billing_ms = ms_since(billing_t0)
        if not valid:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        pipeline_result = run_raw_text_pipeline(
            body.user_prompt,
            body.ai_response,
            lang=body.lang,
            skip_layer3=body.skip_layer3,
        )
        return _client_facing_payload(pipeline_result)
    except OSError as exc:
        # Includes EnvironmentError when OPENAI_API_KEY is missing.
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except HTTPException:
        raise
    except Exception as exc:
        # Surface OpenAI auth / API failures instead of a bare 500.
        name = type(exc).__name__
        if name in {
            "AuthenticationError",
            "PermissionDeniedError",
            "RateLimitError",
            "APIConnectionError",
            "APIStatusError",
            "APIError",
        } or "openai" in type(exc).__module__.lower():
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI request failed ({name}): {exc}",
            ) from None
        raise
    finally:
        # Always emit a metering line for authenticated attempts that reached
        # validation; include failures after a valid key when possible.
        audit_body = (pipeline_result or {}).get("audit") or {}
        timing = audit_body.get("_timing") or {}
        try:
            pipeline_ms = int(
                timing.get("pipeline_latency_ms")
                if timing.get("pipeline_latency_ms") is not None
                else 0
            )
            total_ms = ms_since(total_t0)
            # Wall clock can round to 0ms in tests; keep totals coherent.
            bill = int(billing_ms or 0)
            total_ms = max(total_ms, pipeline_ms + bill)
            emit_request_log(
                build_request_log_record(
                    api_key_hash=key_hash,
                    verdict=audit_body.get("verdict"),
                    ruleset_ids=resolve_ruleset_ids(
                        effective_population=timing.get("effective_population"),
                        matched_rules=audit_body.get("matched_rules"),
                    ),
                    total_latency_ms=total_ms,
                    pipeline_latency_ms=pipeline_ms if pipeline_result else total_ms,
                    billing_validate_ms=billing_ms,
                    extraction_ms=timing.get("extraction_ms"),
                    stage1_extraction_ms=timing.get("stage1_extraction_ms"),
                    stage2_extraction_ms=timing.get("stage2_extraction_ms"),
                    layer3_ms=timing.get("layer3_ms"),
                    skip_layer3=body.skip_layer3,
                )
            )
        except Exception:
            # Logging must never break the request path.
            pass
