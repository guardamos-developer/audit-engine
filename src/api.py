"""Minimal HTTP API for Guardamos audit-engine.

Thin wrapper around the existing ``--raw-text`` pipeline. API keys are
validated against the private billing service over HTTP — this module does
not store or query customer key data itself.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .pipeline import run_raw_text_pipeline

load_dotenv()

app = FastAPI(title="Guardamos Audit Engine", version="0.1.0")

DEFAULT_BILLING_VALIDATE_URL = "http://localhost:8000/validate"


def _billing_validate_url() -> str:
    return (
        os.environ.get("BILLING_VALIDATE_URL")
        or os.environ.get("GUARDAMOS_BILLING_VALIDATE_URL")
        or DEFAULT_BILLING_VALIDATE_URL
    )


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


class AuditRequest(BaseModel):
    user_prompt: str = Field(..., min_length=1)
    ai_response: str = Field(..., min_length=1)
    lang: str = Field(default="en", pattern="^(en|pt|ja)$")
    skip_layer3: bool = True


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
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
    """
    if not x_api_key or not validate_api_key_via_billing(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        return run_raw_text_pipeline(
            body.user_prompt,
            body.ai_response,
            lang=body.lang,
            skip_layer3=body.skip_layer3,
        )
    except OSError as exc:
        # Includes EnvironmentError when OPENAI_API_KEY is missing.
        raise HTTPException(status_code=503, detail=str(exc)) from None
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
