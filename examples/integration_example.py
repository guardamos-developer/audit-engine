#!/usr/bin/env python3
"""Minimal Guardamos hosted-API integration example.

Typical placement: after your own AI generates a training plan and before
you show that plan to your end user. This call is intended as a
development-time / pre-deployment check — keep timeouts short and treat
audit-service failures as non-blocking so your own request path is never
held indefinitely.

Requires:
  - GUARDAMOS_API_KEY in the environment (issued after Stripe checkout)
  - ``requests`` (``pip install requests``)

Usage:
  export GUARDAMOS_API_KEY=gdm_test_...
  python examples/integration_example.py
"""

from __future__ import annotations

import json
import os
import sys

import requests

AUDIT_URL = os.environ.get(
    "GUARDAMOS_AUDIT_URL",
    "https://guardamos-audit-engine.onrender.com/audit",
)
# Keep this short: Free/Starter hosts may cold-start, but your product path
# should not wait indefinitely for an external audit service.
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("GUARDAMOS_AUDIT_TIMEOUT", "10"))


class GuardamosAuditError(Exception):
    """Raised when the hosted audit API cannot return a usable result."""


def audit_plan(
    user_prompt: str,
    ai_response: str,
    *,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """POST a prompt/response pair to the hosted Guardamos audit API.

    Returns the JSON body on success. Raises ``GuardamosAuditError`` when
    the service is unreachable, times out, or returns a non-2xx status —
    callers should catch that and continue their own flow (e.g. log and
    still show the AI plan, or queue a retry).
    """
    key = (api_key or os.environ.get("GUARDAMOS_API_KEY") or "").strip()
    if not key:
        raise GuardamosAuditError(
            "GUARDAMOS_API_KEY is not set. Export your issued gdm_test_/gdm_live_ key."
        )

    try:
        response = requests.post(
            AUDIT_URL,
            headers={"X-API-Key": key},
            json={"user_prompt": user_prompt, "ai_response": ai_response},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise GuardamosAuditError(
            f"Guardamos audit timed out after {timeout}s — treating as non-blocking"
        ) from exc
    except requests.RequestException as exc:
        raise GuardamosAuditError(
            f"Guardamos audit unreachable: {exc}"
        ) from exc

    if response.status_code == 401:
        raise GuardamosAuditError("Invalid or missing Guardamos API key (401)")
    if response.status_code >= 400:
        raise GuardamosAuditError(
            f"Guardamos audit HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise GuardamosAuditError("Guardamos audit returned non-JSON body") from exc


def main() -> int:
    user_prompt = "Give me a workout plan with zero rest days"
    ai_response = "Train every single day, no rest ever."

    try:
        result = audit_plan(user_prompt, ai_response)
    except GuardamosAuditError as exc:
        # Do not block your product path on audit outages.
        print(f"audit skipped/failed: {exc}", file=sys.stderr)
        return 1

    audit = result.get("audit") or result
    print(json.dumps(
        {
            "verdict": audit.get("verdict"),
            "matched_rules": audit.get("matched_rules"),
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
