"""Structured request logging for latency measurement and future metering.

Emits one JSON object per line to a dedicated logger (stdout under Render).
Never logs user prompts, AI responses, plans, or explanation text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

REQUEST_LOGGER = logging.getLogger("guardamos.request")

# Ensure the metering logger always emits JSON lines even if the root logger
# is not configured (uvicorn typically configures logging; tests may not).
if not REQUEST_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    REQUEST_LOGGER.addHandler(_handler)
    REQUEST_LOGGER.setLevel(logging.INFO)
    REQUEST_LOGGER.propagate = False

# Primary population → ruleset_id (from rules/*.json).
_RULESET_BY_POPULATION = {
    "older_adult_healthy": "L1-RT-NSCA2019-v1",
    "healthy_adult_18plus": "L1-RT-ACSM2026-v1",
}

# Matched rule-id prefixes that imply additional rulesets were in play.
_RULESET_BY_RULE_PREFIX = (
    ("L1-RTT-", "L1-CSCCa-ReturnToTraining-v1"),
    ("L1-ECSS-", "L1-CSCCa-ReturnToTraining-v1"),
    ("L1-RT-NSCA-", "L1-RT-NSCA2019-v1"),
    ("L1-RT-", "L1-RT-ACSM2026-v1"),
)


def hash_api_key(api_key: str) -> str:
    """Return a non-reversible short hash of an API key (never log the raw key)."""
    pepper = (os.environ.get("GUARDAMOS_LOG_PEPPER") or "").strip()
    material = f"{pepper}:{api_key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:32]


def resolve_ruleset_ids(
    *,
    effective_population: str | None,
    matched_rules: list[str] | None,
) -> list[str]:
    """Derive which published rulesets were relevant for this request."""
    ids: list[str] = []
    pop = effective_population or "healthy_adult_18plus"
    primary = _RULESET_BY_POPULATION.get(pop)
    if primary:
        ids.append(primary)

    for rule_id in matched_rules or []:
        for prefix, ruleset_id in _RULESET_BY_RULE_PREFIX:
            if not rule_id.startswith(prefix):
                continue
            # First matching prefix wins (list is ordered specific → broad).
            if ruleset_id not in ids:
                ids.append(ruleset_id)
            break
    return ids


def build_request_log_record(
    *,
    api_key_hash: str,
    verdict: str | None,
    ruleset_ids: list[str],
    total_latency_ms: int,
    pipeline_latency_ms: int,
    billing_validate_ms: int | None,
    extraction_ms: int | None,
    layer3_ms: int | None,
    skip_layer3: bool | None = None,
    stage1_extraction_ms: int | None = None,
    stage2_extraction_ms: int | None = None,
    stage2_group_a_ms: int | None = None,
    stage2_group_b_ms: int | None = None,
) -> dict[str, Any]:
    """Assemble a PII-free metering / latency record.

    ``extraction_ms`` is the total extraction wall time (stage1 + stage2 when
    both ran). ``stage1_extraction_ms`` / ``stage2_extraction_ms`` are the
    per-stage breakdown; ``stage2_extraction_ms`` is null when stage 2 was
    skipped (population-gate early-exit). ``stage2_group_a_ms`` /
    ``stage2_group_b_ms`` are per-parallel-group timers (null when skipped).
    """
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "event": "audit_request",
        "api_key_hash": api_key_hash,
        "ruleset_ids": list(ruleset_ids),
        "verdict": verdict,
        "total_latency_ms": int(total_latency_ms),
        "pipeline_latency_ms": int(pipeline_latency_ms),
        "billing_validate_ms": (
            int(billing_validate_ms) if billing_validate_ms is not None else None
        ),
        "extraction_ms": int(extraction_ms) if extraction_ms is not None else None,
        "stage1_extraction_ms": (
            int(stage1_extraction_ms) if stage1_extraction_ms is not None else None
        ),
        "stage2_extraction_ms": (
            int(stage2_extraction_ms) if stage2_extraction_ms is not None else None
        ),
        "stage2_group_a_ms": (
            int(stage2_group_a_ms) if stage2_group_a_ms is not None else None
        ),
        "stage2_group_b_ms": (
            int(stage2_group_b_ms) if stage2_group_b_ms is not None else None
        ),
        "layer3_ms": int(layer3_ms) if layer3_ms is not None else None,
    }
    if skip_layer3 is not None:
        record["skip_layer3"] = bool(skip_layer3)
    return record


def emit_request_log(record: dict[str, Any]) -> None:
    """Write one JSON line to the request logger (stdout on Render)."""
    REQUEST_LOGGER.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def ms_since(start: float) -> int:
    """Elapsed milliseconds since ``time.perf_counter()`` start.

    Uses ceil-to-at-least-1 when any positive elapsed time is observed so
    sub-millisecond operations still register as a non-zero duration in logs.
    """
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0
    return max(1, int(round(elapsed * 1000)))
