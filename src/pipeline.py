"""Shared --raw-text audit pipeline (CLI and HTTP API).

No new judgment logic: wraps detect_injection_patterns → extract_plan →
run_audit with the same post-processing as ``main.py --raw-text``.
"""

from __future__ import annotations

from typing import Any

from .audit import build_summary, run_audit
from .injection_guard import detect_injection_patterns
from .plan_extractor import extract_plan


def _format_extraction_payload(extraction: dict) -> dict[str, Any]:
    """Same payload shape printed under === Extraction === in main.py."""
    from .plan_extractor import fields_left_null_without_evidence

    plan = extraction.get("plan") or {}
    evidence = extraction.get("extraction_evidence") or {}
    warnings = extraction.get("extraction_warnings") or []
    queried = extraction.get("queried_fields")
    if queried is None:
        # Backward compat for tests that stub extract_plan without queried_fields.
        null_without_evidence = sorted(
            name
            for name, quote in evidence.items()
            if name not in plan and quote is None
        )
    else:
        null_without_evidence = fields_left_null_without_evidence(
            plan, evidence, queried_fields=set(queried)
        )
    return {
        "plan": plan,
        "extraction_evidence": {
            k: v for k, v in evidence.items() if k in plan or v is not None
        },
        "fields_left_null_without_evidence": null_without_evidence,
        "extraction_warnings": warnings,
        "possible_meta_instruction_detected": bool(
            extraction.get("possible_meta_instruction_detected")
        ),
        "meta_instruction_evidence": extraction.get("meta_instruction_evidence"),
        "stage2_ran": bool(extraction.get("stage2_ran")),
        "effective_population": extraction.get("effective_population"),
    }


def _apply_raw_text_verdict_overrides(
    result: dict[str, Any],
    extraction: dict[str, Any],
    injection_warning: list[str],
    *,
    lang: str = "en",
) -> dict[str, Any]:
    """Apply injection / meta / missed-cue overrides (identical to CLI)."""
    meta_detected = bool(extraction.get("possible_meta_instruction_detected"))
    if injection_warning or meta_detected:
        result["verdict"] = "flagged_for_review"
        if injection_warning:
            result["injection_warning"] = injection_warning
        if meta_detected:
            result["possible_meta_instruction_detected"] = True
            result["meta_instruction_evidence"] = extraction.get(
                "meta_instruction_evidence"
            )
        result["checked_facts"] = []
        result["layer3_response"] = None
        result["summary"] = build_summary(
            result["verdict"],
            result.get("matched_rules") or [],
            result.get("checked_facts") or [],
            lang=lang,
            injection_warning=result.get("injection_warning"),
            possible_meta_instruction_detected=result.get(
                "possible_meta_instruction_detected"
            ),
            meta_instruction_evidence=result.get("meta_instruction_evidence"),
        )
        return result

    missed_cue = any(
        "extraction may have missed an explicit clue" in w
        for w in (extraction.get("extraction_warnings") or [])
    )
    if missed_cue and result["verdict"] == "pass":
        result["verdict"] = "insufficient_data"
        result["checked_facts"] = []
        result["layer3_response"] = None
        result["summary"] = build_summary(
            result["verdict"],
            result.get("matched_rules") or [],
            result.get("checked_facts") or [],
            lang=lang,
        )
    return result


def run_raw_text_pipeline(
    user_prompt: str,
    ai_response: str,
    *,
    lang: str = "en",
    skip_layer3: bool = False,
) -> dict[str, Any]:
    """Run the full free-text audit pipeline.

    Returns::

        {
          "extraction": <Extraction section payload>,
          "audit": <Audit section payload>,
        }

    Timing metadata is attached under ``audit["_timing"]`` for the HTTP layer
    to log; callers that serialize the audit payload for clients should strip
    underscore-prefixed keys if desired (the API does this).
    """
    from time import perf_counter

    from .layer1_engine import (
        apply_deterministic_age_derived_flags,
        effective_target_population,
    )
    from .request_log import ms_since

    pipeline_t0 = perf_counter()

    injection_hits = detect_injection_patterns(user_prompt) + detect_injection_patterns(
        ai_response
    )
    seen: set[str] = set()
    injection_warning: list[str] = []
    for phrase in injection_hits:
        if phrase not in seen:
            seen.add(phrase)
            injection_warning.append(phrase)

    extraction_t0 = perf_counter()
    extraction = extract_plan(user_prompt, ai_response)
    extraction_ms = ms_since(extraction_t0)
    extraction_timing = dict(extraction.get("_timing") or {})
    stage1_ms = extraction_timing.get("stage1_extraction_ms")
    stage2_ms = extraction_timing.get("stage2_extraction_ms")
    if stage1_ms is not None:
        # Prefer sum of stage timers when available; else wall-clock extract_plan.
        parts = [int(stage1_ms)]
        if stage2_ms is not None:
            parts.append(int(stage2_ms))
        extraction_ms = sum(parts)

    # Early-exit and full path both use run_audit on the (possibly stage1-only)
    # plan so Layer1 evaluation stays a single code path.
    result = run_audit(
        extraction["plan"],
        lang=lang,
        skip_layer3=skip_layer3,
    )
    result = _apply_raw_text_verdict_overrides(
        result, extraction, injection_warning, lang=lang
    )

    routed_plan = apply_deterministic_age_derived_flags(dict(extraction.get("plan") or {}))
    audit_timing = dict(result.get("_timing") or {})
    audit_timing["extraction_ms"] = extraction_ms
    audit_timing["stage1_extraction_ms"] = stage1_ms
    audit_timing["stage2_extraction_ms"] = stage2_ms
    audit_timing["pipeline_latency_ms"] = ms_since(pipeline_t0)
    audit_timing["effective_population"] = effective_target_population(routed_plan)
    result["_timing"] = audit_timing

    return {
        "extraction": _format_extraction_payload(extraction),
        "audit": result,
    }
