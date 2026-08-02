"""Layer1-B: collect pass facts for plans that cleared Layer1 (no new judgments)."""

from __future__ import annotations

from .explanation import render_explanation


def collect_applicable_facts(plan: dict, layer1_engine_result) -> list[dict]:
    """Render pass-side facts for in-scope rules that did not violate.

    Intended for ``verdict == 'pass'`` only. ``layer1_engine_result`` is the
    return value of ``evaluate_layer1_detailed`` (or a list of applicable rule
    dicts). Violated rules are excluded — those use the flagged side instead.
    """
    if isinstance(layer1_engine_result, dict):
        applicable = layer1_engine_result.get("applicable") or []
    else:
        applicable = list(layer1_engine_result or [])

    facts: list[dict] = []
    for rule in applicable:
        if rule.get("violated") or rule.get("skip_pass_fact"):
            continue
        text = render_explanation(
            rule,
            rule.get("pass_parameters") or {},
            lang="en",
            side="pass",
        )
        if not text:
            continue
        facts.append({"rule_id": rule["rule_id"], "text": text})
    return facts
