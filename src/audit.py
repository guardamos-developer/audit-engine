"""Audit orchestration: Layer1 + Layer2 stub + Layer3 + explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .explanation import render_explanation
from .layer1_engine import evaluate_layer1, load_merged_rulesets, load_ruleset
from .layer2_stub import evaluate_layer2
from .layer3_generator import generate_layer3_response

REJECT_ACTIONS = frozenset({"route_to_layer2_or_reject", "reject"})


def _derive_verdict(layer1_matches: list[dict], layer2_matches: list[dict]) -> str:
    all_matches = layer1_matches + layer2_matches
    if not all_matches:
        return "pass"
    # TODO: Layer2実装後はLayer2へのroutingを検討
    # (action=route_to_layer2_or_reject を即rejectedにしているのはLayer2スタブ時点の安全側フォールバック)
    if any(m.get("action") in REJECT_ACTIONS for m in all_matches):
        return "rejected"
    return "flagged"


def _verdict_for_layer3(verdict: str) -> str:
    """Map audit verdict to Layer3 gate vocabulary (clear / flagged / rejected)."""
    if verdict == "pass":
        return "clear"
    return verdict


def run_audit(
    plan: dict,
    lang: str = "en",
    rules_path: str | Path | None = None,
    *,
    skip_layer3: bool = False,
) -> dict[str, Any]:
    """Run the full audit pipeline and return a structured audit log."""
    if rules_path is not None:
        ruleset = load_ruleset(rules_path)
    else:
        ruleset = load_merged_rulesets()
    ruleset_version = ruleset.get("ruleset_id", "L1-RT-ACSM2026-v1")

    layer1_matches = evaluate_layer1(plan, ruleset=ruleset)
    layer2_matches = evaluate_layer2(plan)

    verdict = _derive_verdict(layer1_matches, layer2_matches)
    all_matches = layer1_matches + layer2_matches

    explanations = [
        render_explanation(m, m.get("matched_parameters") or {}, lang=lang)
        for m in all_matches
    ]
    explanations = [e for e in explanations if e]

    layer3_response = None
    layer3_verdict = _verdict_for_layer3(verdict)
    # Only call LLM when clear; never for flagged/rejected.
    if layer3_verdict == "clear" and not skip_layer3:
        layer3_response = generate_layer3_response(plan, layer3_verdict) or None

    return {
        "verdict": verdict,
        "matched_rules": [m["rule_id"] for m in all_matches],
        "explanations": explanations,
        "layer3_response": layer3_response,
        "ruleset_version": ruleset_version,
    }
