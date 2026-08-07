"""Audit orchestration: Layer1 + Layer1-B + Layer2 stub + Layer3 + explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .explanation import render_explanation
from .layer1_engine import evaluate_layer1_detailed, load_merged_rulesets, load_ruleset
from .layer1b_synthesizer import collect_applicable_facts
from .layer2_stub import evaluate_layer2
from .layer3_generator import generate_layer3_response

REJECT_ACTIONS = frozenset({"route_to_layer2_or_reject", "reject"})

_SUMMARY_LANGS = frozenset({"en", "pt", "ja"})


def build_summary(
    verdict: str,
    matched_rules: list[str] | None = None,
    checked_facts: list[dict] | None = None,
    *,
    lang: str = "en",
    **context: Any,
) -> str:
    """Deterministically build a short, always-present summary string.

    Assembled only from already-computed counts and ids (matched_rules,
    checked_facts, optional injection/meta context). Does not call an LLM
    and does not add claims beyond what those fields already state.

    ``lang`` selects en / pt / ja (falls back to en).
    """
    lang = lang if lang in _SUMMARY_LANGS else "en"
    rules = list(matched_rules or [])
    facts = list(checked_facts or [])

    if verdict == "pass":
        n = len(facts)
        return {
            "en": f"{n} checks passed, 0 flagged.",
            "pt": f"{n} verificações passaram, 0 sinalizadas.",
            "ja": f"{n}件のチェックがパス、フラグ0件。",
        }[lang]

    if verdict in ("flagged", "rejected"):
        ids = ", ".join(rules)
        n = len(rules)
        if ids:
            return {
                "en": (
                    f"{n} issue(s) flagged: {ids}. "
                    "See explanations for details."
                ),
                "pt": (
                    f"{n} problema(s) sinalizado(s): {ids}. "
                    "Consulte as explanations para detalhes."
                ),
                "ja": (
                    f"{n}件の問題がフラグされました: {ids}。"
                    "詳細は explanations を参照してください。"
                ),
            }[lang]
        return {
            "en": f"{n} issue(s) flagged. See explanations for details.",
            "pt": f"{n} problema(s) sinalizado(s). Consulte as explanations.",
            "ja": f"{n}件の問題がフラグされました。詳細は explanations を参照してください。",
        }[lang]

    if verdict == "insufficient_data":
        return {
            "en": "Not enough information could be extracted to complete an audit.",
            "pt": (
                "Não foi possível extrair informação suficiente "
                "para concluir a auditoria."
            ),
            "ja": "監査を完了するのに十分な情報を抽出できませんでした。",
        }[lang]

    if verdict == "flagged_for_review":
        details: list[str] = []
        injection_warning = context.get("injection_warning") or []
        if injection_warning:
            preview = ", ".join(str(p) for p in injection_warning[:3])
            if len(injection_warning) > 3:
                preview += ", ..."
            details.append(
                {
                    "en": f"injection pattern(s) detected ({preview})",
                    "pt": f"padrão(ões) de injeção detectado(s) ({preview})",
                    "ja": f"インジェクションパターンを検出 ({preview})",
                }[lang]
            )
        meta_detected = bool(context.get("possible_meta_instruction_detected"))
        meta_evidence = context.get("meta_instruction_evidence")
        if meta_detected or meta_evidence:
            if isinstance(meta_evidence, str) and meta_evidence.strip():
                short = " ".join(meta_evidence.strip().split())
                if len(short) > 80:
                    short = short[:77] + "..."
                details.append(
                    {
                        "en": f"possible meta-instruction detected in input ({short})",
                        "pt": (
                            f"possível meta-instrução detectada na entrada ({short})"
                        ),
                        "ja": f"入力にメタ指示の可能性 ({short})",
                    }[lang]
                )
            else:
                details.append(
                    {
                        "en": "possible meta-instruction detected in input",
                        "pt": "possível meta-instrução detectada na entrada",
                        "ja": "入力にメタ指示の可能性",
                    }[lang]
                )
        prefix = {
            "en": "Flagged for manual review",
            "pt": "Sinalizado para revisão manual",
            "ja": "人手レビューのためフラグ",
        }[lang]
        if details:
            return prefix + ": " + "; ".join(details) + "."
        return prefix + "."

    return {
        "en": f"Audit completed with verdict '{verdict}'.",
        "pt": f"Auditoria concluída com veredicto '{verdict}'.",
        "ja": f"監査完了（判定: '{verdict}'）。",
    }[lang]


def _derive_verdict(
    layer1_matches: list[dict],
    layer2_matches: list[dict],
    *,
    checked_facts: list[dict] | None = None,
) -> str:
    """Derive verdict from matches and (when clear) checked_facts.

    - matched_rules > 0 → rejected / flagged (unchanged)
    - matched_rules == 0 and checked_facts > 0 → pass
    - matched_rules == 0 and checked_facts == 0 → insufficient_data
    """
    all_matches = layer1_matches + layer2_matches
    if all_matches:
        # Until Layer2 exists, treat route_to_layer2_or_reject as reject
        # (safe fallback while the Layer2 stub returns no matches).
        if any(m.get("action") in REJECT_ACTIONS for m in all_matches):
            return "rejected"
        return "flagged"
    if checked_facts:
        return "pass"
    return "insufficient_data"


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

    layer1_result = evaluate_layer1_detailed(plan, ruleset=ruleset)
    layer1_matches = layer1_result["matched"]
    layer2_matches = evaluate_layer2(plan)
    all_matches = layer1_matches + layer2_matches

    explanations = [
        render_explanation(
            m,
            m.get("matched_parameters") or {},
            lang=lang,
            side=m.get("explanation_side") or "flagged",
        )
        for m in all_matches
    ]
    explanations = [e for e in explanations if e]

    checked_facts: list[dict] = []
    # Collect pass facts only when nothing matched — needed to distinguish
    # pass vs insufficient_data before Layer3.
    if not all_matches:
        checked_facts = collect_applicable_facts(plan, layer1_result)

    verdict = _derive_verdict(
        layer1_matches, layer2_matches, checked_facts=checked_facts
    )

    layer3_response = None
    layer3_verdict = _verdict_for_layer3(verdict)
    # Layer1-B + Layer3 only on clear/pass; never for flagged/rejected/insufficient.
    if layer3_verdict == "clear" and not skip_layer3:
        layer3_response = (
            generate_layer3_response(
                plan, layer3_verdict, checked_facts=checked_facts
            )
            or None
        )

    matched_rule_ids = [m["rule_id"] for m in all_matches]
    pass_facts = checked_facts if verdict == "pass" else []
    return {
        "verdict": verdict,
        "summary": build_summary(
            verdict, matched_rule_ids, pass_facts, lang=lang
        ),
        "matched_rules": matched_rule_ids,
        "explanations": explanations,
        "checked_facts": pass_facts,
        "layer3_response": layer3_response,
        "ruleset_version": ruleset_version,
    }
