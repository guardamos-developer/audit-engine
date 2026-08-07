"""Coverage for population-gate exclusions, age/minor Zone B, and explanation detail."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
from src.explanation import render_explanation  # noqa: E402
from src.layer1_engine import (  # noqa: E402
    apply_deterministic_age_derived_flags,
    evaluate_layer1,
    load_merged_rulesets,
)


def _adult_plan(**overrides):
    plan = {
        "goal": "strength",
        "sessions_per_week": 3,
        "sets_per_exercise": 3,
        "load_percent_1RM": 80,
        "injury_present": False,
        "post_surgical": False,
        "pain_present": False,
        "pregnant": False,
        "true_beginner_first_weeks": False,
        "program_mandates_training_to_failure": False,
        "program_mandates_complex_periodization_as_required": False,
        "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
        "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
    }
    plan.update(overrides)
    return plan


def _older_plan(**overrides):
    plan = _adult_plan(
        age_years=70,
        frailty_present=False,
        uncontrolled_hypertension=False,
        unstable_cardiovascular_disease=False,
    )
    plan.update(overrides)
    return plan


def _gate_rule(rule_id: str) -> dict:
    data = load_merged_rulesets()
    for rule in data.get("rules") or []:
        if rule.get("rule_id") == rule_id:
            return rule
    raise AssertionError(f"missing rule {rule_id}")


# ---------------------------------------------------------------------------
# Part 1 — simulation miss cases 57 / 140 / 144 / 186 (0-based ids in report)
# ---------------------------------------------------------------------------


def test_sim_case57_age14_minor_null_is_rejected():
    """Case57: age_years=14 extracted, minor left null → Zone B forces exclusion."""
    plan = _adult_plan(age_years=14, minor=None)
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "rejected"
    assert "L1-RT-0001" in result["matched_rules"]
    assert "minor" in result["explanations"][0]


def test_sim_case140_age15_minor_null_is_rejected():
    """Case140: age_years=15 extracted, minor left null → Zone B forces exclusion."""
    plan = _adult_plan(age_years=15, minor=None)
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "rejected"
    assert "L1-RT-0001" in result["matched_rules"]
    assert "minor" in result["explanations"][0]


def test_sim_case144_soft_frailty_excluded_after_lenient_label():
    """Case144: soft Japanese frailty phrasing → frailty_present=True (lenient).

    Unit test uses the post-lenient-extractor plan state (Zone A cannot call the
    LLM); asserts NSCA population gate rejects when frailty is affirmative.
    """
    plan = _older_plan(
        age_years=75,
        frailty_present=True,
        injury_present=False,
        pain_present=False,
    )
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "rejected"
    assert "L1-RT-NSCA-0001" in result["matched_rules"]
    assert "frailty_present" in result["explanations"][0]


def test_sim_case186_soft_frailty_excluded_after_lenient_label():
    """Case186: soft frailty phrasing → frailty_present=True (lenient) → excluded."""
    plan = _older_plan(
        age_years=78,
        frailty_present=True,
        injury_present=False,
        pain_present=False,
    )
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "rejected"
    assert "L1-RT-NSCA-0001" in result["matched_rules"]
    assert "frailty_present" in result["explanations"][0]


# ---------------------------------------------------------------------------
# Part 1 — Fix1 boundary ages 17 / 18
# ---------------------------------------------------------------------------


def test_age_years_17_forces_minor_true():
    forced = apply_deterministic_age_derived_flags({"age_years": 17, "minor": None})
    assert forced["minor"] is True
    result = run_audit(_adult_plan(age_years=17, minor=None), lang="en", skip_layer3=True)
    assert result["verdict"] == "rejected"
    assert "L1-RT-0001" in result["matched_rules"]


def test_age_years_18_does_not_force_minor():
    forced = apply_deterministic_age_derived_flags({"age_years": 18, "minor": None})
    assert forced.get("minor") is None
    # Healthy adult at 18 with no exclusion flags must not hit the population gate.
    matches = evaluate_layer1(_adult_plan(age_years=18, minor=None))
    assert all(m["rule_id"] != "L1-RT-0001" for m in matches)


# ---------------------------------------------------------------------------
# Part 2 — triggered_exclusions in explanations (en/pt/ja)
# ---------------------------------------------------------------------------


def test_single_triggered_exclusion_named_in_explanation():
    result = run_audit(
        _adult_plan(injury_present=True),
        lang="en",
        skip_layer3=True,
    )
    assert result["verdict"] == "rejected"
    text = result["explanations"][0]
    assert "injury_present" in text
    assert "Triggered exclusion(s):" in text
    assert "future release" in text
    assert "Layer 2" not in text


def test_multiple_triggered_exclusions_listed():
    result = run_audit(
        _adult_plan(injury_present=True, pregnant=True, pain_present=True),
        lang="en",
        skip_layer3=True,
    )
    assert result["verdict"] == "rejected"
    text = result["explanations"][0]
    for flag in ("injury_present", "pain_present", "pregnant"):
        assert flag in text


@pytest.mark.parametrize("lang", ["en", "pt", "ja"])
def test_triggered_exclusions_embedded_in_all_langs(lang: str):
    result = run_audit(
        _adult_plan(injury_present=True),
        lang=lang,
        skip_layer3=True,
    )
    assert result["verdict"] == "rejected"
    assert "injury_present" in result["explanations"][0]


@pytest.mark.parametrize("lang", ["en", "pt", "ja"])
def test_nsca_gate_triggered_exclusions_all_langs(lang: str):
    result = run_audit(
        _older_plan(frailty_present=True),
        lang=lang,
        skip_layer3=True,
    )
    assert result["verdict"] == "rejected"
    assert "L1-RT-NSCA-0001" in result["matched_rules"]
    assert "frailty_present" in result["explanations"][0]


def test_gate_templates_include_triggered_placeholder_and_no_layer2_routing():
    for rule_id in ("L1-RT-0001", "L1-RT-NSCA-0001"):
        rule = _gate_rule(rule_id)
        flagged = (rule.get("reason_template") or {}).get("flagged") or {}
        for lang in ("en", "pt", "ja"):
            tmpl = flagged[lang]
            assert "{triggered_exclusions}" in tmpl
            assert "Layer 2" not in tmpl and "Layer2" not in tmpl
            rendered = render_explanation(
                rule,
                {"triggered_exclusions": "injury_present"},
                lang=lang,
                side="flagged",
            )
            assert "injury_present" in rendered


def test_population_gate_regression_injury_still_short_circuits():
    """Existing gate behavior: injury alone still rejects via L1-RT-0001 only."""
    matches = evaluate_layer1(_adult_plan(injury_present=True, sessions_per_week=1))
    assert len(matches) == 1
    assert matches[0]["rule_id"] == "L1-RT-0001"
    assert matches[0]["matched_parameters"]["triggered_exclusions"] == "injury_present"
