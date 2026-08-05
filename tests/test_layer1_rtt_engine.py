"""Tests for CSCCa return-to-training Layer1 rules (L1-RTT-0001 / 0002a-h)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit
from src.layer1_engine import (
    evaluate_layer1,
    evaluate_long_inactivity_track_compliance,
)

CHATGPT_PLAN_PATH = ROOT / "sample_plans" / "chatgpt_6month_layoff.json"
CORRECTED_PLAN_PATH = ROOT / "sample_plans" / "chatgpt_6month_layoff_corrected.json"
THREE_WEEK_BREAK_PATH = ROOT / "sample_plans" / "chatgpt_3week_break.json"
ZERO_REST_PLAN_PATH = ROOT / "sample_plans" / "chatgpt_zero_rest_days.json"
GROUND_TRUTH_DIR = ROOT / "tests" / "extraction_ground_truth"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def test_chatgpt_6month_layoff_matches_rtt_0001_and_week1_exceedances():
    """6-month layoff without track → 0001 + Week1 sets/frequency exceedances."""
    plan = _load_json(CHATGPT_PLAN_PATH)
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}

    assert "L1-RTT-0001" in rule_ids
    assert "L1-RTT-0002a" in rule_ids  # sets=4 > Week1 max 2
    assert "L1-RTT-0002d" in rule_ids  # frequency=4 > Week1 max 2
    # intensity / rest are null → these must not fire
    assert "L1-RTT-0002b" not in rule_ids
    assert "L1-RTT-0002c" not in rule_ids

    result = run_audit(plan, lang="en", skip_layer3=True)
    assert "L1-RTT-0001" in result["matched_rules"]
    assert "L1-RTT-0002a" in result["matched_rules"]
    assert "L1-RTT-0002d" in result["matched_rules"]
    assert "L1-RTT-0008" not in result["matched_rules"]  # long track, not moderate
    assert result["verdict"] == "rejected"
    assert any(
        "prolonged period of inactivity" in e or "Table 9" in e or "Week 1" in e
        for e in result["explanations"]
    )


def test_corrected_layoff_plan_matches_nothing():
    """CSCCa Week1-aligned corrected plan → no matched rules (pass)."""
    plan = _load_json(CORRECTED_PLAN_PATH)
    matches = evaluate_layer1(plan)
    assert matches == []

    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["matched_rules"] == []
    assert result["verdict"] == "pass"


def test_corrected_layoff_plan_checked_facts_include_week1_rtt_pass():
    """Pass path: Layer1-B collects week-1 RTT pass facts (not Week2).

    intensity_percent_1RM is null in the sample plan, so L1-RTT-0002b has no
    observed metric and is omitted (Layer1-B does not invent pass claims).
    """
    plan = _load_json(CORRECTED_PLAN_PATH)
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "pass"
    fact_ids = {f["rule_id"] for f in result["checked_facts"]}
    for rid in (
        "L1-RTT-0001",
        "L1-RTT-0002a",
        "L1-RTT-0002c",
        "L1-RTT-0002d",
    ):
        assert rid in fact_ids
    assert "L1-RTT-0002b" not in fact_ids  # intensity not reported
    # Week2 progression rules are out of scope for weeks_since_return == 1
    assert "L1-RTT-0002e" not in fact_ids
    assert "L1-RTT-0002f" not in fact_ids
    assert "L1-RTT-0002g" not in fact_ids
    assert "L1-RTT-0002h" not in fact_ids


def test_no_layoff_does_not_fire_rtt_rules():
    """No layoff (unset or <4 weeks) → L1-RTT-0001 / 0002a-h do not fire."""
    normal = {
        "target_population": "healthy_adult_18plus",
        "goal": "strength",
        "sessions_per_week": 3,
        "sets_per_exercise": 3,
        "load_percent_1RM": 80,
        "weekly_sets_per_muscle_group": 12,
        "program_mandates_training_to_failure": False,
        "program_mandates_complex_periodization_as_required": False,
        "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
        "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
    }
    matches = evaluate_layer1(normal)
    rule_ids = {m["rule_id"] for m in matches}
    assert not any(rid.startswith("L1-RTT-") for rid in rule_ids)
    assert matches == []

    short_break = {
        **normal,
        "inactivity_duration_weeks": 2,
        "weeks_since_return": 1,
        "plan_week_parameters": {
            "1": {
                "sets": 4,
                "reps": "5-6",
                "intensity_percent_1RM": None,
                "rest_minutes": None,
                "frequency_days_per_week": 4,
            }
        },
    }
    matches_short = evaluate_layer1(short_break)
    short_ids = {m["rule_id"] for m in matches_short}
    assert "L1-RTT-0001" not in short_ids
    assert "L1-RTT-0002a" not in short_ids
    assert "L1-RTT-0002d" not in short_ids


def test_chatgpt_3week_break_rejects_moderate_track_gate():
    """3-week break without moderate track → L1-RTT-0008 rejected."""
    plan = _load_json(THREE_WEEK_BREAK_PATH)
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}
    assert "L1-RTT-0008" in rule_ids
    assert "L1-RTT-0001" not in rule_ids  # exclusive of long-inactivity gate

    result = run_audit(plan, lang="en", skip_layer3=True)
    assert "L1-RTT-0008" in result["matched_rules"]
    assert "L1-RTT-0001" not in result["matched_rules"]
    assert result["verdict"] == "rejected"
    assert any(
        "returning athlete" in e.lower() or "fit rule" in e.lower() or "break of" in e.lower()
        for e in result["explanations"]
    )


def test_long_and_moderate_gates_are_week_exclusive():
    """>=4 weeks → 0001 only; 2-to-<4 weeks → 0008 only; <2 → neither."""
    long_plan = {
        "target_population": "healthy_adult_18plus",
        "goal": "general",
        "inactivity_duration_weeks": 4,
        "plan_follows_moderate_return_track": False,
        "sessions_per_week": 3,
        "sets_per_exercise": 3,
    }
    long_ids = {m["rule_id"] for m in evaluate_layer1(long_plan)}
    assert "L1-RTT-0001" in long_ids
    assert "L1-RTT-0008" not in long_ids
    assert evaluate_long_inactivity_track_compliance(long_plan) == "violated"

    moderate_plan = {
        **long_plan,
        "inactivity_duration_weeks": 2,
        "plan_follows_moderate_return_track": False,
    }
    mod_ids = {m["rule_id"] for m in evaluate_layer1(moderate_plan)}
    assert "L1-RTT-0008" in mod_ids
    assert "L1-RTT-0001" not in mod_ids
    assert evaluate_long_inactivity_track_compliance(moderate_plan) == "not_applicable"

    short_plan = {**long_plan, "inactivity_duration_weeks": 1}
    short_ids = {m["rule_id"] for m in evaluate_layer1(short_plan)}
    assert "L1-RTT-0001" not in short_ids
    assert "L1-RTT-0008" not in short_ids


def test_long_inactivity_insufficient_data_is_not_evaluable():
    """Long layoff with no Table 9 metrics → rule skipped (not flagged, not pass)."""
    plan = {
        "target_population": "healthy_adult_18plus",
        "goal": "general",
        "inactivity_duration_weeks": 20,
    }
    assert evaluate_long_inactivity_track_compliance(plan) == "insufficient_data"
    matches = evaluate_layer1(plan)
    assert matches == []
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "insufficient_data"
    assert result["matched_rules"] == []
    assert result["checked_facts"] == []


def test_synthetic_pass_track_compliance_followed():
    case = _load_json(GROUND_TRUTH_DIR / "case_synthetic_pass.json")
    plan = case["expected_plan"]
    assert evaluate_long_inactivity_track_compliance(plan) == "followed"
    assert case["expected_track_compliance"] == "followed"
    matches = evaluate_layer1(plan)
    assert "L1-RTT-0001" not in {m["rule_id"] for m in matches}
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "pass"
    fact_ids = {f["rule_id"] for f in result["checked_facts"]}
    assert "L1-RTT-0001" in fact_ids


def test_synthetic_fail_track_compliance_violated():
    case = _load_json(GROUND_TRUTH_DIR / "case_synthetic_fail.json")
    plan = case["expected_plan"]
    assert evaluate_long_inactivity_track_compliance(plan) == "violated"
    assert case["expected_track_compliance"] == "violated"
    matches = evaluate_layer1(plan)
    assert "L1-RTT-0001" in {m["rule_id"] for m in matches}
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "rejected"
    assert "L1-RTT-0001" in result["matched_rules"]


def test_chatgpt_zero_rest_days_flags_ecss_0002():
    """Zero full rest days → L1-ECSS-0002 flagged (not rejected)."""
    plan = _load_json(ZERO_REST_PLAN_PATH)
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}
    assert "L1-ECSS-0002" in rule_ids

    result = run_audit(plan, lang="en", skip_layer3=True)
    assert "L1-ECSS-0002" in result["matched_rules"]
    assert result["verdict"] == "flagged"
    assert any(
        "at least one full passive rest day" in e or "full rest days per week" in e
        for e in result["explanations"]
    )


def test_missing_rest_days_per_week_skips_ecss_0002():
    """Unset rest_days_per_week must not fire L1-ECSS-0002 (null → skip)."""
    layoff = _load_json(CHATGPT_PLAN_PATH)
    assert "rest_days_per_week" not in layoff
    layoff_ids = {m["rule_id"] for m in evaluate_layer1(layoff)}
    assert "L1-ECSS-0002" not in layoff_ids

    corrected = _load_json(CORRECTED_PLAN_PATH)
    assert "rest_days_per_week" not in corrected
    corrected_result = run_audit(corrected, lang="en", skip_layer3=True)
    assert "L1-ECSS-0002" not in corrected_result["matched_rules"]
    fact_ids = {f["rule_id"] for f in corrected_result.get("checked_facts") or []}
    assert "L1-ECSS-0002" not in fact_ids


def test_existing_sample_plans_do_not_false_fire_tier3_rules():
    """Tier3 fields absent on existing samples → 0003–0006 / ECSS-0001 must not fire."""
    tier3 = {
        "L1-RTT-0003",
        "L1-RTT-0004",
        "L1-RTT-0005",
        "L1-RTT-0006",
        "L1-ECSS-0001",
    }
    for path in (
        CHATGPT_PLAN_PATH,
        CORRECTED_PLAN_PATH,
        THREE_WEEK_BREAK_PATH,
        ZERO_REST_PLAN_PATH,
    ):
        plan = _load_json(path)
        hits = {m["rule_id"] for m in evaluate_layer1(plan)} & tier3
        assert hits == set(), f"{path.name} unexpectedly matched {hits}"


def test_work_rest_ratio_denominator_resolves_week1_min_denominator():
    """L1-RTT-0004: week 1 uses week_1_min_denominator (4); denominator 2 → fire."""
    from src.layer1_engine import _enrich_week_dependent_params

    plan = {
        "target_population": "healthy_adult_18plus",
        "goal": "general",
        "weeks_since_return": 1,
        "work_rest_ratio_denominator": 2,
        "sessions_per_week": 2,
        "sets_per_exercise": 2,
        "program_mandates_training_to_failure": False,
    }
    enriched = _enrich_week_dependent_params(
        {"week_1_min_denominator": 4, "week_2_min_denominator": 3},
        plan,
    )
    assert enriched["min_denominator_for_week"] == 4

    matches = evaluate_layer1(plan)
    by_id = {m["rule_id"]: m for m in matches}
    assert "L1-RTT-0004" in by_id
    params = by_id["L1-RTT-0004"]["matched_parameters"]
    assert params["observed_value"] == 2
    assert params["min_denominator_for_week"] == 4


def test_tier3_intentional_violations_fire_compound_and_numeric_rules():
    """Intentional long-inactivity plan that trips RTT-0003–0006 and ECSS-0001."""
    plan = {
        "target_population": "healthy_adult_18plus",
        "goal": "general",
        "inactivity_duration_weeks": 26,
        "weeks_since_return": 1,
        "sessions_per_week": 2,
        "sets_per_exercise": 2,
        # L1-RTT-0003
        "plan_uses_FIT_rule_IRV_as_primary_constraint": True,
        # L1-RTT-0004 (1:2 rest < week-1 minimum 1:4)
        "work_rest_ratio_denominator": 2,
        # L1-RTT-0005 (return + failure within first 2 weeks)
        "program_mandates_training_to_failure": True,
        "eccentric_emphasis_flagged": False,
        "novel_high_volume_circuit": False,
        # L1-RTT-0006
        "plan_output_lacks_medical_clearance_recommendation": True,
        # L1-ECSS-0001
        "user_reports_persistent_unexplained_fatigue_or_performance_decline_weeks": 6,
        "plan_recommends_continuing_programmed_progression_without_reevaluation": True,
    }
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}
    for rid in (
        "L1-RTT-0003",
        "L1-RTT-0004",
        "L1-RTT-0005",
        "L1-RTT-0006",
        "L1-ECSS-0001",
    ):
        assert rid in rule_ids, f"expected {rid} to fire; got {sorted(rule_ids)}"

    result = run_audit(plan, lang="en", skip_layer3=True)
    for rid in (
        "L1-RTT-0003",
        "L1-RTT-0004",
        "L1-RTT-0005",
        "L1-RTT-0006",
        "L1-ECSS-0001",
    ):
        assert rid in result["matched_rules"]
    # L1-RTT-0005 action is reject → overall rejected
    assert result["verdict"] == "rejected"


def test_compound_boolean_unevaluable_when_any_or_branch_field_is_null():
    """OR-group fields must all be present; one null → skip L1-RTT-0005."""
    plan = {
        "target_population": "healthy_adult_18plus",
        "goal": "general",
        "inactivity_duration_weeks": 26,
        "weeks_since_return": 1,
        "program_mandates_training_to_failure": True,
        "eccentric_emphasis_flagged": True,
        # novel_high_volume_circuit intentionally omitted (null)
    }
    rule_ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert "L1-RTT-0005" not in rule_ids
