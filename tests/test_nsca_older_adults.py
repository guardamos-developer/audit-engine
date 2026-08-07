"""Tests for NSCA older-adult Table 1 rules and exclusive population routing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
from src.layer1_engine import (  # noqa: E402
    effective_target_population,
    evaluate_layer1,
    evaluate_layer1_detailed,
)
from src.plan_extractor import (  # noqa: E402
    EXCLUSION_FLAG_FIELDS,
    _PLAN_FIELD_SPECS,
    build_extraction_json_schema,
)

NSCA_RULE_IDS = {
    "L1-RT-NSCA-0001",
    "L1-RT-NSCA-0002",
    "L1-RT-NSCA-0003",
    "L1-RT-NSCA-0004",
    "L1-RT-NSCA-0005",
}

# ACSM rules that conflict numerically with NSCA Table 1 for the same inputs.
CONFLICTING_ACSM_IDS = {"L1-RT-0003", "L1-RT-0004", "L1-RT-0006"}


def _older_adult_strength_plan(**overrides):
    plan = {
        "age_years": 68,
        "goal": "strength",
        "sets_per_exercise": 5,  # outside NSCA 1-3 → L1-RT-NSCA-0002
        "repetitions_per_set": 20,  # outside 6-15 → L1-RT-NSCA-0003
        "load_percent_1RM": 50,  # outside 70-85 → L1-RT-NSCA-0004
        "injury_present": False,
        "post_surgical": False,
        "pain_present": False,
        "minor": False,
        "pregnant": False,
        "frailty_present": False,
        "uncontrolled_hypertension": False,
        "unstable_cardiovascular_disease": False,
        "true_beginner_first_weeks": False,
        "cardiovascular_disease_present": False,
        "osteoporosis_present": False,
        "program_mandates_training_to_failure": False,
        "program_mandates_complex_periodization_as_required": False,
        "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
        "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
    }
    plan.update(overrides)
    return plan


def test_schema_includes_nsca_exclusion_and_age_fields():
    for name in (
        "age_years",
        "repetitions_per_set",
        "frailty_present",
        "uncontrolled_hypertension",
        "unstable_cardiovascular_disease",
    ):
        assert name in _PLAN_FIELD_SPECS
    schema = build_extraction_json_schema()
    for name in (
        "frailty_present",
        "uncontrolled_hypertension",
        "unstable_cardiovascular_disease",
    ):
        assert name in EXCLUSION_FLAG_FIELDS
        assert name in schema["properties"]


def test_effective_target_population_age_routing():
    assert effective_target_population({"age_years": 65}) == "older_adult_healthy"
    assert effective_target_population({"age_years": 72}) == "older_adult_healthy"
    assert effective_target_population({"age_years": 64}) == "healthy_adult_18plus"
    assert effective_target_population({"age_years": 40}) == "healthy_adult_18plus"
    # Age unknown → default general-adult (NSCA does not apply).
    assert effective_target_population({}) == "healthy_adult_18plus"
    assert (
        effective_target_population({"target_population": "older_adult_healthy"})
        == "older_adult_healthy"
    )
    # Age overrides a stale general-adult tag.
    assert (
        effective_target_population(
            {"age_years": 70, "target_population": "healthy_adult_18plus"}
        )
        == "older_adult_healthy"
    )


def test_older_adult_fires_nsca_not_conflicting_acsm():
    plan = _older_adult_strength_plan()
    matches = evaluate_layer1(plan)
    ids = {m["rule_id"] for m in matches}
    assert "L1-RT-NSCA-0002" in ids
    assert "L1-RT-NSCA-0003" in ids
    assert "L1-RT-NSCA-0004" in ids
    assert ids.isdisjoint(CONFLICTING_ACSM_IDS)
    assert "L1-RT-0001" not in ids

    detailed = evaluate_layer1_detailed(plan)
    applicable_ids = {a["rule_id"] for a in detailed["applicable"]}
    assert "L1-RT-NSCA-0001" in applicable_ids
    assert "L1-RT-0001" not in applicable_ids


def test_older_adult_power_fires_nsca_0005_not_acsm_0006():
    plan = _older_adult_strength_plan(
        goal="power",
        load_percent_1RM=30,  # in ACSM 30-70 but outside NSCA 40-60
        sets_per_exercise=2,
        repetitions_per_set=10,
    )
    ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert "L1-RT-NSCA-0005" in ids
    assert "L1-RT-0006" not in ids


def test_general_adult_fires_acsm_not_nsca():
    plan = {
        "age_years": 35,
        "target_population": "healthy_adult_18plus",
        "goal": "strength",
        "sessions_per_week": 3,
        "sets_per_exercise": 1,  # ACSM L1-RT-0003
        "load_percent_1RM": 50,  # ACSM L1-RT-0004
        "injury_present": False,
        "post_surgical": False,
        "pain_present": False,
        "minor": False,
        "pregnant": False,
        "true_beginner_first_weeks": False,
        "program_mandates_training_to_failure": False,
        "program_mandates_complex_periodization_as_required": False,
        "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
        "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
    }
    ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert "L1-RT-0003" in ids
    assert "L1-RT-0004" in ids
    assert ids.isdisjoint(NSCA_RULE_IDS)


def test_null_age_defaults_to_acsm_not_nsca():
    plan = {
        "goal": "strength",
        "sets_per_exercise": 1,
        "load_percent_1RM": 50,
        "injury_present": False,
        "post_surgical": False,
        "pain_present": False,
        "minor": False,
        "pregnant": False,
        "true_beginner_first_weeks": False,
        "program_mandates_training_to_failure": False,
        "program_mandates_complex_periodization_as_required": False,
        "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
        "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
    }
    assert effective_target_population(plan) == "healthy_adult_18plus"
    ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert "L1-RT-0003" in ids
    assert ids.isdisjoint(NSCA_RULE_IDS)


def test_frailty_routes_nsca_gate_to_reject():
    plan = _older_adult_strength_plan(frailty_present=True)
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "rejected"
    assert "L1-RT-NSCA-0001" in result["matched_rules"]


def test_uncontrolled_hypertension_routes_nsca_gate_to_reject():
    plan = _older_adult_strength_plan(uncontrolled_hypertension=True)
    matches = evaluate_layer1(plan)
    assert matches and matches[0]["rule_id"] == "L1-RT-NSCA-0001"
    assert matches[0]["action"] == "route_to_layer2_or_reject"
