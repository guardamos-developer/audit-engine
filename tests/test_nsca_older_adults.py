"""Tests for NSCA older-adult Table 1 rules and exclusive population routing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
from src.layer1_engine import (  # noqa: E402
    apply_deterministic_age_derived_flags,
    effective_target_population,
    evaluate_layer1,
    evaluate_layer1_detailed,
)
from src.plan_extractor import (  # noqa: E402
    EXCLUSION_FLAG_FIELDS,
    LENIENT_EXCLUSION_FIELDS,
    STRICT_ACCOMMODATION_FIELDS,
    _PLAN_FIELD_SPECS,
    _SYSTEM_PROMPT,
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
        "stated_age_category",
        "repetitions_per_set",
        "frailty_present",
        "uncontrolled_hypertension",
        "unstable_cardiovascular_disease",
    ):
        assert name in _PLAN_FIELD_SPECS
    schema = build_extraction_json_schema()
    value_schema = schema["properties"]["stated_age_category"]["properties"]["value"]
    enums: list[str] = []
    for branch in value_schema.get("anyOf") or []:
        if isinstance(branch, dict) and "enum" in branch:
            enums.extend(branch["enum"])
    assert enums == ["minor", "older_adult", "adult"]
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
    # Qualitative stated_age_category when age_years is null.
    assert (
        effective_target_population({"stated_age_category": "older_adult"})
        == "older_adult_healthy"
    )
    assert (
        effective_target_population({"stated_age_category": "minor"})
        == "healthy_adult_18plus"
    )
    # Numeric age wins over stated_age_category.
    assert (
        effective_target_population(
            {"age_years": 40, "stated_age_category": "older_adult"}
        )
        == "healthy_adult_18plus"
    )


def test_age_under_18_forces_minor_even_when_extractor_left_null():
    """Numeric age < 18 deterministically sets minor=True (Zone B)."""
    forced = apply_deterministic_age_derived_flags(
        {"age_years": 14, "minor": None, "goal": "general"}
    )
    assert forced["minor"] is True
    forced_false = apply_deterministic_age_derived_flags(
        {"age_years": 15, "minor": False}
    )
    assert forced_false["minor"] is True
    # Adult ages are not forced either way.
    adult = apply_deterministic_age_derived_flags({"age_years": 40, "minor": None})
    assert adult.get("minor") is None
    # Qualitative underage self-ID without numeric age.
    qualitative = apply_deterministic_age_derived_flags(
        {"stated_age_category": "minor", "minor": None}
    )
    assert qualitative["minor"] is True

    # End-to-end: age 14 with minor null must hit ACSM population gate.
    plan = {
        "age_years": 14,
        "minor": None,
        "goal": "strength",
        "sets_per_exercise": 3,
        "load_percent_1RM": 70,
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
    matches = evaluate_layer1(plan)
    assert matches and matches[0]["rule_id"] == "L1-RT-0001"
    assert matches[0]["action"] == "route_to_layer2_or_reject"


def test_stated_age_category_older_adult_routes_to_nsca():
    plan = _older_adult_strength_plan()
    plan.pop("age_years", None)
    plan["stated_age_category"] = "older_adult"
    plan["sets_per_exercise"] = 4  # outside NSCA 1-3
    matches = evaluate_layer1(plan)
    ids = {m["rule_id"] for m in matches}
    assert "L1-RT-NSCA-0002" in ids
    assert "L1-RT-0001" not in ids


def test_lenient_exclusion_fields_opposite_strict_accommodation():
    assert LENIENT_EXCLUSION_FIELDS.isdisjoint(STRICT_ACCOMMODATION_FIELDS)
    assert "minor" in LENIENT_EXCLUSION_FIELDS
    assert "frailty_present" in LENIENT_EXCLUSION_FIELDS
    assert "injury_present" in LENIENT_EXCLUSION_FIELDS
    prompt = _SYSTEM_PROMPT
    assert "LENIENT exclusion" in prompt
    assert "少し体が弱っている" in prompt or "frail" in prompt
    assert "opposite of item 5" in prompt or "Contrast with item 5" in prompt


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


# ACSM L1-RT-0007–0010 ↔ NSCA L1-RT-NSCA-0012–0015 population exclusivity
ACSM_CAUTION_IDS = {"L1-RT-0007", "L1-RT-0008", "L1-RT-0009", "L1-RT-0010"}
NSCA_CAUTION_IDS = {
    "L1-RT-NSCA-0012",
    "L1-RT-NSCA-0013",
    "L1-RT-NSCA-0014",
    "L1-RT-NSCA-0015",
}

# (nsca_id, acsm_id, field overrides that make that caution fire)
_CAUTION_PAIR_CASES = [
    (
        "L1-RT-NSCA-0012",
        "L1-RT-0007",
        {"program_mandates_training_to_failure": True},
    ),
    (
        "L1-RT-NSCA-0013",
        "L1-RT-0008",
        {
            "output_claims_RT_is_unsafe_for_older_adult_without_specific_contraindication": True,
            # ACSM 0008 uses the healthy_adult field; set both so under-65 can fire ACSM.
            "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": True,
        },
    ),
    (
        "L1-RT-NSCA-0014",
        "L1-RT-0009",
        {"program_mandates_complex_periodization_as_required": True},
    ),
    (
        "L1-RT-NSCA-0015",
        "L1-RT-0010",
        {"output_recommends_zero_resistance_training_for_muscle_function_goal": True},
    ),
]


def _in_range_older_adult(**overrides):
    """Older adult with Table 1 metrics in-range (avoid numeric NSCA noise)."""
    plan = _older_adult_strength_plan(
        sets_per_exercise=2,
        repetitions_per_set=10,
        load_percent_1RM=75,
        program_mandates_training_to_failure=False,
        program_mandates_complex_periodization_as_required=False,
        output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication=False,
        output_claims_RT_is_unsafe_for_older_adult_without_specific_contraindication=False,
        output_recommends_zero_resistance_training_for_muscle_function_goal=False,
    )
    plan.update(overrides)
    return plan


def _in_range_general_adult(**overrides):
    plan = {
        "age_years": 40,
        "goal": "strength",
        "sessions_per_week": 3,
        "sets_per_exercise": 3,
        "load_percent_1RM": 80,
        "injury_present": False,
        "post_surgical": False,
        "pain_present": False,
        "minor": False,
        "pregnant": False,
        "true_beginner_first_weeks": False,
        "program_mandates_training_to_failure": False,
        "program_mandates_complex_periodization_as_required": False,
        "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
        "output_claims_RT_is_unsafe_for_older_adult_without_specific_contraindication": False,
        "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
    }
    plan.update(overrides)
    return plan


def test_older_adult_fires_nsca_caution_mirrors_not_acsm():
    for nsca_id, acsm_id, field_overrides in _CAUTION_PAIR_CASES:
        plan = _in_range_older_adult(**field_overrides)
        ids = {m["rule_id"] for m in evaluate_layer1(plan)}
        assert nsca_id in ids, (nsca_id, ids)
        assert acsm_id not in ids, (acsm_id, ids)
        assert ids.isdisjoint(ACSM_CAUTION_IDS - {acsm_id}) or acsm_id not in ids


def test_general_adult_fires_acsm_caution_not_nsca_mirrors():
    for nsca_id, acsm_id, field_overrides in _CAUTION_PAIR_CASES:
        plan = _in_range_general_adult(**field_overrides)
        ids = {m["rule_id"] for m in evaluate_layer1(plan)}
        assert acsm_id in ids, (acsm_id, ids)
        assert nsca_id not in ids, (nsca_id, ids)
        assert ids.isdisjoint(NSCA_CAUTION_IDS)


def test_schema_includes_older_adult_unsafe_rt_claim_field():
    assert (
        "output_claims_RT_is_unsafe_for_older_adult_without_specific_contraindication"
        in _PLAN_FIELD_SPECS
    )
    schema = build_extraction_json_schema()
    assert (
        "output_claims_RT_is_unsafe_for_older_adult_without_specific_contraindication"
        in schema["properties"]
    )


def test_ruleset_notes_include_population_exclusivity_gap_fix():
    import json

    path = ROOT / "rules" / "layer1_rules_nsca_older_adults_v1.json"
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    note = raw["ruleset_notes"]["population_exclusivity_gap_fix"]
    assert "L1-RT-NSCA-0012" in note
    assert "L1-RT-0007" in note
    assert "L1-RT-NSCA-0016" in note
    assert "L1-RT-0002" in note


def test_older_adult_low_frequency_flags_nsca_0016_not_acsm_0002():
    plan = _in_range_older_adult(sessions_per_week=1)
    ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert "L1-RT-NSCA-0016" in ids
    assert "L1-RT-0002" not in ids


def test_general_adult_low_frequency_flags_acsm_0002_not_nsca_0016():
    plan = _in_range_general_adult(sessions_per_week=1)
    ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert "L1-RT-0002" in ids
    assert "L1-RT-NSCA-0016" not in ids


def test_older_adult_adequate_frequency_does_not_flag_0016():
    plan = _in_range_older_adult(sessions_per_week=3)
    ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert "L1-RT-NSCA-0016" not in ids
    audit = run_audit(plan, lang="en", skip_layer3=True)
    assert audit["verdict"] == "pass"
    checked = {c["rule_id"] for c in audit.get("checked_facts") or []}
    assert "L1-RT-NSCA-0016" in checked
