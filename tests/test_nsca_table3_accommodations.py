"""Tests for NSCA Table 3 accommodation_check rules (L1-RT-NSCA-0006–0011)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
from src.explanation import render_explanation  # noqa: E402
from src.layer1_engine import (  # noqa: E402
    ACTIVE_RULE_IDS,
    evaluate_condition,
    evaluate_layer1,
    evaluate_layer1_detailed,
    evaluate_long_inactivity_track_compliance,
    load_merged_rulesets,
)
from src.plan_extractor import (  # noqa: E402
    EQUIPMENT_MODALITY_ENUM,
    LENIENT_EXCLUSION_FIELDS,
    STRICT_ACCOMMODATION_FIELDS,
    _PLAN_FIELD_SPECS,
    _SYSTEM_PROMPT,
    _materialize_plan_and_evidence,
    build_extraction_json_schema,
)

TABLE3_RULE_IDS = {
    "L1-RT-NSCA-0006",
    "L1-RT-NSCA-0007",
    "L1-RT-NSCA-0008",
    "L1-RT-NSCA-0009",
    "L1-RT-NSCA-0010",
    "L1-RT-NSCA-0011",
}

# (rule_id, condition_field overrides, accommodation_field, accommodation false/true values)
BOOLEAN_RULE_CASES = [
    (
        "L1-RT-NSCA-0006",
        {"mobility_limitation_present": True},
        "plan_offers_seated_position_option",
    ),
    (
        "L1-RT-NSCA-0007",
        {"cognitive_impairment_present": True},
        "plan_uses_simple_exercise_selection_with_instruction",
    ),
    (
        "L1-RT-NSCA-0008",
        {"diabetes_present": True},
        "blood_glucose_monitoring_mentioned",
    ),
    (
        "L1-RT-NSCA-0009",
        {"osteoporosis_present": True},
        "spinal_flexion_or_twisting_caution_mentioned",
    ),
    (
        "L1-RT-NSCA-0010",
        {"joint_pain_or_limited_rom_present": True},
        "rom_restricted_training_mentioned",
    ),
]


def _base_older_adult(**overrides):
    """Healthy older adult with Table 1 metrics in-range (avoid Table 1 noise)."""
    plan = {
        "age_years": 70,
        "goal": "strength",
        "sets_per_exercise": 2,
        "repetitions_per_set": 10,
        "load_percent_1RM": 75,
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
        "mobility_limitation_present": False,
        "cognitive_impairment_present": False,
        "diabetes_present": False,
        "joint_pain_or_limited_rom_present": False,
        "poor_vision_or_balance_present": False,
        "fall_risk_present": False,
        "low_back_pain_present": False,
        "plan_offers_seated_position_option": None,
        "plan_uses_simple_exercise_selection_with_instruction": None,
        "blood_glucose_monitoring_mentioned": None,
        "spinal_flexion_or_twisting_caution_mentioned": None,
        "rom_restricted_training_mentioned": None,
        "equipment_modality": None,
        "program_mandates_training_to_failure": False,
        "program_mandates_complex_periodization_as_required": False,
        "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
        "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
    }
    plan.update(overrides)
    return plan


def _match_by_id(matches, rule_id: str):
    return next((m for m in matches if m["rule_id"] == rule_id), None)


def _applicable_by_id(applicable, rule_id: str):
    return next((a for a in applicable if a["rule_id"] == rule_id), None)


def test_table3_rules_are_active_and_loaded():
    assert TABLE3_RULE_IDS <= ACTIVE_RULE_IDS
    data = load_merged_rulesets()
    loaded = {r["rule_id"] for r in data["rules"]}
    assert TABLE3_RULE_IDS <= loaded
    assert any(
        r.get("rule_id") == "L1-RT-NSCA-0006"
        and (r.get("condition") or {}).get("type") == "accommodation_check"
        for r in data["rules"]
    )


def test_ruleset_notes_include_extraction_confidence_asymmetry():
    import json

    path = ROOT / "rules" / "layer1_rules_nsca_older_adults_v1.json"
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    note = raw["ruleset_notes"]["extraction_confidence_asymmetry"]
    assert "null-if-uncertain" in note
    assert "false positive" in note.lower() or "incorrectly marking" in note


def test_schema_and_prompt_cover_table3_strict_fields():
    for name in (
        "mobility_limitation_present",
        "plan_offers_seated_position_option",
        "cognitive_impairment_present",
        "plan_uses_simple_exercise_selection_with_instruction",
        "diabetes_present",
        "blood_glucose_monitoring_mentioned",
        "spinal_flexion_or_twisting_caution_mentioned",
        "joint_pain_or_limited_rom_present",
        "rom_restricted_training_mentioned",
        "poor_vision_or_balance_present",
        "fall_risk_present",
        "low_back_pain_present",
        "equipment_modality",
    ):
        assert name in _PLAN_FIELD_SPECS

    assert STRICT_ACCOMMODATION_FIELDS == frozenset(
        {
            "plan_offers_seated_position_option",
            "plan_uses_simple_exercise_selection_with_instruction",
            "blood_glucose_monitoring_mentioned",
            "spinal_flexion_or_twisting_caution_mentioned",
            "rom_restricted_training_mentioned",
            "equipment_modality",
        }
    )
    assert LENIENT_EXCLUSION_FIELDS.isdisjoint(STRICT_ACCOMMODATION_FIELDS)
    schema = build_extraction_json_schema()
    modality = schema["properties"]["equipment_modality"]["properties"]["value"]
    assert modality["anyOf"][0]["enum"] == list(EQUIPMENT_MODALITY_ENUM)

    prompt = _SYSTEM_PROMPT
    assert "STRICT accommodation" in prompt or "null-if-uncertain" in prompt
    assert "LENIENT exclusion" in prompt
    assert "blood_glucose_monitoring_mentioned" in prompt
    assert "equipment_modality" in prompt
    assert "false positive" in prompt.lower() or "incorrectly pass" in prompt.lower()


def test_materialize_rejects_guessed_accommodation_without_quote():
    """Strict fields: value without evidence_quote → null (never keep a bare true)."""
    raw = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    raw["blood_glucose_monitoring_mentioned"] = {
        "value": True,
        "evidence_quote": None,
    }
    raw["equipment_modality"] = {
        "value": "machine_preferred_or_only",
        "evidence_quote": None,
    }
    plan, _ = _materialize_plan_and_evidence(raw)
    assert plan["blood_glucose_monitoring_mentioned"] is None
    assert plan["equipment_modality"] is None


def test_lenient_exclusion_asymmetry_does_not_relax_strict_accommodations():
    """Lenient exclusion policy must not change STRICT accommodation null-if-uncertain.

    Directly asserts the asymmetry: every STRICT field still drops bare true
    values without evidence_quote, while remaining disjoint from LENIENT set.
    """
    assert LENIENT_EXCLUSION_FIELDS.isdisjoint(STRICT_ACCOMMODATION_FIELDS)
    raw = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    for field in STRICT_ACCOMMODATION_FIELDS:
        if field == "equipment_modality":
            raw[field] = {
                "value": "machine_preferred_or_only",
                "evidence_quote": None,
            }
        else:
            raw[field] = {"value": True, "evidence_quote": None}
    plan, _ = _materialize_plan_and_evidence(raw)
    for field in STRICT_ACCOMMODATION_FIELDS:
        assert plan[field] is None, field


def test_materialize_rejects_invalid_equipment_modality_enum():
    raw = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    raw["equipment_modality"] = {
        "value": "dumbbells_maybe",
        "evidence_quote": "uses dumbbells",
    }
    plan, _ = _materialize_plan_and_evidence(raw)
    assert plan["equipment_modality"] is None


def test_boolean_accommodation_rules_flagged_pass_insufficient():
    for rule_id, cond_overrides, acc_field in BOOLEAN_RULE_CASES:
        flagged_plan = _base_older_adult(**cond_overrides, **{acc_field: False})
        match = _match_by_id(evaluate_layer1(flagged_plan), rule_id)
        assert match is not None, rule_id
        assert match["explanation_side"] == "flagged"
        assert match["action"] == "flag_caution"

        pass_plan = _base_older_adult(**cond_overrides, **{acc_field: True})
        detailed = evaluate_layer1_detailed(pass_plan)
        assert _match_by_id(detailed["matched"], rule_id) is None, rule_id
        app = _applicable_by_id(detailed["applicable"], rule_id)
        assert app is not None, rule_id
        assert app["violated"] is False
        assert app["skip_pass_fact"] is False

        insuff_plan = _base_older_adult(**cond_overrides, **{acc_field: None})
        match = _match_by_id(evaluate_layer1(insuff_plan), rule_id)
        assert match is not None, rule_id
        assert match["explanation_side"] == "insufficient_data"
        assert match["action"] == "flag_caution"
        text = render_explanation(
            {"reason_template": match["reason_template"]},
            match["matched_parameters"],
            side="insufficient_data",
        )
        assert text


def test_l1_rt_nsca_0011_flagged_pass_insufficient():
    cond = {"fall_risk_present": True}
    flagged = _base_older_adult(**cond, equipment_modality="free_weight_only")
    match = _match_by_id(evaluate_layer1(flagged), "L1-RT-NSCA-0011")
    assert match is not None
    assert match["explanation_side"] == "flagged"

    passed = _base_older_adult(**cond, equipment_modality="machine_preferred_or_only")
    detailed = evaluate_layer1_detailed(passed)
    assert _match_by_id(detailed["matched"], "L1-RT-NSCA-0011") is None
    app = _applicable_by_id(detailed["applicable"], "L1-RT-NSCA-0011")
    assert app is not None and app["violated"] is False and not app["skip_pass_fact"]

    insuff = _base_older_adult(**cond, equipment_modality=None)
    match = _match_by_id(evaluate_layer1(insuff), "L1-RT-NSCA-0011")
    assert match is not None
    assert match["explanation_side"] == "insufficient_data"


def test_condition_fields_all_false_skips_rule():
    plan = _base_older_adult(
        mobility_limitation_present=False,
        plan_offers_seated_position_option=False,
    )
    detailed = evaluate_layer1_detailed(plan)
    assert _match_by_id(detailed["matched"], "L1-RT-NSCA-0006") is None
    assert _applicable_by_id(detailed["applicable"], "L1-RT-NSCA-0006") is None


def test_condition_fields_all_null_skips_rule():
    plan = _base_older_adult(mobility_limitation_present=None)
    detailed = evaluate_layer1_detailed(plan)
    assert _match_by_id(detailed["matched"], "L1-RT-NSCA-0006") is None
    assert _applicable_by_id(detailed["applicable"], "L1-RT-NSCA-0006") is None


def test_l1_rt_nsca_0011_or_condition_any_single_true():
    for field in (
        "poor_vision_or_balance_present",
        "fall_risk_present",
        "low_back_pain_present",
    ):
        overrides = {
            "poor_vision_or_balance_present": False,
            "fall_risk_present": False,
            "low_back_pain_present": False,
            "equipment_modality": "free_weight_only",
        }
        overrides[field] = True
        plan = _base_older_adult(**overrides)
        match = _match_by_id(evaluate_layer1(plan), "L1-RT-NSCA-0011")
        assert match is not None, field


def test_frailty_excludes_all_table3_rules():
    plan = _base_older_adult(
        frailty_present=True,
        mobility_limitation_present=True,
        plan_offers_seated_position_option=False,
        cognitive_impairment_present=True,
        plan_uses_simple_exercise_selection_with_instruction=False,
        diabetes_present=True,
        blood_glucose_monitoring_mentioned=False,
        osteoporosis_present=True,
        spinal_flexion_or_twisting_caution_mentioned=False,
        joint_pain_or_limited_rom_present=True,
        rom_restricted_training_mentioned=False,
        fall_risk_present=True,
        equipment_modality="free_weight_only",
    )
    detailed = evaluate_layer1_detailed(plan)
    matched_ids = {m["rule_id"] for m in detailed["matched"]}
    applicable_ids = {a["rule_id"] for a in detailed["applicable"]}
    assert matched_ids.isdisjoint(TABLE3_RULE_IDS)
    assert applicable_ids.isdisjoint(TABLE3_RULE_IDS)
    # Population gate short-circuits on frailty before other NSCA rules run.
    assert "L1-RT-NSCA-0001" in matched_ids


def test_under_65_does_not_fire_table3_rules():
    plan = _base_older_adult(
        age_years=40,
        diabetes_present=True,
        blood_glucose_monitoring_mentioned=False,
        mobility_limitation_present=True,
        plan_offers_seated_position_option=False,
    )
    ids = {m["rule_id"] for m in evaluate_layer1(plan)}
    assert ids.isdisjoint(TABLE3_RULE_IDS)
    detailed = evaluate_layer1_detailed(plan)
    assert not {
        a["rule_id"] for a in detailed["applicable"] if a["rule_id"] in TABLE3_RULE_IDS
    }


def test_accommodation_check_evaluated_in_other_rules_not_as_context_gate():
    """accommodation_check must not be classified as context_gate."""
    from src.layer1_engine import _is_context_gate

    data = load_merged_rulesets()
    for rule in data["rules"]:
        if rule.get("rule_id") in TABLE3_RULE_IDS:
            assert not _is_context_gate(rule)
            assert (rule.get("condition") or {}).get("type") == "accommodation_check"


def test_evaluate_condition_accommodation_check_four_branches():
    condition = {
        "type": "accommodation_check",
        "parameters": {
            "condition_fields": ["diabetes_present"],
            "accommodation_field": "blood_glucose_monitoring_mentioned",
            "accommodation_type": "boolean",
        },
    }
    rule = {"rule_id": "x"}
    hit, _ = evaluate_condition(
        condition, {"diabetes_present": False, "blood_glucose_monitoring_mentioned": False}, rule
    )
    assert hit is False

    hit, params = evaluate_condition(
        condition, {"diabetes_present": True, "blood_glucose_monitoring_mentioned": None}, rule
    )
    assert hit is True
    assert params["explanation_side"] == "insufficient_data"

    hit, params = evaluate_condition(
        condition, {"diabetes_present": True, "blood_glucose_monitoring_mentioned": False}, rule
    )
    assert hit is True
    assert params.get("explanation_side", "flagged") == "flagged"

    hit, _ = evaluate_condition(
        condition, {"diabetes_present": True, "blood_glucose_monitoring_mentioned": True}, rule
    )
    assert hit is False


def test_regression_long_inactivity_track_compliance_unchanged():
    """L1-RTT-0001 long_inactivity_track_compliance semantics stay intact."""
    import json

    case = json.loads(
        (ROOT / "tests" / "extraction_ground_truth" / "case_synthetic_fail.json").read_text(
            encoding="utf-8"
        )
    )
    plan = case["expected_plan"]
    assert evaluate_long_inactivity_track_compliance(plan) == "violated"
    assert "L1-RTT-0001" in {m["rule_id"] for m in evaluate_layer1(plan)}
    audit = run_audit(plan, lang="en", skip_layer3=True)
    assert "L1-RTT-0001" in audit["matched_rules"]
    assert audit["verdict"] == "rejected"

    pass_case = json.loads(
        (ROOT / "tests" / "extraction_ground_truth" / "case_synthetic_pass.json").read_text(
            encoding="utf-8"
        )
    )
    pass_plan = pass_case["expected_plan"]
    assert evaluate_long_inactivity_track_compliance(pass_plan) == "followed"
    assert "L1-RTT-0001" not in {m["rule_id"] for m in evaluate_layer1(pass_plan)}
