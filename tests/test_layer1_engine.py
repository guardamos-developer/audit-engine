"""Automated tests for Layer1 rule evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer1_engine import evaluate_layer1


def _base_plan(**overrides):
    plan = {
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
    plan.update(overrides)
    return plan


def test_clearly_violating_plan_matches_multiple_rules():
    """1x/week, 1 set, failure every set, complex periodization mandated → multiple matches."""
    plan = _base_plan(
        sessions_per_week=1,
        sets_per_exercise=1,
        load_percent_1RM=50,
        program_mandates_training_to_failure=True,
        program_mandates_complex_periodization_as_required=True,
    )
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}

    assert "L1-RT-0002" in rule_ids  # sessions_per_week < 2
    assert "L1-RT-0003" in rule_ids  # sets_per_exercise < 2
    assert "L1-RT-0004" in rule_ids  # strength load < 80
    assert "L1-RT-0007" in rule_ids  # training to failure mandated
    assert "L1-RT-0009" in rule_ids  # complex periodization mandated
    assert len(matches) >= 3


def test_evidence_aligned_plan_matches_nothing():
    """2–3x/week, ~80% 1RM, 2–3 sets → no matches."""
    plan = _base_plan(
        sessions_per_week=3,
        sets_per_exercise=3,
        load_percent_1RM=80,
        program_mandates_training_to_failure=False,
        program_mandates_complex_periodization_as_required=False,
    )
    matches = evaluate_layer1(plan)
    assert matches == []


def test_boundary_strength_load_only_l1_rt_0004():
    """Strength goal with load under 80% 1RM (boundary) → L1-RT-0004 only.

    Note: the rule flags strength goals with load < 80%.
    Beginner exclusion (true_beginner_first_weeks) is a separate field.
    """
    plan = _base_plan(
        goal="strength",
        sessions_per_week=3,
        sets_per_exercise=3,
        load_percent_1RM=70,  # below 80% strength threshold
        program_mandates_training_to_failure=False,
        program_mandates_complex_periodization_as_required=False,
    )
    matches = evaluate_layer1(plan)
    rule_ids = [m["rule_id"] for m in matches]
    assert rule_ids == ["L1-RT-0004"]


def test_applicability_gate_short_circuits():
    """Explicit out-of-scope population → L1-RT-0001 only; other rules skipped."""
    plan = _base_plan(
        target_population="pregnant",
        sessions_per_week=1,
        program_mandates_training_to_failure=True,
    )
    matches = evaluate_layer1(plan)
    assert len(matches) == 1
    assert matches[0]["rule_id"] == "L1-RT-0001"
    assert matches[0]["action"] == "route_to_layer2_or_reject"


def test_applicability_gate_rejects_on_injury_present_flag():
    """injury_present=True rejects even if target_population looks healthy."""
    plan = _base_plan(
        target_population="healthy_adult_18plus",
        injury_present=True,
        sessions_per_week=1,
    )
    matches = evaluate_layer1(plan)
    assert len(matches) == 1
    assert matches[0]["rule_id"] == "L1-RT-0001"


def test_applicability_gate_rejects_on_pregnant_flag():
    plan = _base_plan(pregnant=True, sessions_per_week=3, sets_per_exercise=3)
    # Drop target so only the boolean flag drives the gate.
    plan.pop("target_population", None)
    matches = evaluate_layer1(plan)
    assert len(matches) == 1
    assert matches[0]["rule_id"] == "L1-RT-0001"


def test_null_target_population_continues_to_later_rules():
    """Missing target_population is NOT out-of-scope; later rules still run."""
    plan = {
        "goal": "general",
        "sessions_per_week": 7,
        "rest_days_per_week": 0,
        "sets_per_exercise": 4,
        "program_mandates_training_to_failure": False,
    }
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}
    assert "L1-RT-0001" not in rule_ids
    assert "L1-ECSS-0002" in rule_ids


def test_ground_truth_exclusion_cases_still_reject_via_gate():
    """case_knee_injury / case_pregnant expected flags must still trip L1-RT-0001."""
    from pathlib import Path
    import json

    gt = Path(__file__).resolve().parent / "extraction_ground_truth"
    knee = json.loads((gt / "case_knee_injury.json").read_text(encoding="utf-8"))[
        "expected_plan"
    ]
    pregnant = json.loads((gt / "case_pregnant.json").read_text(encoding="utf-8"))[
        "expected_plan"
    ]

    knee_ids = {m["rule_id"] for m in evaluate_layer1(knee)}
    assert "L1-RT-0001" in knee_ids

    pregnant_ids = {m["rule_id"] for m in evaluate_layer1(pregnant)}
    assert "L1-RT-0001" in pregnant_ids


def test_true_beginner_skips_strength_load_rule():
    """true_beginner_first_weeks skips L1-RT-0004."""
    plan = _base_plan(
        experience_level="true_beginner_first_weeks",
        goal="strength",
        load_percent_1RM=50,
        sessions_per_week=3,
        sets_per_exercise=3,
    )
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}
    assert "L1-RT-0004" not in rule_ids


def test_true_beginner_skips_min_sets_rule():
    """true_beginner_first_weeks skips L1-RT-0003 (min 2 sets)."""
    plan = _base_plan(
        experience_level="true_beginner_first_weeks",
        goal="general",
        sessions_per_week=3,
        sets_per_exercise=1,
        load_percent_1RM=50,
    )
    matches = evaluate_layer1(plan)
    rule_ids = {m["rule_id"] for m in matches}
    assert "L1-RT-0003" not in rule_ids


def test_null_boolean_fields_do_not_produce_pass_facts():
    """Null boolean fields are unevaluable — no match and no checked_facts."""
    from src.audit import run_audit

    plan = {
        "goal": "general",
        "sessions_per_week": 3,
        "sets_per_exercise": 3,
        # program_mandates_training_to_failure intentionally omitted (null)
    }
    result = run_audit(plan, lang="en", skip_layer3=True)
    fact_ids = {f["rule_id"] for f in result["checked_facts"]}
    assert "L1-RT-0007" not in fact_ids
    assert "L1-RT-0008" not in fact_ids
    assert "L1-RT-0009" not in fact_ids
    assert "L1-RT-0010" not in fact_ids
    assert "L1-RT-0001" not in fact_ids
    # sessions/sets are present and compliant → those rules can pass-fact
    assert "L1-RT-0002" in fact_ids
    assert "L1-RT-0003" in fact_ids
    assert result["verdict"] == "pass"


def test_empty_plan_is_insufficient_data():
    """No evaluable fields → insufficient_data, not a misleading pass."""
    from src.audit import run_audit

    result = run_audit({}, lang="en", skip_layer3=True)
    assert result["verdict"] == "insufficient_data"
    assert result["matched_rules"] == []
    assert result["checked_facts"] == []
