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
    """週1回・1セット・毎セットfailure・複雑な期分け強制 → 複数ルールが同時マッチ."""
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
    """週2〜3回・80%1RM前後・2〜3セット → マッチなし."""
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
    """筋力目標で80%1RM未満（境界）→ L1-RT-0004のみマッチ.

    注: ルール本体は「筋力目標で load < 80%」をフラグする。
    初心者除外(true_beginner_first_weeks)は別フィールドで表現する。
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
    """対象外集団なら L1-RT-0001 のみ返し、他ルールは評価しない."""
    plan = _base_plan(
        target_population="pregnant",
        sessions_per_week=1,
        program_mandates_training_to_failure=True,
    )
    matches = evaluate_layer1(plan)
    assert len(matches) == 1
    assert matches[0]["rule_id"] == "L1-RT-0001"
    assert matches[0]["action"] == "route_to_layer2_or_reject"


def test_true_beginner_skips_strength_load_rule():
    """true_beginner_first_weeks は L1-RT-0004 の適用外."""
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
