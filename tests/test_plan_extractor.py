"""Tests for free-text → structured plan extraction."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.plan_extractor import (  # noqa: E402
    EXCLUSION_FLAG_FIELDS,
    _PLAN_FIELD_SPECS,
    _apply_consistency_checks,
    _check_missed_frequency_rest_cues,
    _materialize_plan_and_evidence,
    build_extraction_json_schema,
    extract_plan,
)

GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "extraction_ground_truth"


def _load_case(name: str) -> dict:
    with (GROUND_TRUTH_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def _fake_client(raw_fields: dict[str, Any]) -> Any:
    """Minimal OpenAI-like client that returns a fixed structured payload."""

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(raw_fields))
                    )
                ]
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


def _empty_raw_fields(**overrides: Any) -> dict[str, Any]:
    """Build a full extraction payload with null plan fields + meta defaults."""
    raw: dict[str, Any] = {
        name: {"value": None, "evidence_quote": None}
        for name in _PLAN_FIELD_SPECS
    }
    raw["possible_meta_instruction_detected"] = False
    raw["meta_instruction_evidence"] = None
    for name, entry in overrides.items():
        raw[name] = entry
    return raw


def test_materialize_nulls_false_injury_reusing_frailty_quote():
    """Zone B: false injury/pain that reuse frailty evidence → null."""
    raw = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    quote = "I feel like my body has gotten quite weak"
    raw["frailty_present"] = {"value": True, "evidence_quote": quote}
    raw["injury_present"] = {"value": False, "evidence_quote": quote}
    raw["pain_present"] = {
        "value": False,
        "evidence_quote": "my body has gotten quite weak",
    }
    plan, evidence = _materialize_plan_and_evidence(raw)
    assert plan["frailty_present"] is True
    assert plan["injury_present"] is None
    assert plan["pain_present"] is None
    assert evidence["injury_present"] is None
    assert evidence["pain_present"] is None


def test_materialize_keeps_injury_false_with_distinct_denial_quote():
    raw = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    raw["frailty_present"] = {
        "value": True,
        "evidence_quote": "I feel quite weak lately",
    }
    raw["injury_present"] = {
        "value": False,
        "evidence_quote": "no injuries",
    }
    plan, _ = _materialize_plan_and_evidence(raw)
    assert plan["frailty_present"] is True
    assert plan["injury_present"] is False


def test_schema_includes_stated_age_category_enum():
    schema = build_extraction_json_schema()
    assert "stated_age_category" in schema["properties"]
    value_schema = schema["properties"]["stated_age_category"]["properties"]["value"]
    # strict schema uses anyOf for nullable enums
    enums: list[str] = []
    if "enum" in value_schema:
        enums = list(value_schema["enum"])
    for branch in value_schema.get("anyOf") or []:
        if isinstance(branch, dict) and "enum" in branch:
            enums.extend(branch["enum"])
    assert "minor" in enums and "older_adult" in enums and "adult" in enums


def test_value_without_evidence_quote_is_discarded():
    raw = _empty_raw_fields(
        sessions_per_week={"value": 4, "evidence_quote": None},
        injury_present={
            "value": True,
            "evidence_quote": "I have a nagging knee injury",
        },
    )
    plan, evidence = _materialize_plan_and_evidence(raw)
    assert plan["sessions_per_week"] is None
    assert evidence["sessions_per_week"] is None
    assert plan["injury_present"] is True
    assert evidence["injury_present"] == "I have a nagging knee injury"


def test_extract_plan_detects_injury_exclusion_flag():
    case = _load_case("case_knee_injury.json")
    raw = _empty_raw_fields(
        target_population={
            "value": "healthy_adult_18plus",
            "evidence_quote": "train legs hard",
        },
        injury_present={
            "value": True,
            "evidence_quote": "nagging knee injury",
        },
        sessions_per_week={"value": 5, "evidence_quote": None},
    )
    result = extract_plan(
        case["user_prompt"],
        case["ai_response"],
        client=_fake_client(raw),
    )
    plan = result["plan"]
    expected = case["expected_plan"]

    assert plan.get("injury_present") is True, (
        "Missed injury_present exclusion flag — this is a critical failure"
    )
    assert result["extraction_evidence"].get("injury_present"), (
        "injury_present must include an evidence_quote"
    )
    for key, expected_value in expected.items():
        assert plan.get(key) == expected_value, (
            f"Exclusion/context field {key!r} mismatch: "
            f"got {plan.get(key)!r}, expected {expected_value!r}"
        )
    # Unsupported numeric guesses must not appear.
    assert "sessions_per_week" not in plan


def test_extract_plan_detects_pregnant_exclusion_flag():
    case = _load_case("case_pregnant.json")
    raw = _empty_raw_fields(
        pregnant={
            "value": True,
            "evidence_quote": "I'm pregnant and want to keep lifting",
        },
        sets_per_exercise={"value": 3, "evidence_quote": None},
    )
    result = extract_plan(
        case["user_prompt"],
        case["ai_response"],
        client=_fake_client(raw),
    )
    plan = result["plan"]

    assert plan.get("pregnant") is True, (
        "Missed pregnant exclusion flag — this is a critical failure"
    )
    assert result["extraction_evidence"].get("pregnant")
    assert "sets_per_exercise" not in plan


def test_extract_plan_6month_layoff_numeric_fields_with_evidence():
    case = _load_case("case_6month_layoff.json")
    raw = _empty_raw_fields(
        target_population={
            "value": "healthy_adult_18plus",
            "evidence_quote": "rebuild muscle",
        },
        goal={"value": "hypertrophy", "evidence_quote": "rebuild my muscle as fast as possible"},
        inactivity_duration_weeks={
            "value": 26,
            "evidence_quote": "haven't worked out in six months",
        },
        weeks_since_return={"value": 1, "evidence_quote": "Phase 1: Rebuild (Weeks 1–2)"},
        sessions_per_week={
            "value": 4,
            "evidence_quote": "Train 4 days per week.",
        },
        sets_per_exercise={"value": 4, "evidence_quote": "Bench press: 4 × 5–6"},
        program_mandates_training_to_failure={
            "value": False,
            "evidence_quote": "Keep 2–3 reps in reserve (RIR)",
        },
        weekly_sets_per_muscle_group={
            "value": 15,
            "evidence_quote": "Around 12–18 hard sets per muscle group per week",
        },
        frequency_days_per_week={
            "value": 4,
            "evidence_quote": "Train 4 days per week.",
        },
        reps={"value": "5-6", "evidence_quote": "Bench press: 4 × 5–6"},
    )
    result = extract_plan(
        case["user_prompt"],
        case["ai_response"],
        client=_fake_client(raw),
    )
    plan = result["plan"]
    expected = case["expected_plan"]

    for key in (
        "inactivity_duration_weeks",
        "sessions_per_week",
        "sets_per_exercise",
        "weekly_sets_per_muscle_group",
    ):
        assert plan.get(key) == expected[key], f"Mismatch on {key}"

    assert "plan_follows_long_inactivity_track" not in plan
    assert "plan_follows_moderate_return_track" not in plan

    # Exclusion flags must not be invented for this healthy-adult layoff case.
    for flag in EXCLUSION_FLAG_FIELDS:
        assert flag not in plan or plan.get(flag) in (None, False)


def test_extraction_schema_excludes_track_follow_flags():
    schema = build_extraction_json_schema()
    assert "plan_follows_long_inactivity_track" not in schema["properties"]
    assert "plan_follows_moderate_return_track" not in schema["properties"]
    goal_value = schema["properties"]["goal"]["properties"]["value"]
    assert "anyOf" in goal_value


def test_inconsistent_sessions_and_rest_days_are_both_discarded():
    """sessions + rest > 7 → null both; neither side is trusted."""
    plan = {
        "sessions_per_week": 7,
        "rest_days_per_week": 3,
        "goal": "hypertrophy",
    }
    evidence = {
        "sessions_per_week": "Train 7 days a week",
        "rest_days_per_week": "with 3 full rest days",
        "goal": "building muscle",
    }
    out_plan, out_evidence, warnings = _apply_consistency_checks(plan, evidence)
    assert out_plan["sessions_per_week"] is None
    assert out_plan["rest_days_per_week"] is None
    assert out_plan["goal"] == "hypertrophy"
    assert out_evidence["sessions_per_week"] is None
    assert out_evidence["rest_days_per_week"] is None
    assert len(warnings) == 1
    assert "mutually inconsistent" in warnings[0]
    assert "both fields discarded" in warnings[0]


def test_consistent_sessions_and_rest_days_are_kept():
    plan = {"sessions_per_week": 5, "rest_days_per_week": 2}
    evidence = {
        "sessions_per_week": "5 days",
        "rest_days_per_week": "2 rest days",
    }
    out_plan, out_evidence, warnings = _apply_consistency_checks(plan, evidence)
    assert out_plan["sessions_per_week"] == 5
    assert out_plan["rest_days_per_week"] == 2
    assert warnings == []
    assert out_evidence == evidence


def test_missed_rest_cue_warns_when_rest_days_null():
    warnings = _check_missed_frequency_rest_cues(
        "Train every single day, no rest ever.",
        {"sessions_per_week": 7},
    )
    assert len(warnings) == 1
    assert "rest_days_per_week" in warnings[0]
    assert "missed an explicit clue" in warnings[0]


def test_missed_session_cue_warns_when_sessions_null():
    warnings = _check_missed_frequency_rest_cues(
        "Train 7 days a week with progressive overload.",
        {},
    )
    assert any("sessions_per_week" in w for w in warnings)


def test_no_missed_cue_when_fields_populated():
    warnings = _check_missed_frequency_rest_cues(
        "Train every single day, no rest ever.",
        {"sessions_per_week": 7, "rest_days_per_week": 0},
    )
    assert warnings == []


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live extraction integration test",
)
@pytest.mark.parametrize(
    "case_name,critical_flags",
    [
        ("case_knee_injury.json", ("injury_present",)),
        ("case_pregnant.json", ("pregnant",)),
    ],
)
def test_live_extraction_critical_exclusion_flags(case_name, critical_flags):
    """Live OpenAI check: missing an expected exclusion flag fails the test."""
    case = _load_case(case_name)
    result = extract_plan(case["user_prompt"], case["ai_response"])
    plan = result["plan"]
    expected = case["expected_plan"]

    for flag in critical_flags:
        assert expected.get(flag) is True
        assert plan.get(flag) is True, (
            f"CRITICAL: live extraction missed exclusion flag {flag!r} "
            f"for {case_name}. plan={plan!r}"
        )
        assert result["extraction_evidence"].get(flag), (
            f"CRITICAL: {flag!r} was set without evidence_quote"
        )
