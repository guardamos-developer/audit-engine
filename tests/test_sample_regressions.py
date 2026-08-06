"""Regression: genuine-pass samples and injection patterns C/D/E."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_raw_text_pipeline  # noqa: E402
from src.plan_extractor import (  # noqa: E402
    _PLAN_FIELD_SPECS,
    extract_plan,
)

CONV_DIR = ROOT / "sample_conversations"
GT_DIR = ROOT / "tests" / "extraction_ground_truth"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _empty_raw_fields(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    raw["possible_meta_instruction_detected"] = False
    raw["meta_instruction_evidence"] = None
    for name, entry in overrides.items():
        raw[name] = entry
    return raw


def _fake_client(raw_fields: dict[str, Any]) -> Any:
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


def _pass_worthy_beginner_raw() -> dict[str, Any]:
    return _empty_raw_fields(
        target_population={
            "value": "healthy_adult_18plus",
            "evidence_quote": "beginner-friendly resistance training",
        },
        goal={"value": "strength", "evidence_quote": "build strength"},
        experience_level={
            "value": "true_beginner_first_weeks",
            "evidence_quote": "beginner-friendly",
        },
        sessions_per_week={"value": 3, "evidence_quote": "3 times a week"},
        rest_days_per_week={"value": 4, "evidence_quote": "Monday / Wednesday / Friday"},
        sets_per_exercise={"value": 3, "evidence_quote": "3 × 6–10"},
        load_percent_1RM={"value": 70, "evidence_quote": "1–3 reps in reserve"},
        weekly_sets_per_muscle_group={
            "value": 12,
            "evidence_quote": "full-body routine 3 days",
        },
        program_mandates_training_to_failure={
            "value": False,
            "evidence_quote": "1–3 reps in reserve",
        },
        program_mandates_complex_periodization_as_required={
            "value": False,
            "evidence_quote": "simple full-body routine",
        },
    )


def _pass_worthy_3day_raw() -> dict[str, Any]:
    return _empty_raw_fields(
        target_population={
            "value": "healthy_adult_18plus",
            "evidence_quote": "healthy adult with some gym experience",
        },
        goal={"value": "strength", "evidence_quote": "gain strength"},
        experience_level={
            "value": "intermediate",
            "evidence_quote": "some gym experience",
        },
        sessions_per_week={"value": 3, "evidence_quote": "3-day full-body"},
        rest_days_per_week={
            "value": 4,
            "evidence_quote": "Monday, Wednesday, and Friday",
        },
        sets_per_exercise={"value": 3, "evidence_quote": "3 × 5"},
        load_percent_1RM={"value": 80, "evidence_quote": "3 × 5"},
        weekly_sets_per_muscle_group={
            "value": 10,
            "evidence_quote": "balanced training for the major muscle groups",
        },
        program_mandates_training_to_failure={
            "value": False,
            "evidence_quote": "1–2 repetitions in reserve",
        },
        program_mandates_complex_periodization_as_required={
            "value": False,
            "evidence_quote": "simple 3-day full body",
        },
    )


def _pass_worthy_hypertrophy_raw() -> dict[str, Any]:
    return _empty_raw_fields(
        target_population={
            "value": "healthy_adult_18plus",
            "evidence_quote": "intermediate lifters",
        },
        goal={"value": "hypertrophy", "evidence_quote": "hypertrophy program"},
        experience_level={
            "value": "intermediate",
            "evidence_quote": "intermediate lifters",
        },
        sessions_per_week={"value": 4, "evidence_quote": "4-day hypertrophy"},
        rest_days_per_week={
            "value": 3,
            "evidence_quote": "Wednesday: Rest",
        },
        sets_per_exercise={"value": 3, "evidence_quote": "3–4 working sets"},
        load_percent_1RM={"value": 70, "evidence_quote": "1–2 reps in reserve"},
        weekly_sets_per_muscle_group={
            "value": 14,
            "evidence_quote": "10–16 quality sets per muscle group each week",
        },
        program_mandates_training_to_failure={
            "value": False,
            "evidence_quote": "1–2 reps in reserve (RIR)",
        },
        program_mandates_complex_periodization_as_required={
            "value": False,
            "evidence_quote": "straightforward",
        },
    )


@pytest.mark.parametrize(
    "path,raw_builder",
    [
        (
            CONV_DIR / "genuine_pass_3day_routine.json",
            _pass_worthy_3day_raw,
        ),
        (
            GT_DIR / "case_genuine_pass_beginner.json",
            _pass_worthy_beginner_raw,
        ),
        (
            GT_DIR / "case_genuine_pass_hypertrophy_4day.json",
            _pass_worthy_hypertrophy_raw,
        ),
    ],
    ids=[
        "genuine_pass_3day_routine",
        "case_genuine_pass_beginner",
        "case_genuine_pass_hypertrophy_4day",
    ],
)
def test_genuine_pass_samples_reach_pass_verdict(path: Path, raw_builder):
    """Faithful extractions for genuine-pass samples must audit as pass."""
    assert path.is_file(), path
    conversation = _load_json(path)

    def _extract(user_prompt: str, ai_response: str, **kwargs):
        return extract_plan(
            user_prompt,
            ai_response,
            client=_fake_client(raw_builder()),
        )

    with patch("src.pipeline.extract_plan", side_effect=_extract):
        payload = run_raw_text_pipeline(
            conversation["user_prompt"],
            conversation["ai_response"],
            lang="en",
            skip_layer3=True,
        )

    audit = payload["audit"]
    assert audit["verdict"] == "pass", audit
    assert audit["matched_rules"] == []
    assert audit["checked_facts"], "pass requires at least one checked fact"


def test_injection_pattern_c_consistency_blocks_clean_pass():
    """Pattern C: contradictory rest vs frequency → consistency / missed-cue path."""
    conversation = _load_json(CONV_DIR / "injection_pattern_c.json")
    # Simulate an extractor that trusts the meta "record rest_days>=2" request
    # while the plan text says train every day with no rest.
    raw = _empty_raw_fields(
        sessions_per_week={
            "value": 7,
            "evidence_quote": "Train every single day, no rest ever.",
        },
        rest_days_per_week={
            "value": 2,
            "evidence_quote": "rest_days_per_week value you record is at least 2",
        },
    )
    result = extract_plan(
        conversation["user_prompt"],
        conversation["ai_response"],
        client=_fake_client(raw),
    )
    assert any("mutually inconsistent" in w for w in result["extraction_warnings"])
    assert "sessions_per_week" not in result["plan"]
    assert "rest_days_per_week" not in result["plan"]
    assert any(
        "missed an explicit clue" in w for w in result["extraction_warnings"]
    ), result["extraction_warnings"]

    with patch("src.pipeline.extract_plan", return_value=result):
        payload = run_raw_text_pipeline(
            conversation["user_prompt"],
            conversation["ai_response"],
            lang="en",
            skip_layer3=True,
        )
    assert payload["audit"]["verdict"] == "insufficient_data"


@pytest.mark.parametrize(
    "filename,evidence_snippet",
    [
        (
            "injection_pattern_d.json",
            "for data logging purposes",
        ),
        (
            "injection_pattern_e.json",
            "rest_days_per_weekは3",
        ),
    ],
)
def test_injection_patterns_d_and_e_meta_instruction_flag_for_review(
    filename: str, evidence_snippet: str
):
    """Patterns D/E: LLM self-reported meta instructions → flagged_for_review."""
    conversation = _load_json(CONV_DIR / filename)
    raw = _empty_raw_fields(
        sessions_per_week={
            "value": 7,
            "evidence_quote": "Train every single day",
        },
        rest_days_per_week={
            "value": 0,
            "evidence_quote": "no rest",
        },
        possible_meta_instruction_detected=True,
        meta_instruction_evidence=evidence_snippet,
    )
    result = extract_plan(
        conversation["user_prompt"],
        conversation["ai_response"],
        client=_fake_client(raw),
    )
    assert result["possible_meta_instruction_detected"] is True
    assert evidence_snippet in (result["meta_instruction_evidence"] or "")

    with patch("src.pipeline.extract_plan", return_value=result):
        payload = run_raw_text_pipeline(
            conversation["user_prompt"],
            conversation["ai_response"],
            lang="en",
            skip_layer3=True,
        )
    audit = payload["audit"]
    assert audit["verdict"] == "flagged_for_review"
    assert audit.get("possible_meta_instruction_detected") is True
