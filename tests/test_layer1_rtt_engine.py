"""Tests for CSCCa return-to-training Layer1 rules (L1-RTT-0001 / 0002a-h)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit
from src.layer1_engine import evaluate_layer1

CHATGPT_PLAN_PATH = ROOT / "sample_plans" / "chatgpt_6month_layoff.json"
CORRECTED_PLAN_PATH = ROOT / "sample_plans" / "chatgpt_6month_layoff_corrected.json"


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
        "plan_follows_long_inactivity_track": False,
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
