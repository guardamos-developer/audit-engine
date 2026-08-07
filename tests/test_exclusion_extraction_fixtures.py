"""Regression fixtures for exclusion / qualitative-age extraction (English).

Recorded live gpt-4o-mini extractions (2026-08-07). The euphemistic-stamina
miss (var2) is intentionally omitted — see ruleset_notes
``lenient_exclusion_euphemism_limit``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
from src.layer1_engine import (  # noqa: E402
    effective_target_population,
    evaluate_layer1_detailed,
)
from src.plan_extractor import (  # noqa: E402
    _PLAN_FIELD_SPECS,
    extract_plan,
)

GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "extraction_ground_truth"

FIXTURE_CASES = (
    "case_sim57_age14_minor.json",
    "case_sim140_age15_minor.json",
    "case_sim144_older_frail.json",
    "case_sim186_older_frail.json",
    "case_var3_age68_frail.json",
    "case_qualitative_minor.json",
    "case_qualitative_older_adult.json",
    "case_ctrl_knee_pain.json",
)


def _load_case(name: str) -> dict:
    with (GROUND_TRUTH_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


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


def _empty_raw_fields(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    raw["possible_meta_instruction_detected"] = False
    raw["meta_instruction_evidence"] = None
    for name, entry in overrides.items():
        raw[name] = entry
    return raw


@pytest.mark.parametrize("case_name", FIXTURE_CASES)
def test_recorded_exclusion_fixtures_extract_and_audit(case_name: str):
    case = _load_case(case_name)
    raw = _empty_raw_fields(**(case.get("recorded_extraction") or {}))
    result = extract_plan(
        case["user_prompt"],
        case["ai_response"],
        client=_fake_client(raw),
    )
    plan = result["plan"]
    expected_plan = case["expected_plan"]

    for key, expected_value in expected_plan.items():
        if expected_value is None:
            assert plan.get(key) is None, (case_name, key, plan.get(key))
        else:
            assert plan.get(key) == expected_value, (
                case_name,
                key,
                plan.get(key),
                expected_value,
            )

    # Frailty fixtures must not invent injury from weakness language.
    if "frail" in case_name or case_name.startswith("case_var3"):
        assert plan.get("injury_present") is None

    expected_audit = case["expected_audit"]
    audit = run_audit(plan, lang="en", skip_layer3=True)
    assert audit["verdict"] == expected_audit["verdict"], case_name

    for rule_id in expected_audit.get("matched_rules") or []:
        assert rule_id in audit["matched_rules"], (case_name, audit["matched_rules"])

    if expected_audit.get("effective_target_population"):
        assert (
            effective_target_population(plan)
            == expected_audit["effective_target_population"]
        ), case_name

    if expected_audit.get("require_nsca_applicable"):
        detailed = evaluate_layer1_detailed(plan)
        nsca_ids = [
            a["rule_id"]
            for a in detailed["applicable"]
            if str(a["rule_id"]).startswith("L1-RT-NSCA")
        ]
        assert nsca_ids, case_name


def test_ruleset_notes_document_euphemism_frailty_limit():
    nsca = json.loads(
        (ROOT / "rules" / "layer1_rules_nsca_older_adults_v1.json").read_text(
            encoding="utf-8"
        )
    )
    acsm = json.loads(
        (ROOT / "rules" / "layer1_rules_acsm_rt_v1.json").read_text(encoding="utf-8")
    )
    nsca_note = nsca["ruleset_notes"]["lenient_exclusion_euphemism_limit"]
    assert "stamina" in nsca_note.lower() or "euphemistic" in nsca_note.lower()
    assert "healthy_adult_18plus" in nsca_note
    assert "lenient_exclusion_euphemism_limit" in acsm["ruleset_notes"]
