"""Regression fixtures for exclusion / qualitative-age extraction (English).

Recorded live gpt-4o-mini extractions (2026-08-07). The euphemistic-stamina
miss (var2) is intentionally omitted — see ruleset_notes
``lenient_exclusion_euphemism_limit``.

Fixtures may use either:
  - ``recorded_extraction`` (legacy single blob; projected into stage1/stage2), or
  - ``recorded_extraction_stage1`` / ``recorded_extraction_stage2`` (explicit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
from src.layer1_engine import (  # noqa: E402
    effective_target_population,
    evaluate_layer1_detailed,
)
from src.plan_extractor import extract_plan  # noqa: E402
from tests.extraction_fakes import (  # noqa: E402
    empty_raw_fields,
    fake_client,
    fake_client_two_stage,
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
    # Design lock: adult age + frailty does NOT early-exit (ACSM gate ignores
    # frailty); stage 2 must still run — same as pre-two-stage behavior.
    "case_adult_age40_frailty_continues_stage2.json",
)


def _load_case(name: str) -> dict:
    with (GROUND_TRUTH_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def _client_for_case(case: dict):
    """Build a fake client from legacy or stage-split recorded extraction."""
    stage1_over = case.get("recorded_extraction_stage1")
    stage2_over = case.get("recorded_extraction_stage2")
    expect_stage2 = case.get("expect_stage2_called")

    if stage1_over is not None:
        stage1 = empty_raw_fields(**stage1_over)
        # empty_raw_fields fills all plan keys; project happens in fake_client_two_stage
        # via explicit stage payloads — pass stage-scoped dicts directly.
        from tests.extraction_fakes import _project_raw
        from src.plan_extractor import STAGE1_FIELD_NAMES, STAGE2_ADULT_FIELD_NAMES

        s1 = _project_raw(stage1, STAGE1_FIELD_NAMES)
        if expect_stage2 is False or stage2_over is None:
            return fake_client_two_stage(s1, None, expect_stage2=False)
        stage2 = empty_raw_fields(**(stage2_over or {}))
        s2 = _project_raw(stage2, STAGE2_ADULT_FIELD_NAMES)
        # Older stage2 fields may also appear in stage2_over; merge both sets.
        from src.plan_extractor import STAGE2_OLDER_FIELD_NAMES

        s2_older = _project_raw(stage2, STAGE2_OLDER_FIELD_NAMES)
        s2 = {**s2_older, **s2}
        s2["possible_meta_instruction_detected"] = bool(
            stage2.get("possible_meta_instruction_detected", False)
        )
        s2["meta_instruction_evidence"] = stage2.get("meta_instruction_evidence")
        return fake_client_two_stage(s1, s2, expect_stage2=True)

    raw = empty_raw_fields(**(case.get("recorded_extraction") or {}))
    return fake_client(raw, expect_stage2=expect_stage2)


@pytest.mark.parametrize("case_name", FIXTURE_CASES)
def test_recorded_exclusion_fixtures_extract_and_audit(case_name: str):
    case = _load_case(case_name)
    client = _client_for_case(case)
    result = extract_plan(
        case["user_prompt"],
        case["ai_response"],
        client=client,
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

    if case.get("expect_stage2_called") is not None:
        assert result.get("stage2_ran") is case["expect_stage2_called"], case_name

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
            if str(a["rule_id"]).startswith("L1-RT-NSCA-")
        ]
        assert nsca_ids, (case_name, "expected NSCA rules applicable")
