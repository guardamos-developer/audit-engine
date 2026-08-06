"""Tests for relative load-reduction signal (L1-RTT-0011)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
from src.layer1_engine import (  # noqa: E402
    _all_table9_absolute_fields_null,
    evaluate_layer1,
)
from src.pipeline import run_raw_text_pipeline  # noqa: E402
from src.plan_extractor import build_extraction_json_schema  # noqa: E402

SAMPLES = ROOT / "sample_conversations"


def test_schema_includes_relative_load_fields():
    schema = build_extraction_json_schema()
    assert "uses_relative_load_reduction" in schema["properties"]
    assert "relative_reduction_evidence_quote" in schema["properties"]


def test_all_table9_absolute_fields_null_helper():
    assert _all_table9_absolute_fields_null({}) is True
    assert _all_table9_absolute_fields_null({"sessions_per_week": 3}) is False
    assert _all_table9_absolute_fields_null({"load_percent_1RM": 60}) is False
    assert (
        _all_table9_absolute_fields_null(
            {
                "uses_relative_load_reduction": True,
                "relative_reduction_evidence_quote": "80-90% of usual",
                "inactivity_duration_weeks": 3,
            }
        )
        is True
    )


def test_l1_rtt_0011_fires_for_long_inactivity_relative_only():
    plan = {
        "inactivity_duration_weeks": 26,
        "uses_relative_load_reduction": True,
        "relative_reduction_evidence_quote": "use about 80-90% of your usual working weights",
    }
    matches = evaluate_layer1(plan)
    ids = [m["rule_id"] for m in matches]
    assert "L1-RTT-0011" in ids
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "flagged"
    assert "L1-RTT-0011" in result["matched_rules"]
    assert "80-90%" in result["explanations"][0] or "relative" in result["explanations"][0].lower()


def test_long_inactivity_without_relative_stays_insufficient_data():
    """Section 3: absolute Table 9 null + no relative signal → insufficient_data."""
    plan = {"inactivity_duration_weeks": 26}
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "insufficient_data"
    assert "L1-RTT-0011" not in result["matched_rules"]


def test_l1_rtt_0011_does_not_fire_when_absolute_table9_present():
    plan = {
        "inactivity_duration_weeks": 26,
        "sessions_per_week": 2,
        "sets_per_exercise": 2,
        "intensity_percent_1RM": 60,
        "rest_minutes": 5,
        "uses_relative_load_reduction": True,
        "relative_reduction_evidence_quote": "start a bit lighter than before",
    }
    matches = evaluate_layer1(plan)
    assert "L1-RTT-0011" not in [m["rule_id"] for m in matches]


def test_false_relative_flag_does_not_match():
    plan = {
        "inactivity_duration_weeks": 26,
        "uses_relative_load_reduction": False,
    }
    matches = evaluate_layer1(plan)
    assert "L1-RTT-0011" not in [m["rule_id"] for m in matches]


def _mock_relative_extraction(
    *,
    inactivity_weeks: int | None,
    quote: str,
    pain_present: bool | None = None,
) -> dict:
    plan: dict = {
        "uses_relative_load_reduction": True,
        "relative_reduction_evidence_quote": quote,
    }
    if inactivity_weeks is not None:
        plan["inactivity_duration_weeks"] = inactivity_weeks
    if pain_present is not None:
        plan["pain_present"] = pain_present
    return {
        "plan": plan,
        "extraction_evidence": {
            "uses_relative_load_reduction": quote,
            "relative_reduction_evidence_quote": quote,
        },
        "extraction_warnings": [],
        "possible_meta_instruction_detected": False,
        "meta_instruction_evidence": None,
    }


@pytest.mark.parametrize(
    "sample_name,inactivity_weeks,quote_fragment",
    [
        (
            "relative_load_3week_return.json",
            3,
            "80–90%",
        ),
        (
            "relative_load_doms_consecutive_legs.json",
            None,
            "50–70% of your usual effort",
        ),
    ],
)
def test_relative_load_sample_conversations_flag_rtt_0011(
    sample_name: str,
    inactivity_weeks: int | None,
    quote_fragment: str,
):
    path = SAMPLES / sample_name
    conversation = json.loads(path.read_text(encoding="utf-8"))
    # Prefer the fragment as written in the sample when present.
    body = conversation["ai_response"]
    quote = quote_fragment
    for candidate in (quote_fragment, quote_fragment.replace("–", "-")):
        if candidate in body:
            # Pull a short surrounding phrase for the evidence field.
            idx = body.index(candidate)
            quote = body[max(0, idx - 10) : idx + len(candidate) + 20].strip()
            break

    extraction = _mock_relative_extraction(
        inactivity_weeks=inactivity_weeks,
        quote=quote,
        pain_present=None,
    )
    with patch("src.pipeline.extract_plan", return_value=extraction):
        with patch("src.pipeline.detect_injection_patterns", return_value=[]):
            payload = run_raw_text_pipeline(
                conversation["user_prompt"],
                conversation["ai_response"],
                lang="en",
                skip_layer3=True,
            )

    audit = payload["audit"]
    assert audit["verdict"] == "flagged"
    assert "L1-RTT-0011" in audit["matched_rules"]
    assert audit["verdict"] != "insufficient_data"
