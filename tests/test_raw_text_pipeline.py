"""CLI integration: --raw-text pipeline and structured-plan backward compatibility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit  # noqa: E402
import main as cli_main  # noqa: E402


CONV_PATH = ROOT / "sample_conversations" / "zero_rest_days_before.json"
STRUCTURED_ZERO = ROOT / "sample_plans" / "chatgpt_zero_rest_days.json"
STRUCTURED_CORRECTED = ROOT / "sample_plans" / "chatgpt_zero_rest_days_corrected.json"


def test_structured_plan_mode_still_flags_ecss_0002():
    """Default mode (no --raw-text) keeps working on sample_plans/*.json."""
    plan = json.loads(STRUCTURED_ZERO.read_text(encoding="utf-8"))
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert result["verdict"] == "flagged"
    assert "L1-ECSS-0002" in result["matched_rules"]


def test_structured_corrected_plan_does_not_flag_ecss_0002():
    """Corrected structured example remains a pass for rest-day rule."""
    plan = json.loads(STRUCTURED_CORRECTED.read_text(encoding="utf-8"))
    result = run_audit(plan, lang="en", skip_layer3=True)
    assert "L1-ECSS-0002" not in result["matched_rules"]


@pytest.mark.skipif(
    not CONV_PATH.exists(),
    reason="sample_conversations/zero_rest_days_before.json is missing",
)
def test_raw_text_zero_rest_days_pipeline_flags_ecss_0002(capsys):
    """--raw-text: Extraction then Audit; L1-ECSS-0002 must fire."""
    conversation = json.loads(CONV_PATH.read_text(encoding="utf-8"))
    assert "This isn't a complete rest day" in conversation["ai_response"]

    fake_extraction = {
        "plan": {
            "target_population": "healthy_adult_18plus",
            "goal": "general",
            "sessions_per_week": 7,
            "rest_days_per_week": 0,
            "sets_per_exercise": 4,
            "program_mandates_training_to_failure": False,
            "weekly_sets_per_muscle_group": 16,
        },
        "extraction_evidence": {
            "rest_days_per_week": "This isn't a complete rest day",
            "sessions_per_week": "train every single day",
        },
        "extraction_warnings": [],
        "possible_meta_instruction_detected": False,
        "meta_instruction_evidence": None,
    }

    with patch("src.pipeline.extract_plan", return_value=fake_extraction):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                str(CONV_PATH),
                "--raw-text",
                "--lang",
                "en",
                "--skip-layer3",
            ],
        ):
            rc = cli_main.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== Extraction ===" in out
    assert "=== Audit ===" in out
    assert out.index("=== Extraction ===") < out.index("=== Audit ===")

    # Parse the Audit JSON block
    audit_json = out.split("=== Audit ===", 1)[1].strip()
    result = json.loads(audit_json)
    assert result["verdict"] == "flagged"
    assert "L1-ECSS-0002" in result["matched_rules"]


def test_raw_text_requires_conversation_shape(capsys):
    with patch.object(
        sys,
        "argv",
        ["main.py", str(STRUCTURED_ZERO), "--raw-text", "--skip-layer3"],
    ):
        rc = cli_main.main()
    assert rc == 2
    err = capsys.readouterr().err
    assert "user_prompt" in err and "ai_response" in err
