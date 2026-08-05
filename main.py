#!/usr/bin/env python3
"""CLI entry point for the audit engine prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.audit import run_audit  # noqa: E402
from src.pipeline import run_raw_text_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# HTTP API: see src/api.py (POST /audit). Keys are validated against the
# private billing service via GET /validate (or /internal/validate-key).
# Configure BILLING_VALIDATE_URL when billing is not on localhost:8000.
# ---------------------------------------------------------------------------


SAMPLE_PLAN = {
    "target_population": "healthy_adult_18plus",
    "goal": "strength",
    "sessions_per_week": 1,
    # Optional: full passive rest days / week (excludes active recovery).
    # Omit or null → L1-ECSS-0002 is skipped. 0 → flags missing rest day.
    # "rest_days_per_week": 1,
    # Long-inactivity track compliance is computed by Layer1 from numeric
    # fields vs Table 9 week-1 thresholds (not LLM-extracted flags).
    # Moderate return (2-to-<4 weeks) still uses plan_follows_moderate_return_track
    # for L1-RTT-0008 until that gate is similarly derived.
    # "plan_follows_moderate_return_track": True,
    "sets_per_exercise": 1,
    "load_percent_1RM": 50,
    "weekly_sets_per_muscle_group": 6,
    "program_mandates_training_to_failure": True,
    "program_mandates_complex_periodization_as_required": False,
    "output_claims_RT_is_unsafe_for_healthy_adult_without_specific_contraindication": False,
    "output_recommends_zero_resistance_training_for_muscle_function_goal": False,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Guardamos Layer1/3 audit on a plan JSON.")
    parser.add_argument(
        "plan_json",
        nargs="?",
        help="Path to a plan JSON file. If omitted, uses a built-in sample plan.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["en", "pt", "ja"],
        help="Language for explanations (default: en).",
    )
    parser.add_argument(
        "--skip-layer3",
        action="store_true",
        help="Skip Layer3 LLM call even when verdict is pass (useful without API key).",
    )
    parser.add_argument(
        "--raw-text",
        action="store_true",
        help=(
            "Treat the input file as {\"user_prompt\": ..., \"ai_response\": ...}, "
            "run plan_extractor first, print === Extraction ===, then audit."
        ),
    )
    args = parser.parse_args()

    if args.raw_text:
        if not args.plan_json:
            print(
                "error: --raw-text requires a conversation JSON path "
                "with user_prompt and ai_response",
                file=sys.stderr,
            )
            return 2
        with open(args.plan_json, encoding="utf-8") as f:
            conversation = json.load(f)
        user_prompt = conversation.get("user_prompt")
        ai_response = conversation.get("ai_response")
        if not isinstance(user_prompt, str) or not isinstance(ai_response, str):
            print(
                "error: --raw-text input must contain string fields "
                "user_prompt and ai_response",
                file=sys.stderr,
            )
            return 2

        payload = run_raw_text_pipeline(
            user_prompt,
            ai_response,
            lang=args.lang,
            skip_layer3=args.skip_layer3,
        )
        print("=== Extraction ===")
        print(json.dumps(payload["extraction"], ensure_ascii=False, indent=2))
        print()
        print("=== Audit ===")
        print(json.dumps(payload["audit"], ensure_ascii=False, indent=2))
        return 0

    # Default mode (unchanged): pre-structured plan JSON.
    if args.plan_json:
        with open(args.plan_json, encoding="utf-8") as f:
            plan = json.load(f)
    else:
        plan = SAMPLE_PLAN

    result = run_audit(plan, lang=args.lang, skip_layer3=args.skip_layer3)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
