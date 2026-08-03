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

# ---------------------------------------------------------------------------
# Future API auth design (not implemented — audit-engine is still CLI-only):
#
# When this tool is exposed as an HTTP API (e.g. POST /audit), require clients
# to send `X-API-Key: gdm_test_...` (or gdm_live_...). Validate against the
# separate private billing service / shared key store, e.g.:
#
#   # billing lives in github.com/guardamos-developer/billing (private)
#   from billing.api_keys import validate_api_key  # or an HTTP call to billing
#   if not validate_api_key(request.headers.get("X-API-Key", "")):
#       raise HTTPException(401, "Invalid or missing API key")
#
# Keys are minted by billing on Stripe checkout.session.completed.
# Keep billing and audit-engine loosely coupled (shared DB path via
# GUARDAMOS_API_KEYS_DB, or a later shared store / internal validate endpoint).
# ---------------------------------------------------------------------------


SAMPLE_PLAN = {
    "target_population": "healthy_adult_18plus",
    "goal": "strength",
    "sessions_per_week": 1,
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
    args = parser.parse_args()

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
