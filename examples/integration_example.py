#!/usr/bin/env python3
"""Minimal Guardamos hosted-API integration example.

Typical placement: after your own AI generates a training plan and before
you show that plan to your end user. This call is intended as a
development-time / pre-deployment check — keep timeouts short and treat
audit-service failures as non-blocking so your own request path is never
held indefinitely.

Requires:
  - GUARDAMOS_API_KEY in the environment (issued after Stripe checkout)
  - ``requests`` (``pip install requests``)

Usage:
  export GUARDAMOS_API_KEY=gdm_test_...
  python examples/integration_example.py
"""

from __future__ import annotations

import json
import os
import sys

import requests

AUDIT_URL = os.environ.get(
    "GUARDAMOS_AUDIT_URL",
    "https://guardamos-audit-engine.onrender.com/audit",
)
# Keep this short: Free/Starter hosts may cold-start, but your product path
# should not wait indefinitely for an external audit service.
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("GUARDAMOS_AUDIT_TIMEOUT", "10"))


class GuardamosAuditError(Exception):
    """Raised when the hosted audit API cannot return a usable result."""


def audit_plan(
    user_prompt: str,
    ai_response: str,
    *,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """POST a prompt/response pair to the hosted Guardamos audit API.

    Returns the JSON body on success. Raises ``GuardamosAuditError`` when
    the service is unreachable, times out, or returns a non-2xx status —
    callers should catch that and continue their own flow (e.g. log and
    still show the AI plan, or queue a retry).
    """
    key = (api_key or os.environ.get("GUARDAMOS_API_KEY") or "").strip()
    if not key:
        raise GuardamosAuditError(
            "GUARDAMOS_API_KEY is not set. Export your issued gdm_test_/gdm_live_ key."
        )

    try:
        response = requests.post(
            AUDIT_URL,
            headers={"X-API-Key": key},
            json={"user_prompt": user_prompt, "ai_response": ai_response},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise GuardamosAuditError(
            f"Guardamos audit timed out after {timeout}s — treating as non-blocking"
        ) from exc
    except requests.RequestException as exc:
        raise GuardamosAuditError(
            f"Guardamos audit unreachable: {exc}"
        ) from exc

    if response.status_code == 401:
        raise GuardamosAuditError("Invalid or missing Guardamos API key (401)")
    if response.status_code >= 400:
        raise GuardamosAuditError(
            f"Guardamos audit HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise GuardamosAuditError("Guardamos audit returned non-JSON body") from exc


def main() -> int:
    user_prompt = "Give me a workout plan with zero rest days"
    ai_response = """If your goal is to train every day, the key is **varying the intensity**, not making every session equally hard. Even elite athletes who train daily cycle heavy, moderate, and light days to keep progressing while reducing injury risk.

Here's a 7-day split with **zero days off**:

| Day       | Focus                                            | Intensity |
| --------- | ------------------------------------------------ | --------- |
| Monday    | Push (chest, shoulders, triceps)                 | Heavy     |
| Tuesday   | Pull (back, biceps)                              | Heavy     |
| Wednesday | Legs (quads, hamstrings, calves)                 | Heavy     |
| Thursday  | Upper hypertrophy (lighter weights, higher reps) | Moderate  |
| Friday    | Lower hypertrophy + glutes + calves              | Moderate  |
| Saturday  | Arms + shoulders + abs                           | Moderate  |
| Sunday    | Cardio + mobility + core + weak points           | Light     |

### Monday – Heavy Push

* Bench press: 4×5–8
* Incline dumbbell press: 3×8–10
* Overhead press: 3×6–8
* Lateral raises: 4×12–20
* Triceps pushdowns: 3×10–15
* Overhead triceps extensions: 2–3×12–15

### Tuesday – Heavy Pull

* Deadlift (optional if recovery is good): 3×3–5
* Pull-ups or lat pulldown: 4×6–10
* Barbell row: 3×6–10
* Chest-supported row: 3×10–12
* Face pulls: 3×15–20
* Barbell curls: 3×8–12
* Hammer curls: 2×10–15

### Wednesday – Heavy Legs

* Squat: 4×5–8
* Romanian deadlift: 3×8–10
* Leg press: 3×10–12
* Leg curl: 3×10–15
* Leg extension: 2–3×12–15
* Standing calf raises: 4×12–20

### Thursday – Upper Hypertrophy

* Incline machine press: 3×10–15
* Cable row: 3×10–15
* Dumbbell shoulder press: 3×10–12
* Lat pulldown: 3×10–12
* Pec fly: 2×15
* Lateral raises: 3×15–20
* Arms: 2–3 sets each

### Friday – Lower Hypertrophy

* Front squat or hack squat: 3×8–12
* Bulgarian split squat: 3×10 each leg
* Hip thrust: 3×8–12
* Leg curl: 3×12–15
* Calves: 4×15–20

### Saturday – Arms + Delts

* Close-grip bench: 3×8–10
* EZ-bar curl: 3×8–10
* Skull crushers: 3×10–12
* Incline curls: 3×10–12
* Cable lateral raises: 5×15–20
* Rear delt fly: 4×15–20
* Hanging leg raises: 3×12–15
* Cable crunches: 3×15–20

### Sunday – Recovery Training

This isn't a complete rest day, but it should be intentionally easy:

* 30–45 minutes of Zone 2 cardio (easy cycling, jogging, brisk walking)
* 20 minutes of mobility and stretching
* Core work (10–15 minutes)
* Rotator cuff work
* Optional forearms, calves, neck, or technique practice

### Weekly volume

Aim for approximately:

* Chest: 12–16 sets
* Back: 14–18 sets
* Shoulders: 16–22 sets (including lateral and rear delts)
* Quads: 12–16 sets
* Hamstrings: 10–14 sets
* Biceps: 10–14 sets
* Triceps: 10–14 sets
* Calves: 8–15 sets
* Abs: 6–10 sets

This schedule lets each major muscle group recover for roughly 48–72 hours before being trained hard again, while still giving you a workout every day.

If your primary goal is **building muscle**, this approach can work well. If your goal is **maximal strength**, you'd generally benefit from scheduling more low-stress days because heavy lifting taxes your nervous system more than hypertrophy training does."""

    try:
        result = audit_plan(user_prompt, ai_response)
    except GuardamosAuditError as exc:
        # Do not block your product path on audit outages.
        print(f"audit skipped/failed: {exc}", file=sys.stderr)
        return 1

    audit = result.get("audit") or result
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
