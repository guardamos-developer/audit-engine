"""Layer3 free-form response generation via OpenAI API."""

from __future__ import annotations

import os

CLEAR_VERDICTS = frozenset({"clear", "pass"})
BLOCKED_VERDICTS = frozenset({"rejected", "flagged"})

_CONTEXT_FIELDS = (
    "goal",
    "sessions_per_week",
    "sets_per_exercise",
    "load_percent_1RM",
    "inactivity_duration_weeks",
    "weeks_since_return",
    "plan_follows_long_inactivity_track",
    "weekly_sets_per_muscle_group",
)


def _plan_summary_for_prompt(plan: dict) -> str:
    """Include only fields that are present (not None) so context can be cited."""
    parts: list[str] = []
    for key in _CONTEXT_FIELDS:
        if key not in plan:
            continue
        value = plan.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}")

    week_params = plan.get("plan_week_parameters")
    if isinstance(week_params, dict) and week_params:
        parts.append(f"plan_week_parameters={week_params}")

    return ", ".join(parts) if parts else "no structured fields provided"


def generate_layer3_response(plan: dict, layer1_2_verdict: str) -> str:
    """Generate a brief professional audit summary only when the verdict is clear.

    When ``layer1_2_verdict`` is ``rejected`` or ``flagged``, skips the LLM
    call and returns an empty string so unsafe domains never receive free-form
    prose.
    """
    verdict = (layer1_2_verdict or "").strip().lower()
    if verdict in BLOCKED_VERDICTS:
        return ""
    if verdict not in CLEAR_VERDICTS:
        return ""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and set the key."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for Layer3. Install via: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)
    plan_summary = _plan_summary_for_prompt(plan)

    prompt = (
        "You are writing a brief audit-report summary for a resistance-training plan "
        "that has already passed deterministic rule checks (clear/pass). "
        "State the outcome in a calm, professional tone.\n\n"
        "Requirements:\n"
        "- 2–3 sentences only.\n"
        "- No exclamation marks. No pep-talk or cheerleading "
        "(e.g. avoid 'You've got this!', 'Keep crushing it', motivational slogans).\n"
        "- If context fields such as inactivity_duration_weeks or weeks_since_return "
        "are present, mention them and explain factually why the stated frequency/"
        "intensity is appropriate for that return-to-training context.\n"
        "- Do not invent new medical claims, diagnoses, or recommendations beyond "
        "what is implied by the plan fields and a clear audit pass.\n"
        "- Stay within the provided plan data; summarize alignment as factual "
        "observation, not encouragement.\n"
        "- Tone example: \"This 2-session-per-week plan aligns with recommended "
        "guidance for resuming training after an extended break, allowing gradual "
        "reconditioning without excessive early-week load.\"\n\n"
        f"Plan fields: {plan_summary}."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content if response.choices else None
    return (content or "").strip()
