"""Layer3 free-form response generation via OpenAI API."""

from __future__ import annotations

import os

CLEAR_VERDICTS = frozenset({"clear", "pass"})
BLOCKED_VERDICTS = frozenset({"rejected", "flagged"})


def generate_layer3_response(
    plan: dict,
    layer1_2_verdict: str,
    checked_facts: list[dict] | None = None,
) -> str:
    """Generate a brief summary from Layer1-B pass facts when the verdict is clear.

    When ``layer1_2_verdict`` is ``rejected`` or ``flagged``, skips the LLM
    call and returns an empty string. Does not use raw plan fields — only the
    structured ``checked_facts`` list from Layer1-B.
    """
    verdict = (layer1_2_verdict or "").strip().lower()
    if verdict in BLOCKED_VERDICTS:
        return ""
    if verdict not in CLEAR_VERDICTS:
        return ""

    facts = checked_facts or []
    if not facts:
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
    fact_lines = "\n".join(
        f"- [{f.get('rule_id', '?')}] {f.get('text', '').strip()}"
        for f in facts
        if f.get("text")
    )

    prompt = (
        "You are writing a brief audit-report summary. The plan has already "
        "passed deterministic Layer1 checks. You are given only the confirmed "
        "pass facts below.\n\n"
        "Requirements:\n"
        "- Write 1–2 sentences in a calm, professional audit-report tone.\n"
        "- Paraphrase only the provided facts. Do not add any claim, number, "
        "recommendation, or medical assertion that is not already stated in "
        "those facts.\n"
        "- No exclamation marks. No pep-talk or motivational language.\n"
        "- If a fact mentions return-from-inactivity / week-1 context, you may "
        "reflect that context, but only using wording supported by the facts.\n\n"
        f"Confirmed pass facts:\n{fact_lines}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content if response.choices else None
    return (content or "").strip()
