"""Deterministic explanation rendering from reason_template (no LLM)."""

from __future__ import annotations

import re


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def render_explanation(
    rule: dict,
    matched_parameters: dict,
    lang: str = "en",
) -> str:
    """Render a rule's reason_template for ``lang`` via string substitution only.

    Falls back to English if the requested language is missing.
    """
    templates = rule.get("reason_template") or {}
    if not isinstance(templates, dict):
        return ""

    template = templates.get(lang) or templates.get("en") or ""
    if not template:
        return ""

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in matched_parameters:
            return str(matched_parameters[key])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)
