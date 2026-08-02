"""Deterministic explanation rendering from reason_template (no LLM)."""

from __future__ import annotations

import re


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _select_template(reason_template: dict, *, side: str, lang: str) -> str:
    """Pick a language string from either nested flagged/pass or legacy flat form."""
    if not isinstance(reason_template, dict):
        return ""

    # New shape: {"flagged": {"en": ...}, "pass": {"en": ...}}
    branch = reason_template.get(side)
    if isinstance(branch, dict):
        return branch.get(lang) or branch.get("en") or ""

    # Legacy flat shape: {"en": ..., "pt": ..., "ja": ...}
    # Only used for flagged-side rendering of older payloads.
    if side == "flagged":
        return reason_template.get(lang) or reason_template.get("en") or ""
    return ""


def render_explanation(
    rule: dict,
    matched_parameters: dict,
    lang: str = "en",
    *,
    side: str = "flagged",
) -> str:
    """Render a rule's reason_template for ``lang`` via string substitution only.

    ``side`` selects ``reason_template["flagged"]`` or ``reason_template["pass"]``.
    Falls back to English if the requested language is missing.
    """
    templates = rule.get("reason_template") or {}
    template = _select_template(templates, side=side, lang=lang)
    if not template:
        return ""

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in matched_parameters:
            return str(matched_parameters[key])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)
