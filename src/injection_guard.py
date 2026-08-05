"""Deterministic prompt-injection phrase detection (no LLM)."""

from __future__ import annotations

import re

# Provisional phrase list drawn from observed injection attempts.
# Extend when new phrasings appear; keep entries specific enough to avoid
# flagging ordinary fitness-plan language.
_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore the above",
    "disregard previous instructions",
    "note to the system",
    "system note",
    "should be recorded as",
    "when extracting",
    "for the extraction process",
    "override this value",
    "the system processing this",
)

_COMPILED: tuple[re.Pattern[str], ...] = tuple(
    re.compile(re.escape(phrase), re.IGNORECASE) for phrase in _INJECTION_PATTERNS
)


def detect_injection_patterns(text: str) -> list[str]:
    """
    Scans raw text for phrases that appear to address the extraction
    system itself, rather than describing a fitness plan. Pure string/regex
    matching — no LLM involved. Returns the list of matched patterns
    (empty list if none found).
    """
    if not text:
        return []
    matched: list[str] = []
    for phrase, pattern in zip(_INJECTION_PATTERNS, _COMPILED):
        if pattern.search(text):
            matched.append(phrase)
    return matched
