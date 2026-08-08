"""Shared OpenAI client fakes for two-stage extraction tests.

Existing tests can keep calling ``_fake_client(full_raw)``: the client returns
stage-projected payloads per call so most fixtures need no rewrite.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.plan_extractor import (
    STAGE1_FIELD_NAMES,
    STAGE2_ADULT_FIELD_NAMES,
    STAGE2_OLDER_FIELD_NAMES,
    _PLAN_FIELD_SPECS,
)


def empty_raw_fields(**overrides: Any) -> dict[str, Any]:
    """Build a full extraction payload with null plan fields + meta defaults."""
    raw: dict[str, Any] = {
        name: {"value": None, "evidence_quote": None} for name in _PLAN_FIELD_SPECS
    }
    raw["possible_meta_instruction_detected"] = False
    raw["meta_instruction_evidence"] = None
    for name, entry in overrides.items():
        raw[name] = entry
    return raw


def _project_raw(raw: dict[str, Any], field_names: frozenset[str] | set[str]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for name in field_names:
        entry = raw.get(name)
        if isinstance(entry, dict):
            projected[name] = entry
        else:
            projected[name] = {"value": None, "evidence_quote": None}
    projected["possible_meta_instruction_detected"] = bool(
        raw.get("possible_meta_instruction_detected", False)
    )
    evidence = raw.get("meta_instruction_evidence")
    projected["meta_instruction_evidence"] = evidence
    return projected


def fake_client_two_stage(
    stage1_raw: dict[str, Any],
    stage2_raw: dict[str, Any] | None = None,
    *,
    expect_stage2: bool | None = None,
) -> Any:
    """OpenAI-like client that serves stage1 then optional stage2 payloads.

    Parameters
    ----------
    stage1_raw, stage2_raw:
        Raw structured-output dicts (field → {value, evidence_quote}, plus meta).
    expect_stage2:
        If False, a second ``create`` call fails the test.
        If True, a missing second call is not asserted here (callers may check
        ``stage2_ran`` on the extract_plan result).
        If None, a second call is allowed when ``stage2_raw`` is not None;
        when ``stage2_raw`` is None, a second call raises.
    """
    calls = {"n": 0}

    class _Completions:
        @staticmethod
        def create(**kwargs: Any):
            calls["n"] += 1
            n = calls["n"]
            if n == 1:
                payload = stage1_raw
            elif n == 2:
                if expect_stage2 is False:
                    raise AssertionError(
                        "stage2 chat.completions.create was not expected"
                    )
                if stage2_raw is None:
                    raise AssertionError(
                        "stage2 create called but stage2_raw is None "
                        "(early-exit expected?)"
                    )
                payload = stage2_raw
            else:
                raise AssertionError(f"unexpected create call #{n}")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload))
                    )
                ]
            )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
        _calls=calls,
    )


def fake_client(raw_fields: dict[str, Any], *, expect_stage2: bool | None = None) -> Any:
    """Backward-compatible fake: project one full raw into stage1 + stage2.

    Call 1 returns stage-1 keys (projected from ``raw_fields``).
    Call 2 returns the union of adult and older stage-2 keys projected from
    the same raw (so either population can pass without rewriting fixtures).
    """
    stage1 = _project_raw(raw_fields, STAGE1_FIELD_NAMES)
    stage2_fields = STAGE2_ADULT_FIELD_NAMES | STAGE2_OLDER_FIELD_NAMES
    stage2 = _project_raw(raw_fields, stage2_fields)
    # If the caller only populated stage-1 exclusions (gate reject), stage2 may
    # never run — leave expect_stage2 as provided.
    return fake_client_two_stage(stage1, stage2, expect_stage2=expect_stage2)


# Aliases matching historical local helper names in test modules.
_fake_client = fake_client
_fake_client_two_stage = fake_client_two_stage
_empty_raw_fields = empty_raw_fields
