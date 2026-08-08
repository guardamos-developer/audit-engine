"""Shared OpenAI client fakes for two-stage (+ parallel stage2 group) tests.

Existing tests can keep calling ``fake_client(full_raw)``: the client returns
stage-/group-projected payloads based on the JSON-schema ``name`` in kwargs
(order-independent for parallel stage-2 groups).
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any, Callable

from src.plan_extractor import (
    STAGE1_FIELD_NAMES,
    STAGE2_ADULT_FIELD_NAMES,
    STAGE2_ADULT_GROUP_A_FIELD_NAMES,
    STAGE2_ADULT_GROUP_B_FIELD_NAMES,
    STAGE2_OLDER_FIELD_NAMES,
    STAGE2_OLDER_GROUP_A_FIELD_NAMES,
    STAGE2_OLDER_GROUP_B_FIELD_NAMES,
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


def _schema_name_from_kwargs(kwargs: dict[str, Any]) -> str:
    rf = kwargs.get("response_format") or {}
    js = rf.get("json_schema") or {}
    return str(js.get("name") or "")


def fake_client_routing(
    *,
    stage1_raw: dict[str, Any],
    stage2_group_a_raw: dict[str, Any] | None = None,
    stage2_group_b_raw: dict[str, Any] | None = None,
    stage2_union_raw: dict[str, Any] | None = None,
    expect_stage2: bool | None = None,
    fail_on_schema: str | None = None,
    fail_exception: BaseException | None = None,
) -> Any:
    """Route ``create`` by schema name; supports parallel A/B and fail injection.

    Parameters
    ----------
    fail_on_schema:
        If set (e.g. ``guardamos_plan_extraction_stage2_b``), that create raises
        ``fail_exception`` (default ``TimeoutError``) to simulate transport failure.
    """
    calls = {"n": 0, "names": []}
    lock = threading.Lock()
    exc = fail_exception if fail_exception is not None else TimeoutError("injected timeout")

    payloads = {
        "guardamos_plan_extraction_stage1": stage1_raw,
        "guardamos_plan_extraction_stage2_a": stage2_group_a_raw,
        "guardamos_plan_extraction_stage2_b": stage2_group_b_raw,
        "guardamos_plan_extraction_stage2": stage2_union_raw,
    }

    class _Completions:
        @staticmethod
        def create(**kwargs: Any):
            with lock:
                calls["n"] += 1
                name = _schema_name_from_kwargs(kwargs)
                calls["names"].append(name)

                if fail_on_schema and name == fail_on_schema:
                    raise exc

                if name == "guardamos_plan_extraction_stage1":
                    payload = payloads[name]
                elif name in (
                    "guardamos_plan_extraction_stage2_a",
                    "guardamos_plan_extraction_stage2_b",
                    "guardamos_plan_extraction_stage2",
                ):
                    if expect_stage2 is False:
                        raise AssertionError(
                            f"stage2 create not expected (schema={name})"
                        )
                    payload = payloads.get(name)
                    if payload is None:
                        raise AssertionError(
                            f"stage2 create for {name} but no payload configured "
                            "(early-exit expected?)"
                        )
                else:
                    # Legacy single-name / unknown: treat as stage1 if first call
                    if calls["n"] == 1:
                        payload = stage1_raw
                    else:
                        payload = (
                            stage2_union_raw
                            or stage2_group_a_raw
                            or stage2_group_b_raw
                        )
                        if payload is None:
                            raise AssertionError(f"unexpected schema name {name!r}")

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


def fake_client_two_stage(
    stage1_raw: dict[str, Any],
    stage2_raw: dict[str, Any] | None = None,
    *,
    expect_stage2: bool | None = None,
) -> Any:
    """Compat: one stage2 blob is projected into both parallel groups."""
    if stage2_raw is None:
        return fake_client_routing(
            stage1_raw=stage1_raw,
            expect_stage2=False if expect_stage2 is None else expect_stage2,
        )
    # Prefer adult group partitions; older-only keys also appear in older groups.
    group_a = _project_raw(
        stage2_raw,
        STAGE2_ADULT_GROUP_A_FIELD_NAMES | STAGE2_OLDER_GROUP_A_FIELD_NAMES,
    )
    group_b = _project_raw(
        stage2_raw,
        STAGE2_ADULT_GROUP_B_FIELD_NAMES | STAGE2_OLDER_GROUP_B_FIELD_NAMES,
    )
    return fake_client_routing(
        stage1_raw=stage1_raw,
        stage2_group_a_raw=group_a,
        stage2_group_b_raw=group_b,
        stage2_union_raw=_project_raw(
            stage2_raw, STAGE2_ADULT_FIELD_NAMES | STAGE2_OLDER_FIELD_NAMES
        ),
        expect_stage2=expect_stage2,
    )


def fake_client(raw_fields: dict[str, Any], *, expect_stage2: bool | None = None) -> Any:
    """Backward-compatible fake: project one full raw into stage1 + stage2 groups."""
    stage1 = _project_raw(raw_fields, STAGE1_FIELD_NAMES)
    stage2_fields = STAGE2_ADULT_FIELD_NAMES | STAGE2_OLDER_FIELD_NAMES
    stage2 = _project_raw(raw_fields, stage2_fields)
    return fake_client_two_stage(stage1, stage2, expect_stage2=expect_stage2)


def fake_client_fail_stage2_group(
    raw_fields: dict[str, Any],
    *,
    fail_group: str = "b",
    fail_exception: BaseException | None = None,
) -> Any:
    """Full raw fake that raises when the given stage2 group schema is requested."""
    stage1 = _project_raw(raw_fields, STAGE1_FIELD_NAMES)
    stage2 = _project_raw(
        raw_fields, STAGE2_ADULT_FIELD_NAMES | STAGE2_OLDER_FIELD_NAMES
    )
    group_a = _project_raw(
        stage2, STAGE2_ADULT_GROUP_A_FIELD_NAMES | STAGE2_OLDER_GROUP_A_FIELD_NAMES
    )
    group_b = _project_raw(
        stage2, STAGE2_ADULT_GROUP_B_FIELD_NAMES | STAGE2_OLDER_GROUP_B_FIELD_NAMES
    )
    g = fail_group.strip().lower()
    return fake_client_routing(
        stage1_raw=stage1,
        stage2_group_a_raw=group_a,
        stage2_group_b_raw=group_b,
        expect_stage2=True,
        fail_on_schema=f"guardamos_plan_extraction_stage2_{g}",
        fail_exception=fail_exception,
    )


# Aliases matching historical local helper names in test modules.
_fake_client = fake_client
_fake_client_two_stage = fake_client_two_stage
_empty_raw_fields = empty_raw_fields
