"""Layer1 deterministic rule evaluation engine.

Framework-agnostic: importable as a standalone module for Dify code nodes.
"""

from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import Any

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"

DEFAULT_RULESET_FILES = (
    "layer1_rules_acsm_rt_v1.json",
    "layer1_rules_cscca_return_to_training_v1.json",
)

# Only these rule_ids are evaluated. Add IDs here to activate future rules.
ACTIVE_RULE_IDS = frozenset(
    {
        # ACSM 2026 RT
        "L1-RT-0001",
        "L1-RT-0002",
        "L1-RT-0003",
        "L1-RT-0004",
        "L1-RT-0005",
        "L1-RT-0006",
        "L1-RT-0007",
        "L1-RT-0008",
        "L1-RT-0009",
        "L1-RT-0010",
        # CSCCa return-to-training (demo scope)
        "L1-RTT-0001",
        "L1-RTT-0002a",
        "L1-RTT-0002b",
        "L1-RTT-0002c",
        "L1-RTT-0002d",
        "L1-RTT-0002e",
        "L1-RTT-0002f",
        "L1-RTT-0002g",
        "L1-RTT-0002h",
    }
)

DEFAULT_RULES_PATH = RULES_DIR / "layer1_rules_acsm_rt_v1.json"

_COMPARISON_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

# field op <param_name | numeric_literal>
_NUMERIC_CLAUSE_RE = re.compile(
    r"^(?P<field>\w+)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<rhs>-?\d+(?:\.\d+)?|\w+)$"
)
_BOOLEAN_CHECK_RE = re.compile(
    r"^(?P<field>\w+)\s*==\s*(?P<value>true|false)$", re.IGNORECASE
)
_RANGE_CHECK_RE = re.compile(
    r"^(?P<field>\w+)\s+NOT\s+BETWEEN\s+(?P<min_param>\w+)\s+AND\s+(?P<max_param>\w+)$",
    re.IGNORECASE,
)
_NUMERIC_LITERAL_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def load_ruleset(rules_path: str | Path | None = None) -> dict[str, Any]:
    """Load a single ruleset JSON file."""
    path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_merged_rulesets(
    rules_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Load and merge multiple ruleset files into one rules list."""
    if rules_paths is None:
        paths = [RULES_DIR / name for name in DEFAULT_RULESET_FILES]
    else:
        paths = [Path(p) for p in rules_paths]

    merged_rules: list[dict] = []
    ruleset_ids: list[str] = []
    for path in paths:
        data = load_ruleset(path)
        rid = data.get("ruleset_id")
        if rid:
            ruleset_ids.append(str(rid))
        merged_rules.extend(data.get("rules") or [])

    return {
        "ruleset_id": "+".join(ruleset_ids) if ruleset_ids else "merged",
        "rules": merged_rules,
    }


def _plan_exclusion_tags(plan: dict) -> set[str]:
    """Collect population / experience tags that may appear in rule excludes."""
    tags: set[str] = set()
    target = plan.get("target_population")
    if isinstance(target, str):
        tags.add(target)
    experience = plan.get("experience_level")
    if isinstance(experience, str):
        tags.add(experience)
    for key in ("excludes", "flags", "attributes"):
        value = plan.get(key)
        if isinstance(value, list):
            tags.update(str(v) for v in value)
        elif isinstance(value, str):
            tags.add(value)
    return tags


def _is_long_inactivity_context(plan: dict) -> bool:
    """True when the plan is in a long-inactivity return-to-training context."""
    weeks = plan.get("inactivity_duration_weeks")
    if weeks is not None and weeks >= 4:
        return True
    if plan.get("plan_follows_long_inactivity_track") is True:
        return True
    if plan.get("population_is_long_inactivity") is True:
        return True
    return "long_inactivity" in _plan_exclusion_tags(plan)


def _rule_applies(rule: dict, plan: dict) -> bool:
    """Return False if the plan is outside this rule's applicability scope."""
    applicability = rule.get("applicability") or {}
    goal = applicability.get("goal")
    if goal is not None and plan.get("goal") != goal:
        return False

    excludes = set(applicability.get("excludes") or [])
    if excludes & _plan_exclusion_tags(plan):
        # Gate rule handles population excludes separately; for other rules
        # (e.g. true_beginner_first_weeks on L1-RT-0004) skip matching.
        if rule.get("rule_type") != "applicability_gate":
            return False

    population = applicability.get("population") or []
    # Rules tagged for long_inactivity only apply in that return context.
    if "long_inactivity" in population and not _is_long_inactivity_context(plan):
        return False

    return True


def _resolve_plan_value(plan: dict, field: str) -> Any:
    """Resolve a field from top-level plan or the current week's parameters."""
    if field in plan and plan.get(field) is not None:
        return plan.get(field)

    week_n = plan.get("weeks_since_return")
    week_params_root = plan.get("plan_week_parameters") or {}
    if week_n is not None and isinstance(week_params_root, dict):
        observed_week = week_params_root.get(str(week_n))
        if observed_week is None:
            observed_week = week_params_root.get(week_n)
        if isinstance(observed_week, dict) and observed_week.get(field) is not None:
            return observed_week.get(field)

    # Common aliases used across ACSM vs CSCCa field names
    if field == "frequency_days_per_week" and plan.get("sessions_per_week") is not None:
        return plan.get("sessions_per_week")
    if field == "intensity_percent_1RM" and plan.get("load_percent_1RM") is not None:
        return plan.get("load_percent_1RM")

    return None


def _evaluate_population_check(condition: dict, plan: dict, rule: dict) -> tuple[bool, dict]:
    applicability = rule.get("applicability") or {}
    population = set(applicability.get("population") or [])
    excludes = set(applicability.get("excludes") or [])
    target = plan.get("target_population")

    out_of_scope = target not in population or target in excludes
    matched_parameters = {
        "target_population": target,
        "allowed_population": sorted(population),
        "excludes": sorted(excludes),
    }
    return out_of_scope, matched_parameters


def _resolve_rhs(rhs: str, params: dict) -> Any:
    if _NUMERIC_LITERAL_RE.match(rhs):
        return float(rhs) if "." in rhs else int(rhs)
    if rhs in params:
        return params[rhs]
    raise ValueError(f"Unknown threshold rhs {rhs!r}; not a literal or parameter")


def _evaluate_numeric_clause(
    clause: str, plan: dict, params: dict
) -> tuple[bool, dict | None]:
    """Evaluate one `field op rhs` clause. Missing observed values → False."""
    match = _NUMERIC_CLAUSE_RE.match(clause.strip())
    if not match:
        raise ValueError(f"Unsupported numeric_threshold clause: {clause!r}")

    field = match.group("field")
    op_symbol = match.group("op")
    rhs_token = match.group("rhs")
    threshold = _resolve_rhs(rhs_token, params)

    observed = _resolve_plan_value(plan, field)
    if observed is None:
        return False, None

    op_fn = _COMPARISON_OPS[op_symbol]
    hit = op_fn(observed, threshold)
    info = {
        "field": field,
        "observed_value": observed,
        "threshold": threshold,
        "rhs_token": rhs_token,
    }
    return hit, info


def _evaluate_numeric_threshold(condition: dict, plan: dict) -> tuple[bool, dict]:
    """Evaluate numeric_threshold checks, including AND-composed clauses.

    Supports both:
      - legacy: ``sessions_per_week < min_sessions_per_week``
      - CSCCa: ``weeks_since_return == 1 AND sets_per_exercise > 2``
    """
    check = condition.get("check", "")
    params = dict(condition.get("parameters") or {})
    clauses = [c.strip() for c in re.split(r"\s+AND\s+", check.strip()) if c.strip()]
    if not clauses:
        return False, {}

    clause_infos: list[dict] = []
    for clause in clauses:
        hit, info = _evaluate_numeric_clause(clause, plan, params)
        if not hit:
            return False, {}
        if info is not None:
            clause_infos.append(info)

    # Prefer a non-week field for {observed_value} in reason templates.
    primary = next(
        (i for i in reversed(clause_infos) if i["field"] != "weeks_since_return"),
        clause_infos[-1] if clause_infos else None,
    )
    matched_parameters = {**params}
    if primary is not None:
        matched_parameters["observed_value"] = primary["observed_value"]
        matched_parameters["field"] = primary["field"]
        if primary["rhs_token"] not in params and not _NUMERIC_LITERAL_RE.match(
            str(primary["rhs_token"])
        ):
            matched_parameters[primary["rhs_token"]] = primary["threshold"]
    return True, matched_parameters


def _evaluate_range_check(condition: dict, plan: dict) -> tuple[bool, dict]:
    check = condition.get("check", "")
    params = condition.get("parameters") or {}
    match = _RANGE_CHECK_RE.match(check.strip())
    if not match:
        raise ValueError(f"Unsupported range_check check: {check!r}")

    field = match.group("field")
    min_param = match.group("min_param")
    max_param = match.group("max_param")
    if min_param not in params or max_param not in params:
        raise ValueError(f"Range parameters missing for check {check!r}")

    observed = _resolve_plan_value(plan, field)
    lo = params[min_param]
    hi = params[max_param]
    if observed is None:
        return False, {}

    # "NOT BETWEEN min AND max" → outside inclusive range
    hit = not (lo <= observed <= hi)
    matched_parameters = {
        "observed_value": observed,
        "field": field,
        min_param: lo,
        max_param: hi,
    }
    return hit, matched_parameters


def _evaluate_boolean_check(condition: dict, plan: dict) -> tuple[bool, dict]:
    check = condition.get("check", "")
    match = _BOOLEAN_CHECK_RE.match(check.strip())
    if not match:
        raise ValueError(f"Unsupported boolean_check check: {check!r}")

    field = match.group("field")
    expected = match.group("value").lower() == "true"
    observed = _resolve_plan_value(plan, field)
    if observed is None:
        return False, {}

    if isinstance(observed, bool):
        hit = observed is expected
    else:
        hit = bool(observed) == expected

    matched_parameters = {
        "observed_value": observed,
        "field": field,
        "expected": expected,
    }
    return hit, matched_parameters


def _evaluate_context_gate(condition: dict, plan: dict) -> tuple[bool, dict]:
    """Long-inactivity track selection gate (L1-RTT-0001).

    Matches when inactivity_duration_weeks >= threshold AND
    plan_follows_long_inactivity_track is false/missing.
    """
    params = condition.get("parameters") or {}
    threshold = params.get("long_inactivity_threshold_weeks", 4)
    weeks = plan.get("inactivity_duration_weeks")
    if weeks is None:
        return False, {}

    follows_track = plan.get("plan_follows_long_inactivity_track")
    if follows_track is None:
        follows_track = False

    hit = weeks >= threshold and follows_track is False
    matched_parameters = {
        "observed_value": weeks,
        "long_inactivity_threshold_weeks": threshold,
        "plan_follows_long_inactivity_track": follows_track,
    }
    return hit, matched_parameters


def evaluate_condition(condition: dict, plan: dict, rule: dict) -> tuple[bool, dict]:
    """Dispatch to a type-specific evaluator. Never uses eval()."""
    cond_type = condition.get("type")
    if cond_type == "population_check":
        return _evaluate_population_check(condition, plan, rule)
    if cond_type == "numeric_threshold":
        return _evaluate_numeric_threshold(condition, plan)
    if cond_type == "range_check":
        return _evaluate_range_check(condition, plan)
    if cond_type == "boolean_check":
        return _evaluate_boolean_check(condition, plan)
    if cond_type == "context_gate":
        return _evaluate_context_gate(condition, plan)
    raise ValueError(f"Unknown condition type: {cond_type!r}")


def _match_result(rule: dict, matched_parameters: dict) -> dict:
    return {
        "rule_id": rule["rule_id"],
        "action": rule["action"],
        "severity": rule["severity"],
        "matched_parameters": matched_parameters,
        "reason_template": rule.get("reason_template", {}),
    }


def _is_population_gate(rule: dict) -> bool:
    return rule.get("rule_type") == "applicability_gate" or (
        (rule.get("condition") or {}).get("type") == "population_check"
    )


def _is_context_gate(rule: dict) -> bool:
    return (rule.get("condition") or {}).get("type") == "context_gate"


def evaluate_layer1(
    plan: dict,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> list[dict]:
    """Evaluate Layer1 rules against a structured training plan.

    Order:
      1. Population applicability gate (L1-RT-0001) — short-circuit if matched
      2. Context gates (L1-RTT-0001) — evaluated early; matches are collected
         (does not skip remaining rules, so week-specific RTT rules can also fire)
      3. Remaining active rules

    Inactive rule_ids (not in ACTIVE_RULE_IDS) are loaded but never evaluated.
    """
    if ruleset is not None:
        data = ruleset
    elif rules_path is not None:
        data = load_ruleset(rules_path)
    else:
        data = load_merged_rulesets()

    rules: list[dict] = [
        r for r in (data.get("rules") or []) if r.get("rule_id") in ACTIVE_RULE_IDS
    ]

    population_gates = [r for r in rules if _is_population_gate(r)]
    context_gates = [r for r in rules if _is_context_gate(r) and not _is_population_gate(r)]
    other_rules = [
        r for r in rules if not _is_population_gate(r) and not _is_context_gate(r)
    ]

    for rule in population_gates:
        condition = rule.get("condition") or {}
        hit, matched_parameters = evaluate_condition(condition, plan, rule)
        if hit:
            return [_match_result(rule, matched_parameters)]

    matches: list[dict] = []

    # Context gates first (early), but continue so week-specific RTT rules can also match.
    # TODO: Layer2実装後は context_gate の即時reject vs routing を再検討
    for rule in context_gates:
        if not _rule_applies(rule, plan):
            continue
        condition = rule.get("condition") or {}
        hit, matched_parameters = evaluate_condition(condition, plan, rule)
        if hit:
            matches.append(_match_result(rule, matched_parameters))

    for rule in other_rules:
        if not _rule_applies(rule, plan):
            continue
        condition = rule.get("condition") or {}
        hit, matched_parameters = evaluate_condition(condition, plan, rule)
        if hit:
            matches.append(_match_result(rule, matched_parameters))
    return matches
