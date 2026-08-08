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
    "layer1_rules_nsca_older_adults_v1.json",
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
        # CSCCa moderate return track (2-to-<4-week break)
        "L1-RTT-0008",
        "L1-RTT-0009",
        "L1-RTT-0010",
        # ECSS/ACSM overtraining (rest / recovery)
        "L1-ECSS-0001",
        "L1-ECSS-0002",
        # CSCCa FIT / rhabdo / medical clearance (Tier 3 — verified)
        "L1-RTT-0003",
        "L1-RTT-0004",
        "L1-RTT-0005",
        "L1-RTT-0006",
        # Relative load/volume caution (qualitative signal; not numeric compliance)
        "L1-RTT-0011",
        # NSCA older adults Table 1 (healthy older adults)
        "L1-RT-NSCA-0001",
        "L1-RT-NSCA-0002",
        "L1-RT-NSCA-0003",
        "L1-RT-NSCA-0004",
        "L1-RT-NSCA-0005",
        # NSCA older adults Table 3 (condition-specific accommodations)
        "L1-RT-NSCA-0006",
        "L1-RT-NSCA-0007",
        "L1-RT-NSCA-0008",
        "L1-RT-NSCA-0009",
        "L1-RT-NSCA-0010",
        "L1-RT-NSCA-0011",
        # NSCA older-adult mirrors of ACSM caution rules L1-RT-0007–0010
        "L1-RT-NSCA-0012",
        "L1-RT-NSCA-0013",
        "L1-RT-NSCA-0014",
        "L1-RT-NSCA-0015",
    }
)

# Mutually exclusive primary population classes (not co-occurring tags).
_PRIMARY_POPULATIONS = frozenset({"healthy_adult_18plus", "older_adult_healthy"})

# Provisional audit-side age cutoff for older_adult_healthy routing.
# Not a source-stated figure (same convention as the 4-week threshold in L1-RTT-0001).
_OLDER_ADULT_AGE_THRESHOLD_YEARS = 65

# Boolean population-exclusion flags that may appear in applicability.excludes.
_EXCLUSION_BOOLEAN_FLAGS = frozenset(
    {
        "injury_present",
        "post_surgical",
        "pain_present",
        "minor",
        "pregnant",
        "frailty_present",
        "uncontrolled_hypertension",
        "unstable_cardiovascular_disease",
        "cardiovascular_disease_present",
        "osteoporosis_present",
        "true_beginner_first_weeks",
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
    # Boolean exclusion flags from plan_extractor / structured plans.
    for flag in _EXCLUSION_BOOLEAN_FLAGS:
        if plan.get(flag) is True:
            tags.add(flag)
    for key in ("excludes", "flags", "attributes"):
        value = plan.get(key)
        if isinstance(value, list):
            tags.update(str(v) for v in value)
        elif isinstance(value, str):
            tags.add(value)
    return tags


def effective_target_population(plan: dict) -> str:
    """Return the mutually exclusive primary population class for this plan.

    Routing rules (audit-side design, not source-stated cutoffs), in order:
      1. age_years present → numeric exclusivity
         (>=65 → older_adult_healthy; otherwise → healthy_adult_18plus;
         minors under 18 still use the general-adult gate with minor=True)
      2. age_years null + stated_age_category == "older_adult"
         → older_adult_healthy
      3. age_years null + stated_age_category in {"minor", "adult"}
         → healthy_adult_18plus (minor self-ID forces minor via
         ``apply_deterministic_age_derived_flags``)
      4. else explicit target_population if it is a primary class
      5. else default healthy_adult_18plus

    Numeric ``age_years`` always overrides ``stated_age_category`` /
    ``target_population`` so exclusivity cannot be bypassed by a stale tag.
    """
    age = plan.get("age_years")
    if age is not None:
        try:
            age_n = int(age)
        except (TypeError, ValueError):
            age_n = None
        if age_n is not None:
            if age_n >= _OLDER_ADULT_AGE_THRESHOLD_YEARS:
                return "older_adult_healthy"
            return "healthy_adult_18plus"

    category = plan.get("stated_age_category")
    if category == "older_adult":
        return "older_adult_healthy"
    if category in ("minor", "adult"):
        return "healthy_adult_18plus"

    target = plan.get("target_population")
    if isinstance(target, str) and target in _PRIMARY_POPULATIONS:
        return target
    return "healthy_adult_18plus"


def apply_deterministic_age_derived_flags(plan: dict) -> dict:
    """Force exclusion flags that follow from age signals (Zone B).

    When a definite numeric age is known, minor status must not depend on LLM
    language interpretation: if age_years < 18, set ``minor=True`` regardless
    of whether the extractor left minor as true, false, or null.

    When age_years is null but ``stated_age_category == "minor"``, also force
    ``minor=True`` so qualitative underage self-ID cannot skip the population
    gate.

    Symmetric with age >= 65 / stated_age_category older_adult routing in
    ``effective_target_population``: these are deterministic gates, not
    extraction guesses.
    """
    out = dict(plan)
    age = out.get("age_years")
    if age is not None:
        try:
            age_n = int(age)
        except (TypeError, ValueError):
            age_n = None
        else:
            # Age is a concrete number: minorhood is decided by comparison.
            if age_n < 18:
                out["minor"] = True
            return out

    if out.get("stated_age_category") == "minor":
        out["minor"] = True
    return out


def _is_long_inactivity_context(plan: dict) -> bool:
    """True when the plan is in a long-inactivity return-to-training context."""
    weeks = plan.get("inactivity_duration_weeks")
    if weeks is not None and weeks >= 4:
        return True
    if plan.get("population_is_long_inactivity") is True:
        return True
    return "long_inactivity" in _plan_exclusion_tags(plan)


def _is_moderate_return_context(plan: dict) -> bool:
    """True when the plan is in the 2-to-<4-week moderate / returning-athlete track."""
    weeks = plan.get("inactivity_duration_weeks")
    if weeks is not None and 2 <= weeks < 4:
        return True
    if plan.get("population_is_moderate_return") is True:
        return True
    if plan.get("population_is_moderate_inactivity") is True:
        return True
    tags = _plan_exclusion_tags(plan)
    return bool(tags & {"moderate_return", "moderate_inactivity"})


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
    # Mutually exclusive primary classes: ACSM general-adult vs NSCA older-adult.
    primary = _PRIMARY_POPULATIONS & set(population)
    if primary and effective_target_population(plan) not in primary:
        return False
    # Rules tagged for long_inactivity only apply in that return context.
    if "long_inactivity" in population and not _is_long_inactivity_context(plan):
        return False
    # Moderate / returning-athlete track (JSON uses moderate_inactivity or moderate_return).
    if (
        {"moderate_return", "moderate_inactivity"} & set(population)
        and not _is_moderate_return_context(plan)
    ):
        return False

    return True


def _resolve_plan_value(plan: dict, field: str) -> Any:
    """Resolve a field from top-level plan or the current week's parameters."""
    if field in plan and plan.get(field) is not None:
        return plan.get(field)

    # Derived long-inactivity flag (used in L1-RTT-0003/0005/0006 compound checks).
    if field == "population_is_long_inactivity":
        weeks = plan.get("inactivity_duration_weeks")
        if weeks is not None:
            return weeks >= 4
        if "long_inactivity" in _plan_exclusion_tags(plan):
            return True
        return None

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


def _enrich_week_dependent_params(params: dict, plan: dict) -> dict:
    """Resolve week-scoped aliases such as ``min_denominator_for_week`` (L1-RTT-0004)."""
    out = dict(params)
    week = plan.get("weeks_since_return")
    if "week_1_min_denominator" in out or "week_2_min_denominator" in out:
        if week == 1 and "week_1_min_denominator" in out:
            out["min_denominator_for_week"] = out["week_1_min_denominator"]
        elif week == 2 and "week_2_min_denominator" in out:
            out["min_denominator_for_week"] = out["week_2_min_denominator"]
    return out


_ATOMIC_CLAUSE_RE = re.compile(
    r"^(?P<field>\w+)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<rhs>true|false|-?\d+(?:\.\d+)?|\w+)$",
    re.IGNORECASE,
)
_FIELD_LHS_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:==|!=|<=|>=|<|>)"
)
_RESERVED_TOKENS = frozenset({"true", "false", "and", "or", "not", "between"})


def _split_top_level(expr: str, connector: str) -> list[str]:
    """Split ``expr`` on top-level AND/OR (ignoring parentheses)."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    tokens = re.split(rf"(\s+{connector}\s+|\(|\))", expr, flags=re.IGNORECASE)
    for tok in tokens:
        if not tok:
            continue
        if tok == "(":
            depth += 1
            buf.append(tok)
        elif tok == ")":
            depth = max(0, depth - 1)
            buf.append(tok)
        elif depth == 0 and re.fullmatch(rf"\s+{connector}\s+", tok, re.IGNORECASE):
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(tok)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_outer_parens(expr: str) -> str:
    text = expr.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            text = text[1:-1].strip()
        else:
            break
    return text


def _referenced_fields_in_check(check: str) -> list[str]:
    """Extract LHS field names from a compound check expression (order preserved)."""
    fields: list[str] = []
    seen: set[str] = set()
    for match in _FIELD_LHS_RE.finditer(check or ""):
        name = match.group(1)
        if name.lower() in _RESERVED_TOKENS:
            continue
        if name not in seen:
            seen.add(name)
            fields.append(name)
    return fields


def _evaluate_atomic_clause(
    clause: str, plan: dict, params: dict
) -> tuple[bool | None, dict | None]:
    """Evaluate one atomic comparison. Returns (hit, info); hit=None if unevaluable."""
    text = _strip_outer_parens(clause)
    match = _ATOMIC_CLAUSE_RE.match(text)
    if not match:
        raise ValueError(f"Unsupported atomic clause: {clause!r}")

    field = match.group("field")
    op_symbol = match.group("op")
    rhs_token = match.group("rhs")
    observed = _resolve_plan_value(plan, field)
    if observed is None:
        return None, None

    rhs_lower = rhs_token.lower()
    if rhs_lower in {"true", "false"}:
        expected = rhs_lower == "true"
        if isinstance(observed, bool):
            hit = (observed is expected) if op_symbol == "==" else (observed is not expected)
        else:
            hit = (bool(observed) == expected) if op_symbol == "==" else (
                bool(observed) != expected
            )
        return hit, {
            "field": field,
            "observed_value": observed,
            "expected": expected,
            "rhs_token": rhs_token,
        }

    threshold = _resolve_rhs(rhs_token, params)
    op_fn = _COMPARISON_OPS[op_symbol]
    hit = op_fn(observed, threshold)
    return hit, {
        "field": field,
        "observed_value": observed,
        "threshold": threshold,
        "rhs_token": rhs_token,
    }


def _evaluate_compound_expression(
    expr: str, plan: dict, params: dict
) -> tuple[bool | None, list[dict]]:
    """Evaluate AND/OR compound expressions. hit=None when any atom is unevaluable."""
    text = _strip_outer_parens(expr)
    and_parts = _split_top_level(text, "AND")
    if len(and_parts) > 1:
        infos: list[dict] = []
        for part in and_parts:
            hit, part_infos = _evaluate_compound_expression(part, plan, params)
            if hit is None:
                return None, []
            if not hit:
                return False, []
            infos.extend(part_infos)
        return True, infos

    or_parts = _split_top_level(text, "OR")
    if len(or_parts) > 1:
        infos = []
        any_true = False
        for part in or_parts:
            hit, part_infos = _evaluate_compound_expression(part, plan, params)
            if hit is None:
                return None, []
            if hit:
                any_true = True
                infos.extend(part_infos)
        return any_true, infos

    hit, info = _evaluate_atomic_clause(text, plan, params)
    if hit is None:
        return None, []
    return hit, [info] if info is not None else []


def _positive_exclusion_flags(plan: dict, excludes: set[str]) -> set[str]:
    """Return exclude tags that are affirmatively present on the plan.

    A missing / null flag is NOT treated as out-of-scope — only explicit
    positives (boolean True, or target_population equal to an exclude tag).
    """
    positive: set[str] = set()
    for flag in excludes:
        if plan.get(flag) is True:
            positive.add(flag)
    target = plan.get("target_population")
    if isinstance(target, str) and target in excludes:
        positive.add(target)
    return positive


def _evaluate_population_check(condition: dict, plan: dict, rule: dict) -> tuple[bool, dict]:
    """Applicability gate (L1-RT-0001): reject only on positive exclusion flags.

    Previously this also fired when ``target_population`` was null / unknown
    (``target not in population``). That incorrectly short-circuited audits when
    the extractor left the field unset. Null exclusion flags now mean
    "unknown → assume in-scope healthy adult" and continue to later rules.
    """
    applicability = rule.get("applicability") or {}
    population = set(applicability.get("population") or [])
    excludes = set(applicability.get("excludes") or [])
    target = plan.get("target_population")
    positive = _positive_exclusion_flags(plan, excludes)

    out_of_scope = bool(positive)
    triggered = sorted(positive)
    matched_parameters = {
        "target_population": target,
        "allowed_population": sorted(population),
        "excludes": sorted(excludes),
        "positive_exclusion_flags": triggered,
        # Comma-separated for reason_template ``{triggered_exclusions}``.
        "triggered_exclusions": ", ".join(triggered) if triggered else "",
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
      - simple: ``sessions_per_week < min_sessions_per_week``
      - CSCCa compound: ``weeks_since_return == 1 AND sets_per_exercise > 2``
      - week-scoped aliases: ``work_rest_ratio_denominator < min_denominator_for_week``
    """
    check = condition.get("check", "")
    params = _enrich_week_dependent_params(condition.get("parameters") or {}, plan)
    hit, clause_infos = _evaluate_compound_expression(check, plan, params)
    if hit is not True:
        return False, {}

    # Prefer a non-week field for {observed_value} in reason templates.
    primary = next(
        (i for i in reversed(clause_infos) if i.get("field") != "weeks_since_return"),
        clause_infos[-1] if clause_infos else None,
    )
    matched_parameters = {**params}
    if primary is not None:
        matched_parameters["observed_value"] = primary["observed_value"]
        matched_parameters["field"] = primary["field"]
        rhs_token = primary.get("rhs_token")
        if (
            rhs_token is not None
            and rhs_token not in params
            and not _NUMERIC_LITERAL_RE.match(str(rhs_token))
            and "threshold" in primary
        ):
            matched_parameters[rhs_token] = primary["threshold"]
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
    """Evaluate boolean_check, including AND/OR compound expressions.

    Compound checks may mix boolean and numeric atoms (e.g. L1-RTT-0003/0005).
    """
    check = condition.get("check", "")
    params = _enrich_week_dependent_params(condition.get("parameters") or {}, plan)
    hit, clause_infos = _evaluate_compound_expression(check, plan, params)
    if hit is not True:
        return False, {}

    primary = next(
        (
            i
            for i in reversed(clause_infos)
            if i.get("field")
            not in {"weeks_since_return", "population_is_long_inactivity"}
        ),
        clause_infos[-1] if clause_infos else None,
    )
    matched_parameters = {**params}
    if primary is not None:
        matched_parameters["observed_value"] = primary.get("observed_value")
        matched_parameters["field"] = primary.get("field")
        if "expected" in primary:
            matched_parameters["expected"] = primary["expected"]
        if "threshold" in primary:
            matched_parameters["threshold"] = primary["threshold"]
    return True, matched_parameters


def _is_moderate_context_gate(condition: dict) -> bool:
    """Detect the 2-to-<4-week returning-athlete gate (vs long-inactivity gate)."""
    params = condition.get("parameters") or {}
    check = condition.get("check") or ""
    if "plan_follows_moderate_return_track" in check:
        return True
    if "moderate_min_weeks" in params or "moderate_max_weeks" in params:
        return True
    # Canonical JSON uses min_weeks/max_weeks for L1-RTT-0008.
    if "min_weeks" in params and "max_weeks" in params:
        return True
    return False


def _moderate_week_bounds(params: dict) -> tuple[int, int]:
    lo = params.get("moderate_min_weeks", params.get("min_weeks", 2))
    hi = params.get("moderate_max_weeks", params.get("max_weeks", 4))
    return lo, hi


# Table 9 week-1 risk thresholds (same bounds as L1-RTT-0002a/b/c/d).
_TABLE9_WEEK1_METRIC_FIELDS = (
    "sessions_per_week",
    "sets_per_exercise",
    "intensity_percent_1RM",
    "rest_minutes",
)


def evaluate_against_table9_week1(plan: dict) -> list[str]:
    """Return metric field names that exceed Table 9 week-1 thresholds.

    Mirrors L1-RTT-0002a/b/c/d:
      - sets_per_exercise > 2
      - intensity_percent_1RM >= 75 (also via load_percent_1RM alias)
      - rest_minutes < 5
      - frequency/sessions > 2
    Only evaluates fields that are present (null → not a violation here).
    """
    violations: list[str] = []

    sets = _resolve_plan_value(plan, "sets_per_exercise")
    if sets is not None and sets > 2:
        violations.append("sets_per_exercise")

    intensity = _resolve_plan_value(plan, "intensity_percent_1RM")
    if intensity is not None and intensity >= 75:
        violations.append("intensity_percent_1RM")

    rest = _resolve_plan_value(plan, "rest_minutes")
    if rest is not None and rest < 5:
        violations.append("rest_minutes")

    # frequency_days_per_week aliases sessions_per_week when needed.
    freq = _resolve_plan_value(plan, "frequency_days_per_week")
    if freq is None:
        freq = plan.get("sessions_per_week")
    if freq is not None and freq > 2:
        violations.append("sessions_per_week")

    return violations


def _all_table9_absolute_fields_null(plan: dict) -> bool:
    """True when no Table 9 week-1 absolute metric is present on the plan.

    Uses the same field set / aliases as ``evaluate_long_inactivity_track_compliance``.
    Relative language alone must not populate these fields.
    """
    for field in _TABLE9_WEEK1_METRIC_FIELDS:
        if field == "sessions_per_week":
            if plan.get("sessions_per_week") is not None:
                return False
            if _resolve_plan_value(plan, "frequency_days_per_week") is not None:
                return False
        elif _resolve_plan_value(plan, field) is not None:
            return False
    return True


def evaluate_long_inactivity_track_compliance(plan: dict) -> str:
    """Derive long-inactivity track compliance from numeric fields.

    Returns one of: ``followed``, ``violated``, ``insufficient_data``,
    ``not_applicable``.

    Instead of trusting an LLM-extracted boolean, this compares the plan's
    extracted numeric fields (first-phase sessions_per_week, sets_per_exercise,
    intensity_percent_1RM, rest_minutes) against the Table 9 week-1 thresholds
    already encoded in rules L1-RTT-0002a/b/c/d.
    """
    weeks = plan.get("inactivity_duration_weeks")
    if weeks is None:
        return "not_applicable"
    if weeks < 4:
        # Falls under the moderate track instead (L1-RTT-0008).
        return "not_applicable"

    available_fields: list[str] = []
    for field in _TABLE9_WEEK1_METRIC_FIELDS:
        if field == "sessions_per_week":
            present = (
                plan.get("sessions_per_week") is not None
                or _resolve_plan_value(plan, "frequency_days_per_week") is not None
            )
        else:
            present = _resolve_plan_value(plan, field) is not None
        if present:
            available_fields.append(field)

    if not available_fields:
        return "insufficient_data"

    violations = evaluate_against_table9_week1(plan)
    return "violated" if violations else "followed"


def _evaluate_long_inactivity_track_compliance(
    condition: dict, plan: dict
) -> tuple[bool, dict]:
    """Evaluate L1-RTT-0001 via computed Table 9 week-1 compliance."""
    status = evaluate_long_inactivity_track_compliance(plan)
    weeks = plan.get("inactivity_duration_weeks")
    violations = (
        evaluate_against_table9_week1(plan) if status == "violated" else []
    )
    matched_parameters: dict[str, Any] = {
        "observed_value": weeks,
        "track_compliance": status,
        "table9_week1_violations": violations,
        "long_inactivity_threshold_weeks": (condition.get("parameters") or {}).get(
            "long_inactivity_threshold_weeks", 4
        ),
    }

    if status == "violated":
        matched_parameters["explanation_side"] = "flagged"
        matched_parameters["action_override"] = "reject"
        return True, matched_parameters
    if status == "insufficient_data":
        matched_parameters["explanation_side"] = "insufficient_data"
        matched_parameters["action_override"] = "flag_caution"
        return True, matched_parameters
    # followed / not_applicable → no match (followed still enters applicable)
    return False, matched_parameters


def _evaluate_relative_load_reduction_signal(
    condition: dict, plan: dict
) -> tuple[bool, dict]:
    """Match when relative load caution is present and absolute Table 9 metrics are not.

    This is a qualitative signal only — it never converts relative language into
    absolute %1RM / set counts. A match yields ``flag_caution`` (not a numeric
    compliance verdict).
    """
    uses = plan.get("uses_relative_load_reduction")
    quote = plan.get("relative_reduction_evidence_quote")
    if not isinstance(quote, str) or not quote.strip():
        quote = ""
    matched_parameters: dict[str, Any] = {
        "uses_relative_load_reduction": uses,
        "relative_reduction_evidence_quote": quote,
    }
    if uses is not True:
        return False, matched_parameters
    if not _all_table9_absolute_fields_null(plan):
        return False, matched_parameters
    return True, matched_parameters


def _evaluate_accommodation_check(
    condition: dict, plan: dict
) -> tuple[bool, dict]:
    """Evaluate disease/limitation context vs an explicit plan accommodation.

    Semantics (see NSCA Table 3 rules L1-RT-NSCA-0006–0011):
      1. No condition_field is True → not a match (caller should skip via
         evaluability when no True is present).
      2. Any condition_field True and accommodation is null → match with
         ``explanation_side=insufficient_data`` (flag_caution).
      3. Any condition_field True and accommodation absent (False, or enum
         equal to ``excluded_value``) → match flagged.
      4. Any condition_field True and accommodation present → no match
         (eligible for a pass fact).
    """
    params = condition.get("parameters") or {}
    condition_fields = list(params.get("condition_fields") or [])
    accommodation_field = params.get("accommodation_field")
    accommodation_type = params.get("accommodation_type", "boolean")
    excluded_value = params.get("excluded_value")

    condition_values = {name: plan.get(name) for name in condition_fields}
    any_true = any(value is True for value in condition_values.values())
    accommodation_value = (
        plan.get(accommodation_field) if accommodation_field else None
    )
    matched_parameters: dict[str, Any] = {
        **params,
        "condition_values": condition_values,
        "accommodation_value": accommodation_value,
    }

    if not any_true:
        return False, matched_parameters

    if accommodation_value is None:
        matched_parameters["explanation_side"] = "insufficient_data"
        matched_parameters["action_override"] = "flag_caution"
        return True, matched_parameters

    if accommodation_type == "boolean":
        if accommodation_value is False:
            matched_parameters["explanation_side"] = "flagged"
            return True, matched_parameters
        return False, matched_parameters

    if accommodation_type == "enum_exclude":
        if accommodation_value == excluded_value:
            matched_parameters["explanation_side"] = "flagged"
            return True, matched_parameters
        return False, matched_parameters

    raise ValueError(
        f"Unsupported accommodation_type: {accommodation_type!r}"
    )


def _evaluate_context_gate(condition: dict, plan: dict) -> tuple[bool, dict]:
    """Moderate return track gate (L1-RTT-0008).

    Moderate: min_weeks <= weeks < max_weeks.
    When plan_follows_moderate_return_track is absent, treat as not followed
    (conservative). Long-inactivity gating is handled by
    ``long_inactivity_track_compliance``, not this path.
    """
    params = condition.get("parameters") or {}
    weeks = plan.get("inactivity_duration_weeks")
    if weeks is None:
        return False, {}

    if _is_moderate_context_gate(condition):
        lo, hi = _moderate_week_bounds(params)
        follows = plan.get("plan_follows_moderate_return_track")
        if follows is None:
            follows = False
        hit = lo <= weeks < hi and follows is False
        return hit, {
            "observed_value": weeks,
            "min_weeks": lo,
            "max_weeks": hi,
            "moderate_min_weeks": lo,
            "moderate_max_weeks": hi,
            "plan_follows_moderate_return_track": follows,
        }

    # Legacy long-inactivity context_gate (should not remain on L1-RTT-0001).
    threshold = params.get("long_inactivity_threshold_weeks", 4)
    hit = weeks >= threshold
    return hit, {
        "observed_value": weeks,
        "long_inactivity_threshold_weeks": threshold,
    }


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
    if cond_type == "long_inactivity_track_compliance":
        return _evaluate_long_inactivity_track_compliance(condition, plan)
    if cond_type == "relative_load_reduction_signal":
        return _evaluate_relative_load_reduction_signal(condition, plan)
    if cond_type == "accommodation_check":
        return _evaluate_accommodation_check(condition, plan)
    raise ValueError(f"Unknown condition type: {cond_type!r}")


def _match_result(rule: dict, matched_parameters: dict) -> dict:
    action = matched_parameters.get("action_override") or rule["action"]
    return {
        "rule_id": rule["rule_id"],
        "action": action,
        "severity": rule["severity"],
        "matched_parameters": matched_parameters,
        "reason_template": rule.get("reason_template", {}),
        "explanation_side": matched_parameters.get("explanation_side", "flagged"),
    }


def _applicable_result(
    rule: dict,
    *,
    violated: bool,
    pass_parameters: dict | None = None,
    skip_pass_fact: bool = False,
) -> dict:
    return {
        "rule_id": rule["rule_id"],
        "action": rule["action"],
        "severity": rule["severity"],
        "violated": violated,
        "pass_parameters": pass_parameters or {},
        "skip_pass_fact": skip_pass_fact,
        "reason_template": rule.get("reason_template", {}),
        "condition": rule.get("condition") or {},
    }


def _is_population_gate(rule: dict) -> bool:
    return rule.get("rule_type") == "applicability_gate" or (
        (rule.get("condition") or {}).get("type") == "population_check"
    )


def _is_context_gate(rule: dict) -> bool:
    cond_type = (rule.get("condition") or {}).get("type")
    return cond_type in {"context_gate", "long_inactivity_track_compliance"}


_WEEK_EQ_RE = re.compile(
    r"weeks_since_return\s*==\s*(?P<week>-?\d+(?:\.\d+)?|\w+)", re.IGNORECASE
)


def _required_week(condition: dict, params: dict) -> int | None:
    """Return N if check includes ``weeks_since_return == N``, else None."""
    check = condition.get("check") or ""
    match = _WEEK_EQ_RE.search(check)
    if not match:
        return None
    token = match.group("week")
    try:
        return int(_resolve_rhs(token, params))
    except ValueError:
        return None


def _context_gate_in_scope(condition: dict, plan: dict) -> bool:
    """True when the plan's inactivity duration falls in this gate's week window."""
    params = condition.get("parameters") or {}
    weeks = plan.get("inactivity_duration_weeks")
    if weeks is None:
        return False
    if condition.get("type") == "long_inactivity_track_compliance":
        threshold = params.get("long_inactivity_threshold_weeks", 4)
        return weeks >= threshold
    if _is_moderate_context_gate(condition):
        lo, hi = _moderate_week_bounds(params)
        return lo <= weeks < hi
    threshold = params.get("long_inactivity_threshold_weeks", 4)
    return weeks >= threshold


def _primary_metric_field(condition: dict) -> str | None:
    """Field used for {observed_value} in pass templates (non-week clause)."""
    check = condition.get("check") or ""
    cond_type = condition.get("type")
    if cond_type in {"numeric_threshold", "boolean_check"}:
        fields = _referenced_fields_in_check(check)
        for field in reversed(fields):
            if field not in {"weeks_since_return", "population_is_long_inactivity"}:
                return field
        return fields[-1] if fields else None
    if cond_type == "range_check":
        m = _RANGE_CHECK_RE.match(check.strip())
        return m.group("field") if m else None
    return None


# Boolean population-exclusion flags that may appear in applicability.excludes.
# (Canonical set is defined near ACTIVE_RULE_IDS; kept here only as a comment anchor.)


def _condition_is_evaluable(condition: dict, plan: dict, rule: dict) -> bool:
    """Three-valued gate: True only when every referenced field is non-null.

    Unevaluable rules must not enter ``matched`` or ``checked_facts``. A null
    field means "unknown", not "confirmed absent / confirmed compliant".

    Compound AND/OR checks (e.g. L1-RTT-0003/0005) extract every LHS field and
    require all of them to be present — same rule as single-field conditions.
    """
    cond_type = condition.get("type")
    check = condition.get("check") or ""
    params = _enrich_week_dependent_params(condition.get("parameters") or {}, plan)

    if cond_type == "population_check":
        excludes = set((rule.get("applicability") or {}).get("excludes") or [])
        # Affirmative exclusion is enough to evaluate (and reject).
        if _positive_exclusion_flags(plan, excludes):
            return True
        # In-scope pass requires every boolean exclude flag to be explicitly False.
        for flag in excludes:
            if flag in _EXCLUSION_BOOLEAN_FLAGS and plan.get(flag) is not False:
                return False
        return True

    if cond_type in {"boolean_check", "numeric_threshold"}:
        fields = _referenced_fields_in_check(check)
        if not fields:
            return False
        for field in fields:
            if _resolve_plan_value(plan, field) is None:
                return False
        # Dry-run: week-scoped RHS aliases (e.g. min_denominator_for_week) must
        # resolve, and every atomic clause must be evaluable.
        try:
            hit, _ = _evaluate_compound_expression(check, plan, params)
        except ValueError:
            return False
        return hit is not None

    if cond_type == "range_check":
        match = _RANGE_CHECK_RE.match(check.strip())
        if not match:
            return False
        return _resolve_plan_value(plan, match.group("field")) is not None

    if cond_type == "context_gate":
        if plan.get("inactivity_duration_weeks") is None:
            return False
        if _is_moderate_context_gate(condition):
            return plan.get("plan_follows_moderate_return_track") is not None
        return True

    if cond_type == "long_inactivity_track_compliance":
        # not_applicable / insufficient_data → cannot confirm follow or violate.
        status = evaluate_long_inactivity_track_compliance(plan)
        return status in {"followed", "violated"}

    if cond_type == "relative_load_reduction_signal":
        # Need an explicit boolean; null means unknown → do not match or pass-fact.
        return plan.get("uses_relative_load_reduction") is not None

    if cond_type == "accommodation_check":
        # In scope only when at least one condition_field is explicitly True.
        # All-false / all-null → skip (not a Table 3 case).
        params = condition.get("parameters") or {}
        fields = list(params.get("condition_fields") or [])
        return any(plan.get(name) is True for name in fields)

    return False


def _build_pass_parameters(rule: dict, plan: dict) -> dict | None:
    """Collect values needed to render reason_template['pass'].

    Returns None when a required observed metric is missing — Layer1-B must not
    invent a pass claim for unevaluated fields.
    """
    condition = rule.get("condition") or {}
    if not _condition_is_evaluable(condition, plan, rule):
        return None

    params = dict(condition.get("parameters") or {})
    out = {**params}
    cond_type = condition.get("type")

    if cond_type == "context_gate":
        weeks = plan.get("inactivity_duration_weeks")
        out["observed_value"] = weeks
        if _is_moderate_context_gate(condition):
            out["plan_follows_moderate_return_track"] = plan.get(
                "plan_follows_moderate_return_track"
            )
        return out

    if cond_type == "long_inactivity_track_compliance":
        weeks = plan.get("inactivity_duration_weeks")
        status = evaluate_long_inactivity_track_compliance(plan)
        if status != "followed":
            return None
        out["observed_value"] = weeks
        out["track_compliance"] = status
        return out

    if cond_type == "relative_load_reduction_signal":
        # Qualitative caution only — never a confirmed numeric pass fact.
        return None

    if cond_type == "accommodation_check":
        params = condition.get("parameters") or {}
        fields = list(params.get("condition_fields") or [])
        if not any(plan.get(name) is True for name in fields):
            return None
        accommodation_field = params.get("accommodation_field")
        accommodation_value = (
            plan.get(accommodation_field) if accommodation_field else None
        )
        accommodation_type = params.get("accommodation_type", "boolean")
        if accommodation_value is None:
            return None
        if accommodation_type == "boolean" and accommodation_value is not True:
            return None
        if (
            accommodation_type == "enum_exclude"
            and accommodation_value == params.get("excluded_value")
        ):
            return None
        out["accommodation_value"] = accommodation_value
        return out

    if cond_type == "population_check":
        out["target_population"] = plan.get("target_population")
        return out

    field = _primary_metric_field(condition)
    if field:
        observed = _resolve_plan_value(plan, field)
        if observed is None:
            return None
        out["field"] = field
        out["observed_value"] = observed
        return out

    return out


def _population_gate_in_scope(rule: dict, plan: dict) -> bool:
    """Whether this applicability gate belongs to the plan's primary population."""
    population = set((rule.get("applicability") or {}).get("population") or [])
    primary = _PRIMARY_POPULATIONS & population
    if not primary:
        return True
    return effective_target_population(plan) in primary


def _rule_in_evaluation_scope(rule: dict, plan: dict) -> bool:
    """Whether this active rule's applicability / week gate covers the plan."""
    condition = rule.get("condition") or {}
    cond_type = condition.get("type")

    if _is_population_gate(rule):
        return _population_gate_in_scope(rule, plan)

    if cond_type == "context_gate":
        return _context_gate_in_scope(condition, plan)

    if not _rule_applies(rule, plan):
        return False

    required = _required_week(condition, condition.get("parameters") or {})
    if required is not None:
        week_n = plan.get("weeks_since_return")
        if week_n is None:
            return False
        try:
            return int(week_n) == int(required)
        except (TypeError, ValueError):
            return False
    return True


def _prepare_plan_for_layer1(plan: dict) -> dict:
    """Copy plan, apply Zone-B age flags, and normalize primary population."""
    plan = dict(plan)
    # Deterministic age→minor before gates: age_years < 18 or
    # stated_age_category == "minor" forces minor=True.
    plan = apply_deterministic_age_derived_flags(plan)
    raw_target = plan.get("target_population")
    category = plan.get("stated_age_category")
    # Normalize exclusive primary population from age when known.
    # Do not overwrite exclusion labels such as ``pregnant`` / ``minor`` that
    # older fixtures store in ``target_population`` — unless age_years or
    # stated_age_category drives exclusivity (rewrite primary class only).
    if (
        plan.get("age_years") is not None
        or category in ("minor", "older_adult", "adult")
        or raw_target is None
        or (isinstance(raw_target, str) and raw_target in _PRIMARY_POPULATIONS)
    ):
        plan["target_population"] = effective_target_population(plan)
    return plan


def _resolve_layer1_ruleset(
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict[str, Any]:
    if ruleset is not None:
        return ruleset
    if rules_path is not None:
        return load_ruleset(rules_path)
    return load_merged_rulesets()


def evaluate_primary_population_gates(
    plan: dict,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict[str, Any]:
    """Evaluate only in-scope primary population gates (L1-RT-0001 / NSCA-0001).

    Used by the two-stage extraction pipeline to decide whether stage 2 can be
    skipped. Does not run context gates or metric rules.

    Returns::

        {
          "rejected": bool,  # True when an in-scope gate matched (out-of-scope)
          "matched": [violation match dicts...],
          "applicable": [gate applicable-result dicts...],
          "effective_population": str,
          "plan": <normalized plan dict used for evaluation>,
        }
    """
    data = _resolve_layer1_ruleset(rules_path=rules_path, ruleset=ruleset)
    plan = _prepare_plan_for_layer1(plan)
    rules: list[dict] = [
        r for r in (data.get("rules") or []) if r.get("rule_id") in ACTIVE_RULE_IDS
    ]
    population_gates = [r for r in rules if _is_population_gate(r)]

    applicable: list[dict] = []
    for rule in population_gates:
        if not _population_gate_in_scope(rule, plan):
            continue
        condition = rule.get("condition") or {}
        if not _condition_is_evaluable(condition, plan, rule):
            continue
        hit, matched_parameters = evaluate_condition(condition, plan, rule)
        pass_parameters = _build_pass_parameters(rule, plan)
        applicable.append(
            _applicable_result(
                rule,
                violated=hit,
                pass_parameters=pass_parameters if pass_parameters is not None else {},
                skip_pass_fact=(not hit and pass_parameters is None),
            )
        )
        if hit:
            return {
                "rejected": True,
                "matched": [_match_result(rule, matched_parameters)],
                "applicable": applicable,
                "effective_population": effective_target_population(plan),
                "plan": plan,
            }

    return {
        "rejected": False,
        "matched": [],
        "applicable": applicable,
        "effective_population": effective_target_population(plan),
        "plan": plan,
    }


def evaluate_layer1_detailed(
    plan: dict,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict[str, list[dict]]:
    """Evaluate Layer1 and return both violations and in-scope rules.

    Returns:
      {
        "matched": [violation match dicts...],
        "applicable": [in-scope rule dicts with violated/pass_parameters...],
      }
    """
    data = _resolve_layer1_ruleset(rules_path=rules_path, ruleset=ruleset)
    gate = evaluate_primary_population_gates(
        plan, rules_path=rules_path, ruleset=data
    )
    plan = gate["plan"]
    applicable: list[dict] = list(gate["applicable"])

    if gate["rejected"]:
        # Out-of-scope population: short-circuit remaining Layer1 rules.
        return {
            "matched": list(gate["matched"]),
            "applicable": applicable,
        }

    rules: list[dict] = [
        r for r in (data.get("rules") or []) if r.get("rule_id") in ACTIVE_RULE_IDS
    ]
    context_gates = [
        r for r in rules if _is_context_gate(r) and not _is_population_gate(r)
    ]
    other_rules = [
        r for r in rules if not _is_population_gate(r) and not _is_context_gate(r)
    ]

    matched: list[dict] = []

    # Context gates first (early), but continue so week-specific RTT rules can also match.
    # After Layer2 exists, revisit whether context_gate should reject immediately or route.
    for rule in context_gates:
        if not _rule_in_evaluation_scope(rule, plan):
            continue
        condition = rule.get("condition") or {}
        if not _condition_is_evaluable(condition, plan, rule):
            continue
        hit, matched_parameters = evaluate_condition(condition, plan, rule)
        pass_parameters = _build_pass_parameters(rule, plan)
        applicable.append(
            _applicable_result(
                rule,
                violated=hit,
                pass_parameters=pass_parameters if pass_parameters is not None else {},
                skip_pass_fact=(not hit and pass_parameters is None),
            )
        )
        if hit:
            matched.append(_match_result(rule, matched_parameters))

    for rule in other_rules:
        if not _rule_in_evaluation_scope(rule, plan):
            continue
        condition = rule.get("condition") or {}
        if not _condition_is_evaluable(condition, plan, rule):
            continue
        hit, matched_parameters = evaluate_condition(condition, plan, rule)
        pass_parameters = _build_pass_parameters(rule, plan)
        applicable.append(
            _applicable_result(
                rule,
                violated=hit,
                pass_parameters=pass_parameters if pass_parameters is not None else {},
                skip_pass_fact=(not hit and pass_parameters is None),
            )
        )
        if hit:
            matched.append(_match_result(rule, matched_parameters))

    return {"matched": matched, "applicable": applicable}


def evaluate_layer1(
    plan: dict,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> list[dict]:
    """Evaluate Layer1 rules against a structured training plan.

    Returns only violated/matched rules (backward-compatible). For the full
    applicable-rule list used by Layer1-B, call ``evaluate_layer1_detailed``.
    """
    return evaluate_layer1_detailed(plan, rules_path=rules_path, ruleset=ruleset)[
        "matched"
    ]
