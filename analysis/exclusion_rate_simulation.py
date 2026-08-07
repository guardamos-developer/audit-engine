#!/usr/bin/env python3
"""Disposable analysis: synthetic exclusion-rate simulation for Live-mode planning.

NOT part of the pytest suite. Generates N personas from an external-stat-inspired
distribution, LLM-writes natural user prompts, runs plan_extractor → run_audit,
and writes a markdown + JSON report under analysis/.

Does not modify production rules or ACTIVE_RULE_IDS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.audit import run_audit  # noqa: E402
from src.plan_extractor import extract_plan  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_N = 200
DEFAULT_SEED = 20260807
PROMPT_BATCH_SIZE = 10
EXTRACTION_MODEL = os.environ.get("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini")
PROMPT_MODEL = os.environ.get("OPENAI_ANALYSIS_MODEL", EXTRACTION_MODEL)

# gpt-4o-mini list prices (USD / 1M tokens) — used for estimate + rough actual.
PRICE_IN = 0.15
PRICE_OUT = 0.60

# Fixed, evidence-aligned AI plan so Layer1 numeric cautions stay quiet when
# the case is in-scope. Exclusion measurement should not be polluted by load/sets.
ALIGNED_AI_RESPONSE = """\
Here's a simple 3-day full-body resistance program you can rotate through the week:

Session structure (each day):
- Squats or leg press: 3 sets of 8–10 reps at about 75% of 1RM
- Bench press or machine chest press: 3 sets of 8–10 at ~75% 1RM
- Seated row: 3 sets of 8–10
- Rest about 2 minutes between sets
- Stop with 2–3 reps in reserve — do not train every set to failure
- No complex periodization; add a little load when the last set feels easy

Warm up 5–10 minutes, and leave at least one rest day between hard sessions.
"""

AGE_BUCKET_WEIGHTS = {
    "18-64": 0.76,
    "65+_healthy": 0.10,
    "65+_frail": 0.02,
    "minor_under_18": 0.05,
}
# Remaining 7% mass is renormalized implicitly via random.choices weights above
# summing to 0.93 — renormalize to 1.0 for a proper categorical draw.
_AGE_KEYS = list(AGE_BUCKET_WEIGHTS.keys())
_AGE_W = [AGE_BUCKET_WEIGHTS[k] for k in _AGE_KEYS]
_AGE_W_SUM = sum(_AGE_W)
_AGE_W = [w / _AGE_W_SUM for w in _AGE_W]

LABEL_FIELDS = (
    "age_years",
    "injury_present",
    "pain_present",
    "pregnant",
    "minor",
    "frailty_present",
    "post_surgical",
)


def _openai_client():
    from openai import OpenAI

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _sample_injury_or_pain(rng: random.Random) -> tuple[bool, bool]:
    """27% injury_or_pain; among positives, split injury / pain / both."""
    if rng.random() >= 0.27:
        return False, False
    roll = rng.random()
    if roll < 0.55:
        return True, False
    if roll < 0.80:
        return True, True
    return False, True


def sample_persona(rng: random.Random) -> dict[str, Any]:
    """Draw one persona; resolve contradictory combinations."""
    bucket = rng.choices(_AGE_KEYS, weights=_AGE_W, k=1)[0]

    if bucket == "minor_under_18":
        age = rng.randint(12, 17)
        minor = True
        frailty = False
        pregnant = False
        injury = False
        pain = False
    elif bucket == "65+_frail":
        age = rng.randint(65, 92)
        minor = False
        frailty = True
        pregnant = False
        injury, pain = _sample_injury_or_pain(rng)
    elif bucket == "65+_healthy":
        age = rng.randint(65, 84)
        minor = False
        frailty = False
        pregnant = False
        injury, pain = _sample_injury_or_pain(rng)
    else:  # 18-64
        age = rng.randint(18, 64)
        minor = False
        frailty = False
        # Pregnancy only for assumed-female ages 18–49 at 2%.
        if 18 <= age <= 49 and rng.random() < 0.02:
            pregnant = True
        else:
            pregnant = False
        injury, pain = _sample_injury_or_pain(rng)

    sex = "female" if pregnant or (18 <= age <= 49 and rng.random() < 0.5) else (
        "male" if age >= 18 else "unspecified"
    )
    if pregnant:
        sex = "female"

    return {
        "age_bucket": bucket,
        "age_years": age,
        "sex": sex,
        "minor": minor,
        "frailty_present": frailty,
        "pregnant": pregnant,
        "injury_present": injury,
        "pain_present": pain,
        "post_surgical": False,
        "intended_labels": {
            "age_years": age,
            "minor": minor,
            "frailty_present": frailty,
            "pregnant": pregnant,
            "injury_present": injury,
            "pain_present": pain,
            "post_surgical": False,
        },
    }


def _persona_brief(p: dict[str, Any]) -> str:
    bits = [f"age={p['age_years']}", f"bucket={p['age_bucket']}", f"sex={p['sex']}"]
    for k in (
        "minor",
        "frailty_present",
        "pregnant",
        "injury_present",
        "pain_present",
    ):
        if p.get(k):
            bits.append(f"{k}=true")
    return ", ".join(bits)


def generate_prompts_batch(
    client: Any,
    personas: list[dict[str, Any]],
    *,
    model: str,
) -> tuple[list[str], dict[str, int]]:
    """Ask the LLM for one natural fitness-app user message per persona."""
    indexed = [
        {"index": i, "attributes": _persona_brief(p)} for i, p in enumerate(personas)
    ]
    system = (
        "You write short, natural fitness-app user messages (Japanese or English "
        "mixed ok; prefer natural Japanese casual style OR natural English). "
        "Each message must clearly reflect the given attributes without sounding "
        "like a form. Do not invent attributes not listed. Return ONLY JSON: "
        '{"prompts":[{"index":0,"text":"..."}, ...]} with one entry per input.'
    )
    user = (
        "Generate one user request per item. The user asks an AI for a training "
        "plan. Reflect injury/pain/pregnancy/age/frailty/minor when true.\n\n"
        + json.dumps(indexed, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
    }
    raw = json.loads(resp.choices[0].message.content or "{}")
    by_idx: dict[int, str] = {}
    for item in raw.get("prompts") or []:
        try:
            by_idx[int(item["index"])] = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
    out: list[str] = []
    for i, p in enumerate(personas):
        text = by_idx.get(i)
        if not text:
            # Deterministic fallback if the batch omitted an index.
            text = (
                f"I'm {p['age_years']}. "
                + (
                    "I'm under 18 and need a beginner gym plan. "
                    if p["minor"]
                    else ""
                )
                + ("I'm pregnant — is strength training ok? " if p["pregnant"] else "")
                + (
                    "I have frailty / low physical reserve. "
                    if p["frailty_present"]
                    else ""
                )
                + ("Old knee injury. " if p["injury_present"] else "")
                + ("I have joint pain. " if p["pain_present"] else "")
                + "Please make me an upper-body strength plan."
            )
        out.append(text)
    return out, usage


def classify_exclusion(audit: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Map Layer1 reject matches to a primary exclusion cause."""
    matches = []
    # run_audit returns matched_rules as ids; need actions from explanations path
    # Re-derive from evaluate is heavy; use matched_rules + plan flags.
    matched_ids = list(audit.get("matched_rules") or [])
    verdict = audit.get("verdict")
    is_excluded = verdict == "rejected" and any(
        rid in {"L1-RT-0001", "L1-RT-NSCA-0001"} for rid in matched_ids
    )
    # Also treat any reject action as exclusion (route_to_layer2_or_reject / reject)
    # Even if gate id missing from list somehow.
    if verdict == "rejected" and matched_ids:
        # Prefer population gates when present
        gate_hit = [r for r in matched_ids if r in {"L1-RT-0001", "L1-RT-NSCA-0001"}]
        is_excluded = bool(gate_hit) or is_excluded

    causes: list[str] = []
    if plan.get("injury_present") is True:
        causes.append("injury_present")
    if plan.get("pain_present") is True:
        causes.append("pain_present")
    if plan.get("pregnant") is True:
        causes.append("pregnant")
    if plan.get("minor") is True:
        causes.append("minor")
    if plan.get("frailty_present") is True:
        causes.append("frailty_present")
    if plan.get("post_surgical") is True:
        causes.append("post_surgical")

    # Priority for "primary" when multiple flags present.
    priority = [
        "minor",
        "pregnant",
        "frailty_present",
        "injury_present",
        "pain_present",
        "post_surgical",
    ]
    primary = next((c for c in priority if c in causes), None)

    # Age routing itself does not reject; note when older-adult gate fired via frailty.
    age_routing_note = None
    if "L1-RT-NSCA-0001" in matched_ids:
        age_routing_note = "older_adult_healthy_gate"
    elif "L1-RT-0001" in matched_ids:
        age_routing_note = "healthy_adult_18plus_gate"

    return {
        "is_population_exclusion": bool(
            verdict == "rejected"
            and any(r in matched_ids for r in ("L1-RT-0001", "L1-RT-NSCA-0001"))
        ),
        "verdict": verdict,
        "matched_rules": matched_ids,
        "positive_flags_in_extracted_plan": causes,
        "primary_cause": primary if verdict == "rejected" else None,
        "gate": age_routing_note,
    }


def label_agreement(
    intended: dict[str, Any], extracted_plan: dict[str, Any]
) -> dict[str, Any]:
    per_field: dict[str, Any] = {}
    exact = 0
    comparable = 0
    for field in LABEL_FIELDS:
        exp = intended.get(field)
        got = extracted_plan.get(field)  # may be absent → None
        comparable += 1
        match = got == exp
        if match:
            exact += 1
        per_field[field] = {
            "intended": exp,
            "extracted": got,
            "exact_match": match,
            "positive_recall_hit": (
                (got is True) if exp is True else None
            ),
            "false_positive": (got is True and exp is False),
        }
    return {
        "exact_match_rate": exact / comparable if comparable else 0.0,
        "fields": per_field,
    }


def estimate_cost_usd(n: int, batch_size: int = PROMPT_BATCH_SIZE) -> dict[str, float]:
    n_prompt_calls = (n + batch_size - 1) // batch_size
    # Rough token guesses
    prompt_gen_in = n_prompt_calls * 800
    prompt_gen_out = n_prompt_calls * (120 * batch_size)
    extract_in = n * 4500  # schema-heavy system prompt
    extract_out = n * 1200
    total_in = prompt_gen_in + extract_in
    total_out = prompt_gen_out + extract_out
    usd = (total_in / 1e6) * PRICE_IN + (total_out / 1e6) * PRICE_OUT
    return {
        "est_input_tokens": float(total_in),
        "est_output_tokens": float(total_out),
        "est_usd_mid": round(usd, 3),
        "est_usd_low": round(usd * 0.6, 3),
        "est_usd_high": round(usd * 1.8, 3),
        "prompt_calls": float(n_prompt_calls),
        "extraction_calls": float(n),
    }


def run_one(
    client: Any,
    persona: dict[str, Any],
    user_prompt: str,
) -> dict[str, Any]:
    extraction = extract_plan(user_prompt, ALIGNED_AI_RESPONSE, client=client)
    plan = dict(extraction.get("plan") or {})
    audit = run_audit(plan, lang="en", skip_layer3=True)
    excl = classify_exclusion(audit, plan)
    agree = label_agreement(persona["intended_labels"], plan)
    return {
        "persona": persona,
        "user_prompt": user_prompt,
        "extracted_plan": plan,
        "extraction_warnings": extraction.get("extraction_warnings") or [],
        "audit_verdict": audit.get("verdict"),
        "matched_rules": audit.get("matched_rules") or [],
        "exclusion": excl,
        "label_agreement": agree,
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    seed: int,
    model: str,
    usage_totals: dict[str, int],
    cost_est: dict[str, float],
) -> str:
    n = len(results)
    excluded = [r for r in results if r["exclusion"]["is_population_exclusion"]]
    rejected = [r for r in results if r["audit_verdict"] == "rejected"]
    flagged = [r for r in results if r["audit_verdict"] == "flagged"]
    passed = [r for r in results if r["audit_verdict"] == "pass"]
    insuff = [r for r in results if r["audit_verdict"] == "insufficient_data"]
    reviewed = [
        r for r in results if r["audit_verdict"] == "flagged_for_review"
    ]

    cause_counts = Counter(
        r["exclusion"]["primary_cause"] or "unknown"
        for r in excluded
    )
    gate_counts = Counter(r["exclusion"]["gate"] or "none" for r in excluded)

    # Non-excluded = not population-gate rejected
    not_pop_excl = [r for r in results if not r["exclusion"]["is_population_exclusion"]]

    # Label agreement aggregates
    field_exact = Counter()
    field_total = Counter()
    pos_hit = Counter()
    pos_total = Counter()
    fp = Counter()
    for r in results:
        fields = r["label_agreement"]["fields"]
        for name, info in fields.items():
            field_total[name] += 1
            if info["exact_match"]:
                field_exact[name] += 1
            if info["positive_recall_hit"] is not None:
                pos_total[name] += 1
                if info["positive_recall_hit"]:
                    pos_hit[name] += 1
            if info["false_positive"]:
                fp[name] += 1

    overall_exact = sum(field_exact.values()) / max(sum(field_total.values()), 1)

    actual_usd = (
        usage_totals["prompt_tokens"] / 1e6 * PRICE_IN
        + usage_totals["completion_tokens"] / 1e6 * PRICE_OUT
    )

    bucket_excl = Counter()
    bucket_n = Counter()
    for r in results:
        b = r["persona"]["age_bucket"]
        bucket_n[b] += 1
        if r["exclusion"]["is_population_exclusion"]:
            bucket_excl[b] += 1

    lines = [
        "# Exclusion-rate simulation report",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- N: **{n}**",
        f"- Seed: `{seed}`",
        f"- Models: prompt-gen=`{model}`, extraction=`{EXTRACTION_MODEL}`",
        f"- Pipeline: `plan_extractor.extract_plan` → `run_audit(..., skip_layer3=True)`",
        "",
        "## Caveats (read first)",
        "",
        "1. **Minor share 5% is an unverified provisional assumption** "
        "(no strong external base rate was available for fitness-app users). "
        "Treat overall exclusion % as sensitive to this input.",
        "2. Personas are **synthetic**; attribute draws are mostly independent "
        "with light consistency fixes (e.g. frailty ⇒ age≥65; minors skip injury).",
        "3. AI plan text is a **fixed in-range template**, so this measures "
        "population-gate exclusion — not plan-quality flag rates.",
        "4. Pregnancy is applied only for ages 18–49 at 2% (female-assumed).",
        "5. Age-bucket weights were renormalized from the stated 76/10/2/5 "
        f"(summed to {_AGE_W_SUM:.2f}) up to 1.0.",
        "",
        "## Cost",
        "",
        f"- Estimated before run: "
        f"${cost_est['est_usd_low']:.2f}–${cost_est['est_usd_high']:.2f} "
        f"(mid ${cost_est['est_usd_mid']:.2f})",
        f"- Observed tokens: in={usage_totals['prompt_tokens']}, "
        f"out={usage_totals['completion_tokens']}",
        f"- Observed approx USD (list price): **${actual_usd:.3f}** "
        "(prompt-generation usage only fully metered; extraction usage "
        "may be under-counted if the OpenAI client omits usage on some calls)",
        "",
        "## Headline results",
        "",
        f"- Population-gate exclusions "
        f"(`L1-RT-0001` / `L1-RT-NSCA-0001` → verdict rejected): "
        f"**{len(excluded)} / {n} = {100 * len(excluded) / n:.1f}%**",
        f"- Any `rejected` verdict: "
        f"**{len(rejected)} / {n} = {100 * len(rejected) / n:.1f}%**",
        f"- Not population-excluded (normal evaluation path): "
        f"**{len(not_pop_excl)} / {n} = {100 * len(not_pop_excl) / n:.1f}%**",
        "",
        "### Verdict mix (all cases)",
        "",
        f"| verdict | n | % |",
        f"|---|---:|---:|",
        f"| rejected | {len(rejected)} | {100 * len(rejected) / n:.1f} |",
        f"| flagged | {len(flagged)} | {100 * len(flagged) / n:.1f} |",
        f"| pass | {len(passed)} | {100 * len(passed) / n:.1f} |",
        f"| insufficient_data | {len(insuff)} | {100 * len(insuff) / n:.1f} |",
        f"| flagged_for_review | {len(reviewed)} | {100 * len(reviewed) / n:.1f} |",
        "",
        "## Exclusion primary-cause breakdown",
        "",
        "Primary cause = first hit in priority "
        "`minor > pregnant > frailty_present > injury_present > pain_present` "
        "among **extracted** plan flags when a population gate rejected.",
        "",
        "| primary_cause | n | % of exclusions | % of all |",
        "|---|---:|---:|---:|",
    ]
    for cause, c in cause_counts.most_common():
        lines.append(
            f"| {cause} | {c} | {100 * c / max(len(excluded), 1):.1f} | "
            f"{100 * c / n:.1f} |"
        )
    lines += [
        "",
        "### Gate that fired",
        "",
        "| gate | n |",
        "|---|---:|",
    ]
    for g, c in gate_counts.most_common():
        lines.append(f"| {g} | {c} |")

    lines += [
        "",
        "## Exclusion rate by intended age_bucket",
        "",
        "| age_bucket | n | excluded | rate |",
        "|---|---:|---:|---:|",
    ]
    for b in _AGE_KEYS:
        bn = bucket_n[b]
        be = bucket_excl[b]
        rate = 100 * be / bn if bn else 0.0
        lines.append(f"| {b} | {bn} | {be} | {rate:.1f}% |")

    lines += [
        "",
        "## Extractor vs intended-label agreement",
        "",
        f"- Overall exact-match rate across "
        f"{', '.join(LABEL_FIELDS)}: **{100 * overall_exact:.1f}%**",
        "",
        "| field | exact % | positive recall "
        "(intended true → extracted true) | false positives "
        "(intended false → extracted true) |",
        "|---|---:|---:|---:|",
    ]
    for field in LABEL_FIELDS:
        ex = 100 * field_exact[field] / max(field_total[field], 1)
        if pos_total[field]:
            rec = f"{100 * pos_hit[field] / pos_total[field]:.1f}% ({pos_hit[field]}/{pos_total[field]})"
        else:
            rec = "n/a"
        lines.append(
            f"| {field} | {ex:.1f}% | {rec} | {fp[field]} |"
        )

    lines += [
        "",
        "## Sampling design (implemented)",
        "",
        "```",
        "age_bucket (renormalized):",
        f"  18-64:        {100 * _AGE_W[0]:.2f}%",
        f"  65+_healthy:  {100 * _AGE_W[1]:.2f}%",
        f"  65+_frail:    {100 * _AGE_W[2]:.2f}%",
        f"  minor_<18:    {100 * _AGE_W[3]:.2f}%  [PROVISIONAL / UNVERIFIED]",
        "injury_or_pain-ish: injury ~27% (non-minors); pain correlated/secondary",
        "pregnancy: 2% among ages 18-49 only",
        "```",
        "",
        "## Interpretation notes for Live-mode",
        "",
        "- The population-gate exclusion rate above is the closest proxy to "
        "\"routed out of Layer1 automated scoring\".",
        "- `flagged` cases still received normal Layer1 evaluation (not excluded).",
        "- If minors were 0% instead of ~5%, overall exclusion would drop by "
        "roughly the minor contribution shown in the cause table.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Dry-run persona sampling + cost estimate only",
    )
    args = parser.parse_args()

    cost_est = estimate_cost_usd(args.n)
    print("=== COST ESTIMATE (before API) ===")
    print(f"model={PROMPT_MODEL} / extraction={EXTRACTION_MODEL}")
    print(
        f"calls: ~{int(cost_est['prompt_calls'])} prompt-batches + "
        f"{int(cost_est['extraction_calls'])} extractions"
    )
    print(
        f"estimated USD: ${cost_est['est_usd_low']:.2f} – "
        f"${cost_est['est_usd_high']:.2f} (mid ${cost_est['est_usd_mid']:.2f})"
    )
    if args.skip_api:
        return 0

    rng = random.Random(args.seed)
    personas = [sample_persona(rng) for _ in range(args.n)]
    client = _openai_client()

    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
    prompts: list[str] = []
    print(f"Generating {args.n} user prompts in batches of {PROMPT_BATCH_SIZE}...")
    for start in range(0, args.n, PROMPT_BATCH_SIZE):
        batch = personas[start : start + PROMPT_BATCH_SIZE]
        texts, usage = generate_prompts_batch(client, batch, model=PROMPT_MODEL)
        prompts.extend(texts)
        usage_totals["prompt_tokens"] += usage["prompt_tokens"]
        usage_totals["completion_tokens"] += usage["completion_tokens"]
        print(f"  prompts {start + len(batch)}/{args.n}")
        time.sleep(0.2)

    print(f"Running extract+audit with {args.workers} workers...")
    results: list[dict[str, Any] | None] = [None] * args.n

    def _job(i: int) -> tuple[int, dict[str, Any]]:
        # Each thread gets its own client (httpx is not always thread-safe shared).
        local = _openai_client()
        return i, run_one(local, personas[i], prompts[i])

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, i) for i in range(args.n)]
        for fut in as_completed(futs):
            i, row = fut.result()
            results[i] = row
            done += 1
            if done % 20 == 0 or done == args.n:
                print(f"  audited {done}/{args.n}")

    assert all(r is not None for r in results)
    typed_results: list[dict[str, Any]] = [r for r in results if r is not None]

    stamp = "2026-08-07"
    report_path = ANALYSIS_DIR / f"exclusion_rate_report_{stamp}.md"
    raw_path = ANALYSIS_DIR / f"exclusion_rate_raw_{stamp}.jsonl"
    meta_path = ANALYSIS_DIR / f"exclusion_rate_meta_{stamp}.json"

    report = build_report(
        typed_results,
        seed=args.seed,
        model=PROMPT_MODEL,
        usage_totals=usage_totals,
        cost_est=cost_est,
    )
    report_path.write_text(report, encoding="utf-8")
    with raw_path.open("w", encoding="utf-8") as f:
        for row in typed_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path.write_text(
        json.dumps(
            {
                "n": args.n,
                "seed": args.seed,
                "models": {
                    "prompt": PROMPT_MODEL,
                    "extraction": EXTRACTION_MODEL,
                },
                "usage_prompt_gen_only": usage_totals,
                "cost_estimate": cost_est,
                "age_weights_renormalized": dict(zip(_AGE_KEYS, _AGE_W)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {meta_path}")
    print("\n===== REPORT =====\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
