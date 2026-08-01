# Guardamos — Third-Party Audit Layer for AI-Generated Fitness Plans

**Guardamos checks AI-generated training plans against published sports-medicine guidelines before they reach the people who follow them.**

## The problem

General-purpose AI assistants are increasingly used to generate fitness and training plans. They're good at producing plausible-sounding programs, but they don't reliably account for context that matters clinically — for example, a plan for someone returning to training after months of inactivity should look very different from a plan for someone training consistently. Missing that context is a documented risk factor behind real injuries (exertional rhabdomyolysis in particular has well-known links to "too much, too soon" after a layoff).

## What Guardamos does

Guardamos sits between an AI-generated plan and the end user. It checks the plan against a growing set of deterministic rules extracted from published sports-medicine guidelines, and returns:

- a verdict (`pass` / `flagged` / `rejected`)
- which rule(s) were triggered and why, in plain language traceable back to the guideline it came from
- for plans that pass, a short summary of why the plan is appropriate given the stated context

Guardamos doesn't generate training plans, and it doesn't claim to guarantee safety. It's a guardrail and explainability layer, not a substitute for professional medical judgment. See [Rules & sources](#rules--sources) for exactly what's currently checked.

## Example

Prompted a general-purpose AI assistant with: *"I haven't worked out in six months and want to rebuild muscle as fast as possible. Give me a serious training plan."* The response went straight into a high-intensity 4-day/week program in week one — no adjustment for the fact that returning after an extended layoff carries documented injury risk.

```
$ python main.py sample_plans/chatgpt_6month_layoff.json --lang en --skip-layer3
{
  "verdict": "rejected",
  "matched_rules": ["L1-RTT-0001", "L1-RTT-0002a", "L1-RTT-0002d"],
  "explanations": [
    "The input indicates a return to training after a prolonged period of inactivity (26 weeks). CSCCa/NSCA guidelines require this population to follow the dedicated 'return from long inactivity' track...",
    "Week 1 of return-to-training after long inactivity recommends no more than 2 set(s) per exercise (Table 9, Week 1 row). The plan's value (4) exceeds this.",
    "Week 1 of return-to-training after long inactivity recommends no more than 2 session(s) per week (Table 9, Week 1 row). The plan's value (4) exceeds this."
  ]
}
```

A plan adjusted for the same context, in line with the relevant guideline's week-1 recommendations, passes — and still gets an explanation:

```
$ python main.py sample_plans/chatgpt_6month_layoff_corrected.json --lang en
{
  "verdict": "pass",
  "matched_rules": [],
  "layer3_response": "The established resistance-training plan consists of 2 sessions per week, with 2 sets per exercise, which is appropriate for a gradual return to training after a 26-week period of inactivity..."
}
```

## Architecture

Three layers:

1. **Layer 1 — deterministic rules.** Extracted from published guidelines, evaluated in code, no LLM involved. This is what's open-sourced here.
2. **Layer 2 — expert tacit knowledge.** Not yet implemented. Will evaluate whether a plan reflects the contextual judgment a human expert would apply, beyond literature-based thresholds.
3. **Layer 3 — free-form response.** Once a plan clears Layers 1–2, an LLM generates a short natural-language summary. Explanations for `flagged`/`rejected` verdicts are template-based, not LLM-generated — the goal is to avoid introducing new hallucinations into the audit layer itself.

## Rules & sources

| Ruleset | Source | Status |
|---|---|---|
| `layer1_rules_acsm_rt_v1.json` | Currier et al. American College of Sports Medicine Position Stand: Resistance Training Prescription for Muscle Function, Hypertrophy, and Physical Performance in Healthy Adults. *Med Sci Sports Exerc.* 2026;58(4):851-872. | `pending_source_check` |
| `layer1_rules_cscca_return_to_training_v1.json` | Caterisano et al. CSCCa and NSCA Joint Consensus Guidelines for Transition Periods: Safe Return to Training Following Inactivity. *Strength Cond J.* 2019;41(3):1-23. | `pending_source_check` |

`pending_source_check` means a rule was extracted with LLM assistance and hasn't yet been manually cross-checked against the source PDF line-by-line. Treat rule content as a draft until its `verification_status` field is updated to `verified`.

## Running it

```bash
git clone https://github.com/guardamos-developer/audit-engine.git
cd audit-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY

python main.py sample_plans/chatgpt_6month_layoff.json --lang en
python -m pytest tests/ -v
```

## Status

Early-stage, solo-maintained, build-in-public project. Not a medical device. Not intended to make clinical decisions or replace professional medical guidance. Currently scoped to general resistance-training programming for healthy adults — see each rule's `applicability` field for exact population scope and exclusions.

## Contact

- X: [@guardamos_dev](https://x.com/guardamos_dev)
- LinkedIn: [Guardamos](https://linkedin.com/company/guardamos)
- Website: [guardamos.dev](https://guardamos.dev)

## License

MIT — see [LICENSE](./LICENSE).
