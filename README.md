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

Beyond catching hallucinations, Guardamos is built around the kind of
explainability and audit-trail requirements that regulations like the
EU AI Act are pushing toward — every verdict traces back to a specific
guideline, not a black box. That lets the AI teams we work with focus
their own effort on what they do best (the generative, probabilistic
side of their product) instead of building this layer themselves.

Most users should start with the hosted API below. Prefer to run this
yourself instead? See [Self-hosting](#self-hosting-advanced).

## Using the hosted API

If you're calling the hosted API from your own backend (rather than
self-hosting), the typical integration point is right after your own AI
generates a plan and before you show it to your user:

```python
import os
import requests

GUARDAMOS_API_KEY = os.environ["GUARDAMOS_API_KEY"]

def audit_plan(user_prompt: str, ai_response: str) -> dict:
    response = requests.post(
        "https://guardamos-audit-engine.onrender.com/audit",
        headers={"X-API-Key": GUARDAMOS_API_KEY},
        json={
            "user_prompt": user_prompt,
            "ai_response": ai_response,
            # skip_layer3 defaults to true on the hosted API: Layer3 (LLM
            # narrative) is skipped. Deterministic checked_facts still return
            # on pass. Set false only if you also want layer3_response
            # (extra OpenAI call and latency).
            # "skip_layer3": False,
        },
        # Full audits often take ~8–13s; allow headroom for slower outliers.
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
```

**Request body notes:** `lang` is `en` | `pt` | `ja` (default `en`).
`skip_layer3` defaults to **`true`**. That does **not** drop
`checked_facts` on pass — those are deterministic Layer1-B facts. It only
skips the optional LLM `layer3_response`. Omit the field or leave it `true`
for lower latency; set `"skip_layer3": false` when you want the Layer3
narrative.

### Latency expectations (hosted API)

Measured on the production endpoint after warm-up (`skip_layer3=true`,
client wall time, 12 runs per case, Oregon host). Treat these as
**order-of-magnitude guidance**, not an SLA — OpenAI latency varies.

| Path | Typical (p50) | Notes |
|---|---|---|
| Early-exit (`rejected` at the population gate, e.g. injury / frailty) | **~3.5–5 s** | Stage-2 extraction is skipped |
| Full evaluation (`pass` / `flagged`, stage 2 runs) | **~8–13 s** | ACSM pass ≈ 12 s p50; flagged ≈ 9 s; older-adult full path ≈ 8 s |
| Slower outliers | **>15 s** (occasionally ~18–20 s) | p95 on ACSM pass was ~18 s in the same run |

**Integration recommendation:** do **not** block your end-user UI on a
synchronous spinner waiting for `/audit`. Prefer an asynchronous pattern —
enqueue the audit after your own model returns, continue your product
flow, and surface the verdict via a completion notification, webhook, or
background job. If you must call synchronously from a backend worker,
use a timeout of at least **~30 s** and treat timeouts / 5xx as
non-blocking for the user-facing path.

See [`examples/integration_example.py`](examples/integration_example.py)
for a complete, runnable version, including error handling for when the
audit service is slow or unreachable — this call is not meant to block
your own request path indefinitely. See also [Status](#status) below on
intended use.

## Example

Prompted a general-purpose AI assistant with: *"I haven't worked out in six months and want to rebuild muscle as fast as possible. Give me a serious training plan."* The response went straight into a high-intensity 4-day/week program in week one — no adjustment for the fact that returning after an extended layoff carries documented injury risk.

```
$ python main.py sample_plans/chatgpt_6month_layoff.json --lang en --skip-layer3
{
  "verdict": "rejected",
  "summary": "3 issue(s) flagged: L1-RTT-0001, L1-RTT-0002a, L1-RTT-0002d. See explanations for details.",
  "matched_rules": ["L1-RTT-0001", "L1-RTT-0002a", "L1-RTT-0002d"],
  "explanations": [
    "The input indicates a return to training after a prolonged period of inactivity (26 weeks). CSCCa/NSCA guidelines require this population to follow the dedicated 'return from long inactivity' track...",
    "Table 9 (p.16) recommends 1-2 sets per exercise in week 1 of return-to-training after long inactivity. This audit flags exceeding the top of that range (>2) as the risk-relevant threshold. The plan's value (4) exceeds it.",
    "Table 9 (p.16) recommends 1-2 sessions per week in week 1 of return-to-training after long inactivity. This audit flags exceeding the top of that range (>2) as the risk-relevant threshold. The plan's value (4) exceeds it."
  ]
}
```

Every response includes a top-level `summary`: one or two sentences built deterministically from counts and rule ids (no LLM). It does not replace `explanations`, `checked_facts`, or `layer3_response`.

A plan adjusted for the same context, in line with the relevant guideline's week-1 recommendations, passes. Rather than a silent pass, every applicable rule that was checked (not just the ones that failed) is returned as a `checked_facts` entry, and a short natural-language summary is generated from those facts alone — not from freely re-reading the plan:

```
$ python main.py sample_plans/chatgpt_6month_layoff_corrected.json --lang en
{
  "verdict": "pass",
  "summary": "2 checks passed, 0 flagged.",
  "matched_rules": [],
  "checked_facts": [
    { "rule_id": "L1-RTT-0001", "text": "For this long-inactivity context, the plan follows the dedicated 'return from long inactivity' track..." },
    { "rule_id": "L1-RTT-0002a", "text": "Set volume in week 1 (2 set(s)/exercise) falls within the range Table 9 (p.16) recommends for this stage (1-2 sets)." }
  ],
  "layer3_response": "For this long-inactivity context, the plan follows the dedicated return from long inactivity track... the training parameters in week 1 are consistent with the guidelines provided for individuals returning from long inactivity."
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
| `layer1_rules_cscca_return_to_training_v1.json` | Caterisano et al. CSCCa and NSCA Joint Consensus Guidelines for Transition Periods: Safe Return to Training Following Inactivity. *Strength Cond J.* 2019;41(3):1-23. | `verified` |

`pending_source_check` means a rule was extracted with LLM assistance and hasn't yet been manually cross-checked against the source PDF line-by-line. Treat rule content as a draft until its `verification_status` field is updated to `verified`.

## A note on input format

The `--raw-text` mode expects both the original user prompt and the AI's
response, not the response alone. Population-relevant context (injury,
pregnancy, age, recent surgery) usually appears in the prompt, not in the
generated plan itself. Submitting the response without the prompt risks
missing that context entirely — the current design defaults to *not*
excluding a population when this information is simply absent, which is
the safer failure mode, but it is not a substitute for providing the
context in the first place. Always send both.

## Self-hosting (advanced)

This section is for running the audit engine on your own infrastructure
with your own OpenAI API key — most users won't need this, see
[Using the hosted API](#using-the-hosted-api) instead.

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

Guardamos is designed as a development-time and pre-deployment
verification tool, not as a real-time safety-critical component embedded
in a live inference path. It is not intended to be called synchronously
to gate what an end user sees before a response is shown to them.

## Contact

- X: [@guardamos_dev](https://x.com/guardamos_dev)
- LinkedIn: [Guardamos](https://linkedin.com/company/guardamos)
- Website: [guardamos.dev](https://guardamos.dev)

## License

MIT — see [LICENSE](./LICENSE).

<!-- deploy-check: 2026-08-07 verifying Render auto-deploy after GitHub App reinstall -->
