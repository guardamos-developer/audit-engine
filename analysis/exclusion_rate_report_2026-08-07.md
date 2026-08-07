# Exclusion-rate simulation report

- Generated (UTC): `2026-08-07T13:23:44.916877+00:00`
- N: **200**
- Seed: `20260807`
- Models: prompt-gen=`gpt-4o-mini`, extraction=`gpt-4o-mini`
- Pipeline: `plan_extractor.extract_plan` → `run_audit(..., skip_layer3=True)`

## Caveats (read first)

1. **Minor share 5% is an unverified provisional assumption** (no strong external base rate was available for fitness-app users). Treat overall exclusion % as sensitive to this input.
2. Personas are **synthetic**; attribute draws are mostly independent with light consistency fixes (e.g. frailty ⇒ age≥65; minors skip injury).
3. AI plan text is a **fixed in-range template**, so this measures population-gate exclusion — not plan-quality flag rates.
4. Pregnancy is applied only for ages 18–49 at 2% (female-assumed).
5. Age-bucket weights were renormalized from the stated 76/10/2/5 (summed to 0.93) up to 1.0.

## Cost

- Estimated before run: $0.18–$0.53 (mid $0.30)
- Observed tokens: in=7602, out=6295
- Observed approx USD (list price): **$0.005** (prompt-generation usage only fully metered; extraction usage may be under-counted if the OpenAI client omits usage on some calls)

## Headline results

- Population-gate exclusions (`L1-RT-0001` / `L1-RT-NSCA-0001` → verdict rejected): **72 / 200 = 36.0%**
- Any `rejected` verdict: **72 / 200 = 36.0%**
- Not population-excluded (normal evaluation path): **128 / 200 = 64.0%**

### Verdict mix (all cases)

| verdict | n | % |
|---|---:|---:|
| rejected | 72 | 36.0 |
| flagged | 2 | 1.0 |
| pass | 126 | 63.0 |
| insufficient_data | 0 | 0.0 |
| flagged_for_review | 0 | 0.0 |

## Exclusion primary-cause breakdown

Primary cause = first hit in priority `minor > pregnant > frailty_present > injury_present > pain_present` among **extracted** plan flags when a population gate rejected.

| primary_cause | n | % of exclusions | % of all |
|---|---:|---:|---:|
| injury_present | 60 | 83.3 | 30.0 |
| minor | 8 | 11.1 | 4.0 |
| frailty_present | 4 | 5.6 | 2.0 |

### Gate that fired

| gate | n |
|---|---:|
| healthy_adult_18plus_gate | 58 |
| older_adult_healthy_gate | 14 |

## Exclusion rate by intended age_bucket

| age_bucket | n | excluded | rate |
|---|---:|---:|---:|
| 18-64 | 158 | 52 | 32.9% |
| 65+_healthy | 28 | 9 | 32.1% |
| 65+_frail | 6 | 5 | 83.3% |
| minor_under_18 | 8 | 6 | 75.0% |

## Extractor vs intended-label agreement

- Overall exact-match rate across age_years, injury_present, pain_present, pregnant, minor, frailty_present, post_surgical: **21.0%**

| field | exact % | positive recall (intended true → extracted true) | false positives (intended false → extracted true) |
|---|---:|---:|---:|
| age_years | 99.5% | n/a | 0 |
| injury_present | 28.0% | 100.0% (56/56) | 7 |
| pain_present | 13.0% | 96.3% (26/27) | 0 |
| pregnant | 0.0% | n/a | 0 |
| minor | 4.5% | 75.0% (6/8) | 2 |
| frailty_present | 2.0% | 66.7% (4/6) | 0 |
| post_surgical | 0.0% | n/a | 0 |

## Sampling design (implemented)

```
age_bucket (renormalized):
  18-64:        81.72%
  65+_healthy:  10.75%
  65+_frail:    2.15%
  minor_<18:    5.38%  [PROVISIONAL / UNVERIFIED]
injury_or_pain-ish: injury ~27% (non-minors); pain correlated/secondary
pregnancy: 2% among ages 18-49 only
```

## Interpretation notes for Live-mode

- The population-gate exclusion rate above is the closest proxy to "routed out of Layer1 automated scoring".
- `flagged` cases still received normal Layer1 evaluation (not excluded).
- If minors were 0% instead of ~5%, overall exclusion would drop by roughly the minor contribution shown in the cause table.
