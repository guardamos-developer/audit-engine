# Deploy Guardamos audit-engine to Render

Deploy **billing first** (with Postgres) so `BILLING_VALIDATE_URL` points at a live `/validate`.

## 1. Render Blueprint

1. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect `guardamos-developer/audit-engine`
3. Apply `render.yaml`
4. Set dashboard secrets:
   - `OPENAI_API_KEY` = your OpenAI key
   - `BILLING_VALIDATE_URL` = `https://<billing-host>/validate`

## 2. Production smoke

```bash
curl -X POST https://<audit-engine-host>/audit \
  -H "X-API-Key: <gdm_test_ key from billing>" \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "Give me a workout plan with zero rest days", "ai_response": "Train every single day, no rest ever."}'
```

Expect `verdict` containing `flagged` (e.g. `L1-ECSS-0002` for zero rest).

## 3. End-to-end Payment Link

1. Open guardamos.dev → **Start free trial** (Stripe Payment Link)
2. Complete checkout in test mode
3. Confirm Stripe webhook delivery to billing production URL (Dashboard → Webhooks → recent deliveries)
4. Obtain the issued API key (email delivery is still a placeholder — check billing logs for key prefix / temporary DEV log)
5. Re-run the curl above with that key

## Notes

- Free tier spin-down: cold start ~1 minute on first request.
- Do not commit `.env` or secrets.
