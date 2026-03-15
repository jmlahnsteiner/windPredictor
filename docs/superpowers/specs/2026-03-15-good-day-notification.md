# Good-Day Email Notification

**Date:** 2026-03-15
**Status:** Approved

## Goal

Send an email automatically when the morning forecast predicts a good sailing day, giving the user a heads-up ~2 hours before the sailing window opens.

---

## Trigger

The 04:00 UTC GitHub Actions job only (06:00 CET / 07:00 CEST). At this hour `_target_date()` returns **today** (sailing window has not yet closed), so the prediction covers the upcoming window.

No email is sent on any other run.

---

## Condition

Read the most recent prediction row for today's date from Supabase. If `good = 1`, send the email. Otherwise exit 0 silently.

"Most recent" = highest `snapshot_dt` for `predicting_date = today`.

---

## Email

**From:** `WindPredictor <onboarding@resend.dev>` (Resend default sender, no domain config needed)

**To:** `NOTIFY_EMAIL` environment variable

**Subject:** `⛵ Good sailing today — {Day, D Month YYYY}`

**Body (plain text):**

```
Good sailing conditions forecast for today's window ({window_start}–{window_end}).

Probability:  {pct}%  ({condition_label})
Expected wind: avg {mean_wind} kn · gust {max_gust} kn
Sailing window: {window_start}–{window_end}

Full forecast: https://jmlahnsteiner.github.io/windPredictor/
```

NWP wind values (`mean_wind_kn`, `max_gust_kn`) come from the `nwp_forecast` JSON column of the most recent prediction row. If unavailable, those lines are omitted.

---

## Files Changed

| File | Change |
|------|--------|
| `notify/notify.py` | New script — query Supabase, send email via Resend if good |
| `notify/__init__.py` | New (empty) |
| `.github/workflows/forecast.yml` | Add `Send good-day notification` step to 04:00 UTC job |
| `requirements.txt` | Add `resend` |

---

## `notify/notify.py` — Logic

```
1. Load .env (for local testing)
2. Connect to Supabase via SUPABASE_DB_URL (same utils/db path as the rest of the codebase)
3. Query: SELECT probability, good, snapshot_dt, nwp_forecast
          FROM predictions
          WHERE predicting_date = today
          ORDER BY snapshot_dt DESC
          LIMIT 1
4. If no row or good != 1 → exit 0
5. Parse nwp_forecast JSON for mean_wind_kn, max_gust_kn
6. Load config.toml for window_start, window_end
7. POST to Resend API with plain-text email body
8. Print confirmation; exit 0 on success, exit 1 on send failure
```

---

## Workflow Step

Added to the existing `04 00 * * *` job in `forecast.yml`, **after** the predict step:

```yaml
- name: Send good-day notification
  run: python notify/notify.py
  env:
    SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
    RESEND_API_KEY:  ${{ secrets.RESEND_API_KEY }}
    NOTIFY_EMAIL:    ${{ secrets.NOTIFY_EMAIL }}
```

The step runs unconditionally (no `if:` guard) — `notify.py` itself decides whether to send based on the prediction value.

---

## Secrets Required

| Secret | Where to get it |
|--------|----------------|
| `RESEND_API_KEY` | resend.com → API Keys (free tier: 100 emails/day) |
| `NOTIFY_EMAIL` | Your email address |
| `SUPABASE_DB_URL` | Already set |

---

## Error Handling

- No prediction row for today → exit 0 (silent, not an error)
- `good = 0` → exit 0 (silent)
- Resend API error → print error, exit 1 (workflow step fails visibly in Actions UI)
- Missing `RESEND_API_KEY` → exit 1 with clear message

---

## What Stays the Same

- All prediction and render pipeline code
- All other workflow jobs
- Supabase schema
