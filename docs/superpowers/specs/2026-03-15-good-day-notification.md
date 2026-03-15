# Good-Day Email Notification

**Date:** 2026-03-15
**Status:** Approved

## Goal

Send an email automatically when the morning forecast predicts a good sailing day, giving the user a heads-up ~2 hours before the sailing window opens.

---

## Trigger

The 04:00 UTC GitHub Actions job only (05:00 CET / 06:00 CEST). At this hour `_target_date()` returns **today** (sailing window has not yet closed), so the prediction covers the upcoming window.

No email is sent on any other run.

---

## Data Source

Read from `predictions.json` (the rolling forecast file committed to the repo by the predict step, which runs before this step in the same job). This file contains fully-enriched entries including `nwp_forecast`.

No separate Supabase query is needed for the notification.

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

- `condition_label`: the `condition_label` field from the prediction entry (e.g. "Good", "Excellent"). Falls back to "Good" if absent.
- `mean_wind` / `max_gust`: from `nwp_forecast.mean_wind_kn` / `nwp_forecast.max_gust_kn` in the prediction entry. If the `nwp_forecast` key is absent or the values are None, those lines are omitted from the body.

---

## Files Changed

| File | Change |
|------|--------|
| `notify/notify.py` | New script — read predictions.json, send email via Resend if good |
| `notify/__init__.py` | New (empty) |
| `.github/workflows/forecast.yml` | Add `Send good-day notification` step to 04:00 UTC job |
| `requirements.txt` | Add `resend` |

---

## `notify/notify.py` — Logic

```
1. Load .env (for local testing)
2. Check RESEND_API_KEY is set; if missing → print error, exit 1
3. Check NOTIFY_EMAIL is set; if missing → print error, exit 1
4. Load config.toml for window_start, window_end
   (values are already human-readable local time strings — no timezone conversion needed)
5. Load predictions.json (path relative to repo root)
6. Filter entries where predicting_date == today
7. If no entries → exit 0 silently
8. Pick entry with highest "snapshot" field (ISO-8601 string, sorts lexicographically)
9. If not good (falsy) → exit 0 silently
10. Extract: probability, condition_label (fallback "Good"), nwp_forecast (may be absent)
11. Build plain-text email body; omit wind lines if nwp_forecast unavailable
12. POST to Resend API
13. Print confirmation; exit 0 on success, exit 1 on API error
```

---

## Workflow Step

Added to the **shared** `forecast` job in `forecast.yml`, **after** the predict step. The `if:` guard restricts execution to the 04:00 UTC cron only:

```yaml
- name: Send good-day notification
  if: "github.event.schedule == '0  4 * * *'"
  run: python notify/notify.py
  env:
    RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
    NOTIFY_EMAIL:   ${{ secrets.NOTIFY_EMAIL }}
```

The `if:` guard ensures the step is skipped on all five other daily runs. `notify.py` itself also checks `good` before sending, so no email is sent even if the guard were absent and the day is not good.

---

## Secrets Required

| Secret | Where to get it |
|--------|----------------|
| `RESEND_API_KEY` | resend.com → API Keys (free tier: 100 emails/day) |
| `NOTIFY_EMAIL` | Your email address |

---

## Error Handling

- No prediction row for today → exit 0 (silent, not an error)
- Prediction not good (falsy) → exit 0 (silent)
- Missing `RESEND_API_KEY` → print error, exit 1
- Missing `NOTIFY_EMAIL` → print error, exit 1
- Resend API error → print error, exit 1 (workflow step fails visibly in Actions UI)

---

## What Stays the Same

- All prediction and render pipeline code
- All other workflow jobs
- Supabase schema
