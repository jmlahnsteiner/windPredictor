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
2. Load config.toml for window_start, window_end
3. Load predictions.json (path relative to repo root, same as render_html.py uses)
4. Filter entries where predicting_date == today
5. If no entries → exit 0 silently
6. Pick entry with highest snapshot_dt
7. If good != True (bool) or good != 1 (int) → exit 0 silently
8. Extract: probability, condition_label, nwp_forecast (may be absent)
9. Build plain-text email body; omit wind lines if nwp_forecast unavailable
10. Check RESEND_API_KEY is set; if missing → print error, exit 1
11. Check NOTIFY_EMAIL is set; if missing → print error, exit 1
12. POST to Resend API
13. Print confirmation; exit 0 on success, exit 1 on API error
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
