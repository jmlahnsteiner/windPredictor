# Good-Day Email Notification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send a plain-text email via Resend when the 04:00 UTC forecast predicts a good sailing day.

**Architecture:** A new `notify/notify.py` script reads `predictions.json`, picks the latest entry for today, and POSTs to the Resend API if `good` is truthy. Pure helper functions (`load_today_entry`, `build_body`) are factored out so they can be unit-tested without I/O or network mocking. The workflow step uses an `if:` guard so it only runs on the 04:00 UTC cron.

**Tech Stack:** Python 3.12, `resend` SDK, `tomllib` (stdlib), `python-dotenv` (already in deps), `pytest` for tests.

---

## Chunk 1: Scaffold, helpers, and tests

### Task 1: Add `resend` to requirements.txt

**Files:**
- Modify: `requirements.txt`

> Tasks 1 and 2 are pure scaffolding with no testable behavior — no red/green cycle needed.

- [ ] **Step 1: Pin the latest version**

Run the following to find the current latest:

```bash
pip index versions resend 2>/dev/null | head -1
```

Then append the pinned version to `requirements.txt`, e.g.:

```
resend==2.10.0
```

- [ ] **Step 2: Verify it installs**

```bash
pip install resend
```

Expected: installs without error.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add resend dependency for email notification"
```

---

### Task 2: Create module scaffold

**Files:**
- Create: `notify/__init__.py`
- Create: `tests/notify/__init__.py`

> Check whether `tests/__init__.py` exists before creating the subdirectory init. The existing test dirs (`tests/utils/`, `tests/render/`) all have `__init__.py` files — follow the same pattern.

- [ ] **Step 1: Create the empty init files**

```bash
mkdir -p notify tests/notify && touch notify/__init__.py tests/notify/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add notify/__init__.py tests/notify/__init__.py
git commit -m "feat: add notify module scaffold"
```

---

### Task 3: `load_today_entry` — filter predictions.json for today

**Files:**
- Create: `notify/notify.py` (partial — just `load_today_entry`)
- Create: `tests/notify/test_notify.py`

**What `load_today_entry` does:**
- Accepts `predictions: list[dict]` (already-parsed JSON) and `today: str` (ISO date, e.g. `"2026-03-15"`)
- Filters entries where `entry["predicting_date"] == today`
- If no entries → returns `None`
- Otherwise picks the entry with the highest `"snapshot"` value (ISO-8601 string — sorts correctly as string without parsing)

- [ ] **Step 1: Write the failing tests**

```python
# tests/notify/test_notify.py
"""Tests for notify/notify.py."""
import json
import pytest
from datetime import date
from notify.notify import load_today_entry


ENTRY_A = {
    "predicting_date": "2026-03-15",
    "snapshot": "2026-03-15T04:00:00",
    "good": True,
    "probability": 0.72,
    "condition_label": "Good",
    "nwp_forecast": {"mean_wind_kn": 5.2, "max_gust_kn": 9.1},
}
ENTRY_B = {
    "predicting_date": "2026-03-15",
    "snapshot": "2026-03-15T06:00:00",  # later snapshot
    "good": True,
    "probability": 0.80,
    "condition_label": "Excellent",
    "nwp_forecast": {"mean_wind_kn": 6.0, "max_gust_kn": 10.5},
}
ENTRY_OTHER = {
    "predicting_date": "2026-03-16",
    "snapshot": "2026-03-15T04:00:00",
    "good": True,
    "probability": 0.60,
}


def test_load_today_entry_returns_none_when_empty():
    assert load_today_entry([], "2026-03-15") is None


def test_load_today_entry_returns_none_when_no_match():
    assert load_today_entry([ENTRY_OTHER], "2026-03-15") is None


def test_load_today_entry_returns_single_match():
    assert load_today_entry([ENTRY_A, ENTRY_OTHER], "2026-03-15") == ENTRY_A


def test_load_today_entry_picks_latest_snapshot():
    # ENTRY_B has a later snapshot — must be returned
    result = load_today_entry([ENTRY_A, ENTRY_B, ENTRY_OTHER], "2026-03-15")
    assert result == ENTRY_B
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/notify/test_notify.py -v
```

Expected: `ModuleNotFoundError: No module named 'notify.notify'`

- [ ] **Step 3: Implement `load_today_entry` in `notify/notify.py`**

```python
"""notify/notify.py — Good-day email notification."""
from __future__ import annotations

import json
import os
import sys
import tomllib
from datetime import date, datetime
from pathlib import Path

import resend  # top-level import — makes monkeypatching reliable in tests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_today_entry(predictions: list[dict], today: str) -> dict | None:
    """Return the latest-snapshot prediction entry for *today*, or None."""
    todays = [e for e in predictions if e.get("predicting_date") == today]
    if not todays:
        return None
    return max(todays, key=lambda e: e.get("snapshot", ""))
```

- [ ] **Step 4: Run tests — expect all 4 to pass**

```bash
pytest tests/notify/test_notify.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add notify/notify.py tests/notify/test_notify.py
git commit -m "feat: add load_today_entry helper with tests"
```

---

### Task 4: `build_body` — compose the plain-text email body

**Files:**
- Modify: `notify/notify.py` (add `build_body`)
- Modify: `tests/notify/test_notify.py` (add body tests)

**What `build_body` does:**
- Accepts `entry: dict`, `window_start: str`, `window_end: str`
- Reads `probability` → formats as integer percentage
- Reads `condition_label` → falls back to `"Good"` if absent/falsy
- Reads `nwp_forecast.mean_wind_kn` and `nwp_forecast.max_gust_kn` → includes the wind line only if **both** are present
- Returns a multi-line plain-text string

- [ ] **Step 1: Add the failing tests**

Append to `tests/notify/test_notify.py`:

```python
from notify.notify import build_body


def test_build_body_with_nwp():
    entry = {
        "probability": 0.72,
        "condition_label": "Good",
        "nwp_forecast": {"mean_wind_kn": 5.2, "max_gust_kn": 9.1},
    }
    body = build_body(entry, "08:00", "16:00")
    assert "72%" in body
    assert "Good" in body
    assert "avg 5.2 kn" in body
    assert "gust 9.1 kn" in body
    assert "08:00–16:00" in body


def test_build_body_without_nwp():
    entry = {"probability": 0.55, "condition_label": "OK"}
    body = build_body(entry, "08:00", "16:00")
    assert "55%" in body
    assert "avg" not in body
    assert "gust" not in body


def test_build_body_condition_label_fallback():
    entry = {"probability": 0.65}
    body = build_body(entry, "08:00", "16:00")
    assert "Good" in body  # fallback when field absent


def test_build_body_partial_nwp_omits_wind_line():
    # Only mean_wind_kn present, max_gust_kn absent → omit wind line entirely
    entry = {
        "probability": 0.60,
        "nwp_forecast": {"mean_wind_kn": 5.0},  # max_gust_kn missing
    }
    body = build_body(entry, "08:00", "16:00")
    assert "avg" not in body
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/notify/test_notify.py::test_build_body_with_nwp -v
```

Expected: `ImportError` (function not defined yet)

- [ ] **Step 3: Implement `build_body` in `notify/notify.py`**

Add after `load_today_entry`:

```python
def build_body(entry: dict, window_start: str, window_end: str) -> str:
    """Build the plain-text email body for a good-day notification."""
    pct = round(float(entry.get("probability", 0)) * 100)
    label = entry.get("condition_label") or "Good"

    nwp = entry.get("nwp_forecast") or {}
    mean_wind = nwp.get("mean_wind_kn")
    max_gust = nwp.get("max_gust_kn")
    wind_line = (
        f"Expected wind: avg {mean_wind} kn · gust {max_gust} kn\n"
        if mean_wind is not None and max_gust is not None
        else ""
    )

    return (
        f"Good sailing conditions forecast for today's window "
        f"({window_start}–{window_end}).\n"
        f"\n"
        f"Probability:  {pct}%  ({label})\n"
        f"{wind_line}"
        f"Sailing window: {window_start}–{window_end}\n"
        f"\n"
        f"Full forecast: https://jmlahnsteiner.github.io/windPredictor/\n"
    )
```

- [ ] **Step 4: Run all notify tests — expect 8 to pass**

```bash
pytest tests/notify/ -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add notify/notify.py tests/notify/test_notify.py
git commit -m "feat: add build_body helper with tests"
```

---

### Task 5: `main()` — wire everything together and send email

**Files:**
- Modify: `notify/notify.py` (add `main`)
- Modify: `tests/notify/test_notify.py` (add main integration tests)

**What `main()` does (in order):**
1. Check `RESEND_API_KEY` env var — missing → `print(error)`, `sys.exit(1)`
2. Check `NOTIFY_EMAIL` env var — missing → `print(error)`, `sys.exit(1)`
3. Load `config.toml` for `window_start`, `window_end`
4. Load `predictions.json` from repo root
5. Filter for today → `load_today_entry()`
6. No entry → `sys.exit(0)` silently
7. `entry["good"]` falsy → `sys.exit(0)` silently
8. Build subject line: `⛵ Good sailing today — {weekday}, {D Month YYYY}` (no leading zero on day)
9. Build body via `build_body()`
10. Set `resend.api_key`, call `resend.Emails.send()` — exception → print error, `sys.exit(1)`
11. Print confirmation, `sys.exit(0)`

- [ ] **Step 1: Add the failing tests**

Append to `tests/notify/test_notify.py`:

```python
import resend
from notify.notify import main, _format_subject_date


def test_format_subject_date_no_leading_zero():
    # Single-digit day must have no leading zero
    result = _format_subject_date("2026-03-05")
    assert result.startswith("Thursday")
    assert "5 March 2026" in result
    assert "05" not in result


def test_main_exits_1_missing_resend_key(monkeypatch, capsys):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("NOTIFY_EMAIL", "test@example.com")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "RESEND_API_KEY" in capsys.readouterr().out


def test_main_exits_1_missing_notify_email(monkeypatch, capsys):
    monkeypatch.setenv("RESEND_API_KEY", "key")
    monkeypatch.delenv("NOTIFY_EMAIL", raising=False)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "NOTIFY_EMAIL" in capsys.readouterr().out


def test_main_exits_0_silently_when_not_good(monkeypatch, tmp_path):
    """No email sent when prediction is not good."""
    monkeypatch.setenv("RESEND_API_KEY", "key")
    monkeypatch.setenv("NOTIFY_EMAIL", "test@example.com")
    today = date.today().isoformat()
    preds = [{"predicting_date": today, "snapshot": today + "T04:00:00",
               "good": False, "probability": 0.1}]
    (tmp_path / "predictions.json").write_text(json.dumps(preds))
    (tmp_path / "config.toml").write_text(
        '[sailing]\nwindow_start = "08:00"\nwindow_end = "16:00"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resend.Emails, "send", lambda p: pytest.fail("send called unexpectedly"))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_main_sends_email_when_good(monkeypatch, tmp_path):
    """Email is sent when prediction is good."""
    monkeypatch.setenv("RESEND_API_KEY", "key")
    monkeypatch.setenv("NOTIFY_EMAIL", "test@example.com")
    today = date.today().isoformat()
    preds = [{"predicting_date": today, "snapshot": today + "T04:00:00",
               "good": True, "probability": 0.75, "condition_label": "Good"}]
    (tmp_path / "predictions.json").write_text(json.dumps(preds))
    (tmp_path / "config.toml").write_text(
        '[sailing]\nwindow_start = "08:00"\nwindow_end = "16:00"\n'
    )
    monkeypatch.chdir(tmp_path)

    sent = {}
    def fake_send(params):
        sent.update(params)
        return {"id": "test-id"}

    monkeypatch.setattr(resend.Emails, "send", fake_send)

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert sent["to"] == ["test@example.com"]
    assert "⛵" in sent["subject"]
    assert "75%" in sent["text"]
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/notify/test_notify.py::test_main_exits_1_missing_resend_key -v
```

Expected: `ImportError` (main not defined yet)

- [ ] **Step 3: Implement `main()` in `notify/notify.py`**

Add at the end of `notify/notify.py`:

```python
def _format_subject_date(today: str) -> str:
    """Format today's date as 'Monday, 15 March 2026' (no leading zero, cross-platform)."""
    dt = datetime.strptime(today, "%Y-%m-%d")
    return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B %Y')}"


def main() -> None:
    # 1-2. Fail fast: check required env vars before any file I/O
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print("ERROR: RESEND_API_KEY environment variable is not set.")
        sys.exit(1)

    notify_email = os.environ.get("NOTIFY_EMAIL")
    if not notify_email:
        print("ERROR: NOTIFY_EMAIL environment variable is not set.")
        sys.exit(1)

    # 3. Load config
    with open(Path("config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    sailing = cfg.get("sailing", {})
    window_start = sailing.get("window_start", "08:00")
    window_end = sailing.get("window_end", "16:00")

    # 4. Load predictions.json
    with open(Path("predictions.json")) as f:
        predictions = json.load(f)

    # 5-6. Find today's latest entry
    today = date.today().isoformat()
    entry = load_today_entry(predictions, today)
    if entry is None:
        sys.exit(0)

    # 7. Check if good
    if not entry.get("good"):
        sys.exit(0)

    # 8-9. Build email
    subject = f"⛵ Good sailing today — {_format_subject_date(today)}"
    body = build_body(entry, window_start, window_end)

    # 10. Send via Resend
    resend.api_key = api_key
    try:
        resend.Emails.send({
            "from": "WindPredictor <onboarding@resend.dev>",
            "to": [notify_email],
            "subject": subject,
            "text": body,
        })
    except Exception as exc:
        print(f"ERROR: Resend API call failed: {exc}")
        sys.exit(1)

    # 11. Confirm
    print(f"Notification sent to {notify_email}: {subject}")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all notify tests — expect 13 to pass**

```bash
pytest tests/notify/ -v
```

Expected: `13 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add notify/notify.py tests/notify/test_notify.py
git commit -m "feat: implement notify/notify.py with email sending"
```

---

### Task 6: Add workflow step to `forecast.yml`

**Files:**
- Modify: `.github/workflows/forecast.yml`

The step goes **after** the `Predict` step and **before** `Render HTML`.

> **Critical — cron string must match exactly:** `github.event.schedule` is compared as a literal string against the cron definition. The current workflow defines `cron: '0  4 * * *'` with **two spaces** between `0` and `4` (line 5). The `if:` guard must use the same two-space string: `"github.event.schedule == '0  4 * * *'"`. A single-space guard will silently never match.

- [ ] **Step 1: Insert the workflow step**

In `.github/workflows/forecast.yml`, after the `Predict` step block and before `Render HTML`, insert:

```yaml
      - name: Send good-day notification
        if: "github.event.schedule == '0  4 * * *'"
        run: python notify/notify.py
        env:
          RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
          NOTIFY_EMAIL:   ${{ secrets.NOTIFY_EMAIL }}
```

The surrounding context should look like:

```yaml
      - name: Predict
        env:
          SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
        run: python model/predict.py

      - name: Send good-day notification
        if: "github.event.schedule == '0  4 * * *'"
        run: python notify/notify.py
        env:
          RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
          NOTIFY_EMAIL:   ${{ secrets.NOTIFY_EMAIL }}

      - name: Render HTML
        env:
          SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
        run: python render_html.py
```

- [ ] **Step 2: Validate the YAML is well-formed**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/forecast.yml'))"
```

Expected: no output (no errors). Install `pyyaml` first if needed: `pip install pyyaml`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/forecast.yml
git commit -m "feat: add good-day email notification workflow step"
```

---

### Task 7: Add secrets to GitHub repository

This is a manual step — no code changes.

- [ ] **Step 1: Get a Resend API key**

1. Go to [resend.com](https://resend.com) → sign in (free tier: 100 emails/day)
2. API Keys → Create API Key → copy the key

- [ ] **Step 2: Add secrets to GitHub**

In GitHub repo → Settings → Secrets and variables → Actions:

| Name | Value |
|------|-------|
| `RESEND_API_KEY` | Key from Resend |
| `NOTIFY_EMAIL` | Your email address |

- [ ] **Step 3: Verify end-to-end**

Trigger a manual `workflow_dispatch` run from Actions. The notify step will be skipped (not the 04:00 cron). To force a test send, temporarily change the `if:` guard to `if: true`, trigger manually, then revert.

---

## End of Plan

After all tasks complete, run the full test suite one final time:

```bash
pytest -v
```
