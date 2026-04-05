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


import resend
from notify.notify import main, _format_subject_date


def test_format_subject_date_no_leading_zero():
    # Single-digit day must have no leading zero
    result = _format_subject_date("2026-03-05")
    assert result.startswith("Thursday")
    assert "5 March 2026" in result
    assert "05" not in result


def test_main_exits_1_missing_resend_key(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["notify"])
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("NOTIFY_EMAIL", "test@example.com")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "RESEND_API_KEY" in capsys.readouterr().out


def test_main_exits_1_missing_notify_email(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["notify"])
    monkeypatch.setenv("RESEND_API_KEY", "key")
    monkeypatch.delenv("NOTIFY_EMAIL", raising=False)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "NOTIFY_EMAIL" in capsys.readouterr().out


def test_main_exits_0_silently_when_not_good(monkeypatch, tmp_path):
    """No email sent when prediction is not good."""
    monkeypatch.setattr("sys.argv", ["notify"])
    monkeypatch.setenv("RESEND_API_KEY", "key")
    monkeypatch.setenv("NOTIFY_EMAIL", "test@example.com")
    today = date.today().isoformat()
    preds = [{"predicting_date": today, "snapshot": today + "T04:00:00",
               "good": False, "probability": 0.1}]
    (tmp_path / "config.toml").write_text(
        '[sailing]\nwindow_start = "08:00"\nwindow_end = "16:00"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("notify.notify.load_forecast_snapshots", lambda: preds)
    monkeypatch.setattr(resend.Emails, "send", lambda p: pytest.fail("send called unexpectedly"))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_main_sends_email_when_good(monkeypatch, tmp_path):
    """Email is sent when prediction is good."""
    monkeypatch.setattr("sys.argv", ["notify"])
    monkeypatch.setenv("RESEND_API_KEY", "key")
    monkeypatch.setenv("NOTIFY_EMAIL", "test@example.com")
    today = date.today().isoformat()
    preds = [{"predicting_date": today, "snapshot": today + "T04:00:00",
               "good": True, "probability": 0.75, "condition_label": "Good"}]
    (tmp_path / "config.toml").write_text(
        '[sailing]\nwindow_start = "08:00"\nwindow_end = "16:00"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("notify.notify.load_forecast_snapshots", lambda: preds)

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
