"""Tests for render/ package changes."""
import pytest
from render.data import expected_wind_chips
from render.charts import prob_trend_svg


# ── Fixtures ─────────────────────────────────────────────────────────────────

CFG = {
    "sailing": {
        "wind_speed_min": 2.0,
        "wind_speed_max": 12.0,
        "wind_dir_consistency_max": 30.0,
    }
}

HEADLINE_NWP = {
    "nwp_forecast": {
        "mean_wind_kn": 5.2,
        "max_gust_kn": 9.1,
        "dir_consistency_deg": 22.0,
    }
}

HEADLINE_OBS = {
    "window_wind": {
        "speeds_kn":      [4.0, 5.0, 6.0, 5.5, 4.5],
        "gusts_kn":       [6.0, 7.0, 8.0, 7.5, 6.5],
        "directions_deg": [180, 185, 175, 180, 182],
    }
}


# ── expected_wind_chips compact flag ─────────────────────────────────────────

def test_full_chips_include_dir_nwp():
    html = expected_wind_chips(HEADLINE_NWP, CFG)
    assert "dir" in html

def test_compact_chips_exclude_dir_nwp():
    html = expected_wind_chips(HEADLINE_NWP, CFG, compact=True)
    assert "dir" not in html

def test_compact_chips_still_include_avg_and_gust_nwp():
    html = expected_wind_chips(HEADLINE_NWP, CFG, compact=True)
    assert "avg" in html
    assert "gust" in html

def test_full_chips_include_dir_obs():
    html = expected_wind_chips(HEADLINE_OBS, CFG)
    assert "dir" in html

def test_compact_chips_exclude_dir_obs():
    html = expected_wind_chips(HEADLINE_OBS, CFG, compact=True)
    assert "dir" not in html


# ── prob_trend_svg size param and colours ─────────────────────────────────────

SNAPS_2 = [
    {"snapshot": "2026-03-15T08:00:00", "probability": 0.1, "good": False,
     "condition_score": 10, "threshold": 0.3},
    {"snapshot": "2026-03-15T14:00:00", "probability": 0.2, "good": False,
     "condition_score": 20, "threshold": 0.3},
]

def test_prob_trend_svg_returns_empty_for_single_snap():
    assert prob_trend_svg([SNAPS_2[0]]) == ""

def test_prob_trend_svg_default_uses_inline_style():
    svg = prob_trend_svg(SNAPS_2)
    assert 'style="width:100%' in svg

def test_prob_trend_svg_size_param_sets_width_height_attributes():
    svg = prob_trend_svg(SNAPS_2, size=(90, 30))
    assert 'width="90"' in svg
    assert 'height="30"' in svg
    # inline style should NOT be present when size is given
    assert 'style="width:100%' not in svg

def test_prob_trend_svg_uses_accent_colour_for_line():
    svg = prob_trend_svg(SNAPS_2)
    # connecting polyline should use accent colour #22d3ee
    assert '#22d3ee' in svg

def test_prob_trend_svg_threshold_line_uses_muted_colour():
    svg = prob_trend_svg(SNAPS_2)
    # threshold dashed line should use muted #94a3b8
    assert '#94a3b8' in svg
    assert 'stroke="#94a3b8"' in svg


# ── render_html smoke tests ───────────────────────────────────────────────────

from render_html import build_html

CFG_FULL = {
    "sailing": {
        "window_start": "08:00",
        "window_end": "16:00",
        "wind_speed_min": 2.0,
        "wind_speed_max": 12.0,
        "wind_dir_consistency_max": 30.0,
    },
    "prediction": {"min_good_fraction": 0.3},
}

PREDICTIONS = [
    {
        "snapshot": "2026-03-15T08:00:00",
        "predicting_date": "2026-03-15",
        "probability": 0.12,
        "good": False,
        "threshold": 0.3,
        "condition_score": 12,
        "condition_label": "No wind",
        "condition_icon": "🌫",
        "condition_source": "forecast",
        "nwp_forecast": {"mean_wind_kn": 3.1, "max_gust_kn": 9.3,
                         "cloud_cover_pct": 99, "blh_m": 278,
                         "dir_consistency_deg": 52.3},
    },
    {
        "snapshot": "2026-03-15T08:00:00",
        "predicting_date": "2026-03-16",
        "probability": 0.10,
        "good": False,
        "threshold": 0.3,
        "condition_score": 10,
        "condition_label": "No wind",
        "condition_icon": "🌫",
        "condition_source": "forecast",
        "is_extended_forecast": True,
        "lead_days": 1,
        "nwp_forecast": {"mean_wind_kn": 2.8, "max_gust_kn": 7.1, "dir_consistency_deg": 40.0},
    },
    {
        "snapshot": "2026-03-15T08:00:00",
        "predicting_date": "2026-03-17",
        "probability": 0.08,
        "good": False,
        "threshold": 0.3,
        "condition_score": 8,
        "condition_label": "No wind",
        "condition_icon": "🌫",
        "condition_source": "forecast",
        "is_extended_forecast": True,
        "lead_days": 2,
        "nwp_forecast": {"mean_wind_kn": 4.2, "max_gust_kn": 11.0},
    },
]


def test_build_html_contains_hero_card():
    html = build_html(PREDICTIONS, CFG_FULL)
    assert 'class="hero-card"' in html

def test_build_html_contains_compact_grid():
    html = build_html(PREDICTIONS, CFG_FULL)
    assert 'class="compact-grid"' in html

def test_build_html_dark_bg_token():
    html = build_html(PREDICTIONS, CFG_FULL)
    assert '#0f1117' in html

def test_build_html_no_fold_out():
    """Old past-week fold-out must be gone."""
    html = build_html(PREDICTIONS, CFG_FULL)
    assert 'past-week-wrap' not in html

def test_build_html_empty_predictions_no_crash():
    html = build_html([], CFG_FULL)
    assert 'no-data' in html
