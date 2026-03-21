"""render/data.py — Data helpers and history rendering for the forecast page."""
import math
import os

import numpy as np

from utils.circular import circular_std


def score_to_hex(score: int) -> str:
    """Map a 0-100 condition score to a hex colour. (Moved from render_html.py)"""
    if score < 15:   return "#94a3b8"
    if score < 30:   return "#7dd3fc"
    if score < 45:   return "#fbbf24"
    if score < 60:   return "#86efac"
    if score < 75:   return "#22c55e"
    if score < 90:   return "#10b981"
    return "#06b6d4"


def window_stats(window_wind: dict, cfg: dict) -> dict:
    """Compute observed stats from window_wind (local station data).
    (Previously _window_stats in render_html.py)"""
    speeds = window_wind.get("speeds_kn", [])
    gusts  = window_wind.get("gusts_kn", [])
    dirs   = window_wind.get("directions_deg", [])
    if len(speeds) < 3:
        return {}

    sc       = cfg.get("sailing", {})
    wind_min = sc.get("wind_speed_min", 2.0)
    wind_max = sc.get("wind_speed_max", 10.0)

    pct_good = round(sum(1 for s in speeds if wind_min <= s <= wind_max) / len(speeds) * 100)

    dir_std: float | None = None
    if len(dirs) >= 3:
        rad = np.radians(dirs)
        R   = float(np.hypot(np.sin(rad).mean(), np.cos(rad).mean()))
        dir_std = round(math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9)))), 1)

    valid_gusts = [g for g in gusts if g is not None]

    return {
        "mean_kn":     round(sum(speeds) / len(speeds), 1),
        "max_kn":      round(max(speeds), 1),
        "max_gust_kn": round(max(valid_gusts), 1) if valid_gusts else None,
        "pct_good":    pct_good,
        "dir_std_deg": dir_std,
    }


def expected_wind_chips(headline: dict, cfg: dict, compact: bool = False) -> str:
    """
    Return chip HTML for expected average wind, gusts, and direction consistency.
    Prefers observed window data (past days) over NWP (future days).
    Returns '' when neither source is available.
    When compact=True, the direction chip is omitted for space-efficient layouts.
    (Previously _expected_wind_chips in render_html.py)
    """
    obs = window_stats(headline.get("window_wind", {}), cfg)
    nwp = headline.get("nwp_forecast", {})
    chips = []

    if obs:
        chips.append(f'<span class="meta-chip exp-chip">💨 avg {obs["mean_kn"]} kn</span>')
        if obs.get("max_gust_kn") is not None:
            chips.append(f'<span class="meta-chip exp-chip">↑ gust {obs["max_gust_kn"]} kn</span>')
        if not compact and obs.get("dir_std_deg") is not None:
            chips.append(f'<span class="meta-chip exp-chip">〜 dir ±{obs["dir_std_deg"]}°</span>')
    elif nwp:
        if nwp.get("mean_wind_kn") is not None:
            chips.append(f'<span class="meta-chip exp-chip">💨 avg {nwp["mean_wind_kn"]} kn</span>')
        if nwp.get("max_gust_kn") is not None:
            chips.append(f'<span class="meta-chip exp-chip">↑ gust {nwp["max_gust_kn"]} kn</span>')
        if not compact and nwp.get("dir_consistency_deg") is not None:
            chips.append(f'<span class="meta-chip exp-chip">〜 dir ±{nwp["dir_consistency_deg"]}°</span>')

    return "".join(chips)


def stats_html(headline: dict, cfg: dict) -> str:
    """
    Render secondary stats rows (details not shown in the meta chips).
    Returns '' when nothing secondary is available.
    (Previously _stats_html in render_html.py)
    """
    rows = []

    obs = window_stats(headline.get("window_wind", {}), cfg)
    if obs:
        rows.append(
            f'<div class="stats-row stats-observed">'
            f'<span class="stats-label">Observed</span>'
            f'<span class="stats-chip">{obs["pct_good"]}% in range</span>'
            f'<span class="stats-chip">max {obs["max_kn"]} kn</span>'
            f'</div>'
        )

    nwp = headline.get("nwp_forecast", {})
    if nwp:
        cloud_chip = f'<span class="stats-chip">cloud {nwp["cloud_cover_pct"]}%</span>' if nwp.get("cloud_cover_pct") is not None else ""
        blh_chip   = f'<span class="stats-chip">boundary layer {nwp["blh_m"]:,} m</span>' if nwp.get("blh_m") is not None else ""
        if cloud_chip or blh_chip:
            rows.append(
                f'<div class="stats-row stats-nwp">'
                f'<span class="stats-label">Forecast</span>'
                f'{cloud_chip}{blh_chip}'
                f'</div>'
            )

    if not rows:
        return ""
    return '<div class="stats-block">' + "".join(rows) + "</div>\n"


def history_html(
    db_path: str,
    days: int = 14,
    snapshots: list | None = None,
    cfg: dict | None = None,
) -> str:
    """
    Render prediction accuracy as a <details> foldout with stat cards + time-series chart.
    Includes a 1w/2w interactive toggle (rendered as two static SVGs).
    snapshots: full list of forecast snapshot dicts (for intraday window_wind data).
    Returns '' if no history is available.
    """
    from datetime import date, timedelta
    from utils.db import backend, DEFAULT_SQLITE
    from model.history import load_history, accuracy_summary
    from render.charts import history_chart_svg

    bk = backend()
    actual_path = db_path or DEFAULT_SQLITE
    if bk == "sqlite" and not os.path.exists(actual_path):
        return ""

    # Load up to 14 days; subset to 7 for the short view
    df = load_history(db_path=actual_path, days=14)
    if df.empty:
        return ""

    df = df.sort_values("snapshot_dt").groupby("predicting_date").last().reset_index()
    df = df.sort_values("predicting_date")

    threshold = float(df["threshold"].iloc[-1]) if len(df) > 0 else 0.3

    def _rows(dataframe):
        result = []
        for _, row in dataframe.iterrows():
            af = row["actual_frac"]
            result.append({
                "predicting_date": row["predicting_date"],
                "probability":     float(row["probability"]),
                "actual_frac":     float(af) if (af is not None and af == af) else None,
            })
        return result

    cutoff_7 = (date.today() - timedelta(days=7)).isoformat()
    df7  = df[df["predicting_date"] >= cutoff_7]
    rows_14 = _rows(df)
    rows_7  = _rows(df7)

    # Build window_wind lookup from snapshot payloads (for intraday strips)
    wind_by_date: dict[str, dict] = {}
    if snapshots:
        from collections import defaultdict as _dd
        snaps_by_date = _dd(list)
        for s in snapshots:
            if s.get("predicting_date") and s.get("window_wind"):
                snaps_by_date[s["predicting_date"]].append(s)
        for d, snaps in snaps_by_date.items():
            best = max(snaps, key=lambda x: x.get("snapshot", ""))
            wind_by_date[d] = best["window_wind"]

    chart_7  = history_chart_svg(rows_7,  threshold=threshold, window_wind_by_date=wind_by_date, cfg=cfg)
    chart_14 = history_chart_svg(rows_14, threshold=threshold, window_wind_by_date=wind_by_date, cfg=cfg)

    has_toggle = len(rows_14) > len(rows_7) and len(rows_7) >= 2

    # All-time accuracy summary
    summary = accuracy_summary(db_path=actual_path, days=None)

    stat_cards = ""
    if summary:
        n   = summary["n_evaluated"]
        acc = round(summary["accuracy"] * 100)
        stat_cards += (
            f'<div class="hist-stat">'
            f'<span class="hist-stat-val">{acc}%</span>'
            f'<span class="hist-stat-lbl">Accuracy</span></div>'
        )
        stat_cards += (
            f'<div class="hist-stat">'
            f'<span class="hist-stat-val">{n}</span>'
            f'<span class="hist-stat-lbl">Days evaluated</span></div>'
        )
        if summary.get("precision") is not None:
            prec = round(summary["precision"] * 100)
            stat_cards += (
                f'<div class="hist-stat">'
                f'<span class="hist-stat-val">{prec}%</span>'
                f'<span class="hist-stat-lbl">Precision</span></div>'
            )
        if summary.get("recall") is not None:
            rec = round(summary["recall"] * 100)
            stat_cards += (
                f'<div class="hist-stat">'
                f'<span class="hist-stat-val">{rec}%</span>'
                f'<span class="hist-stat-lbl">Recall</span></div>'
            )
        stat_cards = f'<div class="hist-stats">{stat_cards}</div>'

    if has_toggle:
        toggle_html = (
            '<div class="hist-toggle">'
            '<button class="hist-btn active" id="hb7" onclick="histView(7)">1w</button>'
            '<button class="hist-btn" id="hb14" onclick="histView(14)">2w</button>'
            '</div>'
        )
        chart_html = (
            f'<div id="hist-d7">{chart_7}</div>'
            f'<div id="hist-d14" style="display:none">{chart_14}</div>'
            '<script>function histView(d){'
            'document.getElementById("hist-d7").style.display=d===7?"":"none";'
            'document.getElementById("hist-d14").style.display=d===14?"":"none";'
            'document.getElementById("hb7").classList.toggle("active",d===7);'
            'document.getElementById("hb14").classList.toggle("active",d===14);'
            '}</script>'
        )
        header_hint = ""
    else:
        toggle_html = ""
        chart_html  = chart_7 if len(rows_7) >= 2 else chart_14
        header_hint = '<span class="foldout-hint">(last 7d)</span>'

    return f"""
    <details class="foldout">
      <summary class="foldout-summary">📊 Prediction history {header_hint}</summary>
      <div class="foldout-body">
        <div class="hist-header">
          {stat_cards}
          {toggle_html}
        </div>
        {chart_html}
      </div>
    </details>"""
