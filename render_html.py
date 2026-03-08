"""
render_html.py — Generate a self-contained predictions page.

Reads predictions.json (and optionally config.toml) and writes index.html.

Usage:
    python render_html.py
    python render_html.py --predictions predictions.json --out index.html
"""

import argparse
import json
import math
import os
import tomllib
from collections import defaultdict
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


def dir_label(degrees: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[round(degrees / 45) % 8]


def _window_stats(window_wind: dict, cfg: dict) -> dict:
    """Compute observed stats from window_wind (local station data)."""
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


def _stats_html(headline: dict, cfg: dict) -> str:
    """
    Render compact observed + NWP stats rows for a day card.
    Returns '' when neither data source is available.
    """
    rows = []

    obs = _window_stats(headline.get("window_wind", {}), cfg)
    if obs:
        gust = f" / gust {obs['max_gust_kn']} kn" if obs["max_gust_kn"] is not None else ""
        dir_chip = f'<span class="stats-chip">dir ±{obs["dir_std_deg"]}°</span>' if obs["dir_std_deg"] is not None else ""
        rows.append(
            f'<div class="stats-row stats-observed">'
            f'<span class="stats-label">Observed</span>'
            f'<span class="stats-chip">avg {obs["mean_kn"]} kn</span>'
            f'<span class="stats-chip">max {obs["max_kn"]} kn{gust}</span>'
            f'<span class="stats-chip">{obs["pct_good"]}% in range</span>'
            f'{dir_chip}'
            f'</div>'
        )

    nwp = headline.get("nwp_forecast", {})
    if nwp:
        wind_chip = f'<span class="stats-chip">avg {nwp["mean_wind_kn"]} kn</span>' if nwp.get("mean_wind_kn") is not None else ""
        gust_chip = f'<span class="stats-chip">gust {nwp["max_gust_kn"]} kn</span>' if nwp.get("max_gust_kn") is not None else ""
        cloud_chip = f'<span class="stats-chip">cloud {nwp["cloud_cover_pct"]}%</span>' if nwp.get("cloud_cover_pct") is not None else ""
        blh_chip  = f'<span class="stats-chip">BLH {nwp["blh_m"]:,} m</span>' if nwp.get("blh_m") is not None else ""
        rows.append(
            f'<div class="stats-row stats-nwp">'
            f'<span class="stats-label">NWP</span>'
            f'{wind_chip}{gust_chip}{cloud_chip}{blh_chip}'
            f'</div>'
        )

    if not rows:
        return ""
    return '<div class="stats-block">' + "".join(rows) + "</div>\n"


def _prob_trend_svg(snaps: list[dict]) -> str:
    """
    Compact sparkline showing how the sailing probability evolved across
    successive forecast snapshots (earliest → latest).
    Returns '' when fewer than 2 snapshots exist.
    """
    sorted_snaps = sorted(snaps, key=lambda s: s["snapshot"])
    n = len(sorted_snaps)
    if n < 2:
        return ""

    VW, VH  = 360, 38
    PAD_L   = 52
    PAD_R   = 8
    PAD_T   = 4
    PAD_B   = 14
    cw      = VW - PAD_L - PAD_R
    ch      = VH - PAD_T - PAD_B

    times = [datetime.fromisoformat(s["snapshot"]) for s in sorted_snaps]
    span  = (times[-1] - times[0]).total_seconds() or 1.0
    probs = [s["probability"] for s in sorted_snaps]
    thr   = sorted_snaps[-1].get("threshold", 0.3)

    def tx(i):
        frac = (times[i] - times[0]).total_seconds() / span
        frac = max(0.02, min(0.98, frac)) if n > 2 else frac
        return PAD_L + frac * cw

    def ty(p):
        return PAD_T + ch * (1.0 - p)

    xs = [tx(i) for i in range(n)]
    ys = [ty(p) for p in probs]

    def dot_color(p):
        if p >= thr:        return "#16a34a"
        if p >= thr * 0.6:  return "#d97706"
        return "#dc2626"

    out = [
        f'<svg viewBox="0 0 {VW} {VH}" '
        f'style="width:100%;height:{VH}px;display:block;margin-bottom:.5rem" '
        f'class="dropout-svg" aria-hidden="true">'
    ]

    out.append(
        f'<text x="{PAD_L - 3}" y="{PAD_T + ch / 2 + 2.5:.1f}" '
        f'font-size="6.5" text-anchor="end" class="do-lbl">Forecast</text>'
    )
    # Threshold dashed line
    hy = ty(thr)
    out.append(
        f'<line x1="{PAD_L}" y1="{hy:.1f}" x2="{VW - PAD_R}" y2="{hy:.1f}" '
        f'stroke="#16a34a" stroke-width="0.75" stroke-dasharray="3,2" opacity="0.4"/>'
    )
    # Connecting line
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    out.append(
        f'<polyline points="{pts}" fill="none" stroke="#16a34a" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.7"/>'
    )
    # Dots + value labels
    for i, (x, y, prob) in enumerate(zip(xs, ys, probs)):
        c = dot_color(prob)
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
            f'fill="{c}" stroke="white" stroke-width="0.8"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{y - 4:.1f}" font-size="6" fill="{c}" '
            f'text-anchor="middle" font-weight="bold" font-family="sans-serif">'
            f'{round(prob * 100)}%</text>'
        )
    # Time-axis labels
    y_tick = VH - PAD_B
    for i, (x, t) in enumerate(zip(xs, times)):
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        out.append(
            f'<line x1="{x:.1f}" y1="{y_tick:.1f}" x2="{x:.1f}" '
            f'y2="{y_tick + 3:.1f}" stroke-width="0.75" class="do-track"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{VH - 1}" '
            f'font-size="6" text-anchor="{anchor}" class="do-lbl">'
            f'{t.strftime("%H:%M")}</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


def _wind_svg(window_wind: dict, cfg: dict) -> str:
    """
    Two-panel inline SVG:
      Left:  wind speed time series with rolling mean (window=5 readings ≈ 2.5 h)
             and 80 % confidence band + individual error bars.
      Right: wind direction compass rose (frequency by 45° sector).
    Returns '' when window data is unavailable.
    """
    if not window_wind:
        return ""

    times_str  = window_wind.get("times", [])
    speeds     = window_wind.get("speeds_kn", [])
    directions = window_wind.get("directions_deg", [])

    n = len(speeds)
    if n < 3:
        return ""

    sc       = cfg.get("sailing", {})
    wind_min = sc.get("wind_speed_min", 2.0)
    wind_max = sc.get("wind_speed_max", 10.0)

    # ── Rolling mean + 80 % CI (z = 1.282, centred window of 5 readings) ─────
    def _rolling(vals, window=5):
        half = window // 2
        ms, ss = [], []
        for i in range(len(vals)):
            chunk = vals[max(0, i - half): i + half + 1]
            m = sum(chunk) / len(chunk)
            ms.append(m)
            var = (sum((x - m) ** 2 for x in chunk) / max(len(chunk) - 1, 1)
                   if len(chunk) >= 2 else 0.0)
            ss.append(var ** 0.5)
        return ms, ss

    CI_Z     = 1.282   # 80 % CI
    CI_LABEL = "80 % CI"
    means, stds = _rolling(speeds)

    # ── Layout ────────────────────────────────────────────────────────────────
    VW, VH = 360, 155
    SPLIT  = 218          # left / right panel boundary

    LP_L, LP_R = 28, 6
    LP_T, LP_B = 16, 18   # LP_T: panel title; LP_B: time-axis labels
    cw = SPLIT - LP_L - LP_R
    ch = VH - LP_T - LP_B

    # Y scale: include CI headroom
    y_top_ws = max(
        max(m + CI_Z * s for m, s in zip(means, stds)) * 1.08,
        wind_max + 3,
    )

    def wx(i):
        return LP_L + (i / (n - 1)) * cw

    def wy(v):
        raw = LP_T + ch * (1.0 - v / y_top_ws)
        return max(LP_T, min(LP_T + ch, raw))   # clamp to chart area

    # Right panel (compass rose)
    RP_CX  = SPLIT + (VW - SPLIT) / 2
    RP_CY  = VH / 2 + 6
    ROSE_R = min((VW - SPLIT) / 2 - 22, (VH - 36) / 2)   # ≈ 44 px

    # ── Direction binning (8 × 45° sectors) ───────────────────────────────────
    sector_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    dir_counts    = [0] * 8
    for d in directions:
        dir_counts[int((d + 22.5) / 45) % 8] += 1
    max_count = max(dir_counts) if any(dir_counts) else 1

    # ── SVG ───────────────────────────────────────────────────────────────────
    p = [
        f'<svg viewBox="0 0 {VW} {VH}" '
        f'style="width:100%;height:{VH}px;display:block;margin-bottom:.75rem" '
        f'class="wind-svg" aria-hidden="true">'
    ]

    # ══ Left panel: wind speed ════════════════════════════════════════════════

    # Panel title
    p.append(
        f'<text x="{LP_L + cw / 2:.1f}" y="11" '
        f'font-size="7" text-anchor="middle" class="wv-lbl">Wind speed (kn)</text>'
    )

    # Good-range band (green fill)
    p.append(
        f'<rect x="{LP_L}" y="{wy(wind_max):.1f}" width="{cw}" '
        f'height="{wy(wind_min) - wy(wind_max):.1f}" fill="#16a34a" fill-opacity="0.10"/>'
    )

    # Horizontal grid lines + y-axis labels at 0, wind_min, wind_max
    for v in [0, wind_min, wind_max]:
        gy = wy(v)
        p.append(
            f'<line x1="{LP_L}" y1="{gy:.1f}" x2="{LP_L + cw}" y2="{gy:.1f}" '
            f'stroke-width="0.5" class="wv-grid"/>'
        )
        p.append(
            f'<text x="{LP_L - 3}" y="{gy + 2.5:.1f}" '
            f'font-size="6.5" text-anchor="end" class="wv-lbl">{round(v)}</text>'
        )

    # CI band: filled polygon (upper L→R then lower R→L)
    upper = [(wx(i), wy(m + CI_Z * s)) for i, (m, s) in enumerate(zip(means, stds))]
    lower = [(wx(i), wy(m - CI_Z * s)) for i, (m, s) in enumerate(zip(means, stds))]
    poly_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower[::-1])
    p.append(f'<polygon points="{poly_pts}" fill="#3b82f6" fill-opacity="0.13"/>')

    # Error bars at each data point
    for i, (m, s) in enumerate(zip(means, stds)):
        x  = wx(i)
        ub = wy(m + CI_Z * s)
        lb = wy(m - CI_Z * s)
        # Vertical stem
        p.append(
            f'<line x1="{x:.1f}" y1="{ub:.1f}" x2="{x:.1f}" y2="{lb:.1f}" '
            f'stroke="#3b82f6" stroke-width="1.1" opacity="0.4"/>'
        )
        # Caps
        for cap_y in (ub, lb):
            p.append(
                f'<line x1="{x - 3:.1f}" y1="{cap_y:.1f}" '
                f'x2="{x + 3:.1f}" y2="{cap_y:.1f}" '
                f'stroke="#3b82f6" stroke-width="1.1" opacity="0.5"/>'
            )

    # Rolling mean line (drawn on top of CI band)
    mean_pts = " ".join(f"{wx(i):.1f},{wy(m):.1f}" for i, m in enumerate(means))
    p.append(
        f'<polyline points="{mean_pts}" fill="none" stroke="#3b82f6" '
        f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>'
    )

    # Raw measurement dots (coloured by good-range membership)
    for i, s in enumerate(speeds):
        c = "#16a34a" if wind_min <= s <= wind_max else "#94a3b8"
        p.append(
            f'<circle cx="{wx(i):.1f}" cy="{wy(s):.1f}" r="3" '
            f'fill="{c}" stroke="white" stroke-width="0.8" opacity="0.95"/>'
        )

    # CI label (top-right of chart area)
    p.append(
        f'<text x="{LP_L + cw - 2}" y="{LP_T + 9}" font-size="6" '
        f'text-anchor="end" fill="#3b82f6" font-family="sans-serif" '
        f'opacity="0.65">{CI_LABEL}</text>'
    )

    # X-axis time labels (up to 5 evenly spaced)
    step = max(1, (n - 1) // 4)
    shown = list(range(0, n, step))
    if (n - 1) not in shown:
        shown.append(n - 1)
    for i in shown:
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        p.append(
            f'<text x="{wx(i):.1f}" y="{VH - 2}" '
            f'font-size="6.5" text-anchor="{anchor}" class="wv-lbl">{times_str[i]}</text>'
        )

    # ── Panel divider ─────────────────────────────────────────────────────────
    p.append(
        f'<line x1="{SPLIT - 1}" y1="6" x2="{SPLIT - 1}" y2="{VH - 6}" '
        f'stroke-width="0.5" class="wv-grid"/>'
    )

    # ══ Right panel: compass rose ═════════════════════════════════════════════

    # Panel title
    p.append(
        f'<text x="{RP_CX:.1f}" y="11" '
        f'font-size="7" text-anchor="middle" class="wv-lbl">Wind direction</text>'
    )

    # Background reference circle
    p.append(
        f'<circle cx="{RP_CX:.1f}" cy="{RP_CY:.1f}" r="{ROSE_R:.1f}" '
        f'fill="none" stroke-width="0.6" class="wv-grid"/>'
    )

    # Filled pie-wedge sectors
    for i, count in enumerate(dir_counts):
        if count == 0:
            continue
        r       = ROSE_R * (count / max_count)
        svg_s   = i * 45 - 22.5 - 90
        svg_e   = i * 45 + 22.5 - 90
        sr, er  = math.radians(svg_s), math.radians(svg_e)
        x1 = RP_CX + r * math.cos(sr);  y1 = RP_CY + r * math.sin(sr)
        x2 = RP_CX + r * math.cos(er);  y2 = RP_CY + r * math.sin(er)
        opacity = 0.30 + 0.60 * (count / max_count)
        p.append(
            f'<path d="M {RP_CX:.1f},{RP_CY:.1f} '
            f'L {x1:.1f},{y1:.1f} A {r:.1f},{r:.1f} 0 0,1 {x2:.1f},{y2:.1f} Z" '
            f'fill="#3b82f6" fill-opacity="{opacity:.2f}" '
            f'stroke="white" stroke-width="0.7"/>'
        )

    # Compass direction labels (cardinal in bold/larger)
    for i, lbl in enumerate(sector_labels):
        ar       = math.radians(i * 45 - 90)
        lr       = ROSE_R + 11
        lx       = RP_CX + lr * math.cos(ar)
        ly       = RP_CY + lr * math.sin(ar) + 2.5
        cardinal = (i % 2 == 0)
        p.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" '
            f'font-size="{"7.5" if cardinal else "6"}" text-anchor="middle" '
            f'font-weight="{"bold" if cardinal else "normal"}" class="wv-lbl">{lbl}</text>'
        )

    # Centre dot
    p.append(
        f'<circle cx="{RP_CX:.1f}" cy="{RP_CY:.1f}" r="2.5" fill="#64748b" opacity="0.5"/>'
    )

    p.append("</svg>")
    return "\n".join(p)


def _history_html(db_path: str, days: int = 60) -> str:
    """
    Render a prediction accuracy history section from predictions.db.
    Returns an empty string if the DB does not exist or has no evaluated days.

    Each day is shown as a coloured square:
      ■ green  = predicted good  AND was good  (true positive)
      ■ red    = predicted poor  AND was poor  (true negative)
      □ orange = predicted good  BUT was poor  (false positive)
      □ blue   = predicted poor  BUT was good  (false negative)
      · grey   = prediction exists but outcome not yet known (future/today)
    """
    if not os.path.exists(db_path):
        return ""

    try:
        from model.history import load_history, accuracy_summary
    except ImportError:
        return ""

    df = load_history(db_path=db_path, days=days)
    if df.empty:
        return ""

    # One row per predicting_date: use the last snapshot of each day
    df = df.sort_values("snapshot_dt").groupby("predicting_date").last().reset_index()
    df = df.sort_values("predicting_date")

    summary = accuracy_summary(db_path=db_path, days=days)

    # Build the dot grid (max ~60 squares, newest last)
    dots = ""
    for _, row in df.iterrows():
        pred_good = int(row["good"])
        has_outcome = row["actual_good"] is not None and row["actual_good"] == row["actual_good"]
        date_label = row["predicting_date"]
        prob_pct = round(float(row["probability"]) * 100)

        if not has_outcome:
            # No ground truth yet
            color = "#94a3b8"
            symbol = "·"
            title = f"{date_label}: predicted {'good' if pred_good else 'poor'} ({prob_pct}%) — outcome pending"
            dot_style = f"color:{color};font-size:1.3rem;"
        else:
            actual_good = int(row["actual_good"])
            correct = pred_good == actual_good
            if pred_good and actual_good:
                color, title_tag = "#16a34a", "TP"
            elif not pred_good and not actual_good:
                color, title_tag = "#16a34a", "TN"
            elif pred_good and not actual_good:
                color, title_tag = "#d97706", "FP"
            else:
                color, title_tag = "#2563eb", "FN"
            label = "good" if actual_good else "poor"
            title = f"{date_label}: predicted {'good' if pred_good else 'poor'} ({prob_pct}%), actual {label} [{title_tag}]"
            symbol = "■" if correct else "□"
            dot_style = f"color:{color};font-size:1.1rem;"

        dots += f'<span style="{dot_style}" title="{title}">{symbol}</span>'

    if not dots:
        return ""

    # Accuracy stats line
    stats_line = ""
    if summary:
        n   = summary["n_evaluated"]
        acc = round(summary["accuracy"] * 100)
        parts = [f"<strong>{acc}%</strong> accuracy over {n} evaluated days"]
        if summary.get("precision") is not None:
            parts.append(f"precision {round(summary['precision']*100)}%")
        if summary.get("recall") is not None:
            parts.append(f"recall {round(summary['recall']*100)}%")
        stats_line = " · ".join(parts)

    legend = (
        '<span style="color:#16a34a">■</span> correct &nbsp;'
        '<span style="color:#d97706">□</span> false alarm &nbsp;'
        '<span style="color:#2563eb">□</span> missed &nbsp;'
        '<span style="color:#94a3b8">·</span> pending'
    )

    return f"""
    <section class="history-section">
      <h3 class="history-title">Prediction history <span class="history-days">({days}d)</span></h3>
      <div class="history-dots">{dots}</div>
      <div class="history-legend">{legend}</div>
      {"<div class='history-stats'>" + stats_line + "</div>" if stats_line else ""}
    </section>"""


def build_html(predictions: list[dict], cfg: dict, db_path: str | None = None) -> str:
    sailing = cfg.get("sailing", {})
    window_start = sailing.get("window_start", "08:00")
    window_end   = sailing.get("window_end",   "16:00")
    wind_min     = sailing.get("wind_speed_min", 2)
    wind_max     = sailing.get("wind_speed_max", 10)
    threshold    = cfg.get("prediction", {}).get("min_good_fraction", 0.3)

    # Group by predicting_date, keep all snapshots
    by_date: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        by_date[p["predicting_date"]].append(p)

    # Sort dates ascending
    sorted_dates = sorted(by_date.keys())

    # Build day cards HTML
    cards_html = ""
    for date_str in sorted_dates:
        snaps = sorted(by_date[date_str], key=lambda x: x["snapshot"])
        # Best (latest) prediction drives the headline
        headline = snaps[-1]
        prob = headline["probability"]
        good = headline["good"]

        # Formatted date
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_label = dt.strftime("%A, %-d %B %Y")

        pct = round(prob * 100)
        status_class = "good" if good else "poor"
        status_text  = "Good day" if good else "Poor day"
        status_icon  = "⛵" if good else "🌧"

        # Bar color
        if pct >= int(threshold * 100):
            bar_color = "var(--c-good)"
        elif pct >= int(threshold * 100 * 0.6):
            bar_color = "var(--c-warn)"
        else:
            bar_color = "var(--c-poor)"

        # Snapshot rows
        snap_rows = ""
        for s in snaps:
            snap_dt = datetime.fromisoformat(s["snapshot"])
            snap_time = snap_dt.strftime("%H:%M")
            s_pct = round(s["probability"] * 100)
            s_good = s["good"]
            s_dot = "●" if s_good else "○"
            s_class = "snap-good" if s_good else "snap-poor"
            snap_rows += f"""
            <tr class="snap-row {s_class}">
              <td class="snap-time">Snapshot {snap_time}</td>
              <td class="snap-dot">{s_dot}</td>
              <td class="snap-prob">{s_pct}%</td>
              <td class="snap-bar-cell">
                <div class="snap-bar" style="width:{s_pct}%"></div>
              </td>
            </tr>"""

        cards_html += f"""
    <article class="day-card {status_class}">
      <header class="card-header">
        <div class="card-title">
          <span class="card-icon">{status_icon}</span>
          <h2>{day_label}</h2>
        </div>
        <div class="card-badge {status_class}">{status_text}</div>
      </header>

      <div class="card-body">
        <div class="prob-section">
          <span class="prob-label">Sailing probability</span>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{pct}%; background:{bar_color}"></div>
            <div class="threshold-mark" style="left:{int(threshold*100)}%"
                 title="Threshold {int(threshold*100)}%"></div>
          </div>
          <span class="prob-value">{pct}%</span>
        </div>

        <div class="meta-row">
          <span class="meta-chip">⏱ {window_start}–{window_end}</span>
          <span class="meta-chip">💨 {wind_min}–{wind_max} kn</span>
          <span class="meta-chip">Threshold {int(threshold*100)}%</span>
        </div>

        {_wind_svg(headline.get("window_wind", {}), cfg)}

        {_stats_html(headline, cfg)}

        {_prob_trend_svg(snaps)}

        <details class="snapshots">
          <summary>All snapshots ({len(snaps)})</summary>
          <table class="snap-table">
            <tbody>{snap_rows}
            </tbody>
          </table>
        </details>
      </div>
    </article>"""

    generated = datetime.now().strftime("%-d %B %Y, %H:%M")

    # Resolve db_path relative to this file if not provided
    if db_path is None:
        db_path = os.path.join(_HERE, "predictions.db")
    history_section = _history_html(db_path)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Wind Predictor</title>
  <style>
    /* ── Tokens ── */
    :root {{
      --c-bg:       #f7f9fb;
      --c-surface:  #ffffff;
      --c-border:   #e2e8f0;
      --c-text:     #1e293b;
      --c-muted:    #64748b;
      --c-good:     #16a34a;
      --c-good-bg:  #f0fdf4;
      --c-good-bd:  #bbf7d0;
      --c-warn:     #d97706;
      --c-warn-bg:  #fffbeb;
      --c-poor:     #dc2626;
      --c-poor-bg:  #fef2f2;
      --c-poor-bd:  #fecaca;
      --c-track:    #e2e8f0;
      --radius:     12px;
      --shadow:     0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.04);
    }}

    @media (prefers-color-scheme: dark) {{
      :root {{
        --c-bg:       #0f172a;
        --c-surface:  #1e293b;
        --c-border:   #334155;
        --c-text:     #e2e8f0;
        --c-muted:    #94a3b8;
        --c-good-bg:  #052e16;
        --c-good-bd:  #166534;
        --c-warn-bg:  #1c1209;
        --c-poor-bg:  #1c0505;
        --c-poor-bd:  #7f1d1d;
        --c-track:    #334155;
        --shadow:     0 1px 3px rgba(0,0,0,.4);
      }}
    }}

    /* ── Reset ── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--c-bg);
      color: var(--c-text);
      line-height: 1.5;
      padding: 2rem 1rem 4rem;
    }}

    /* ── Layout ── */
    .page-wrap {{ max-width: 680px; margin: 0 auto; }}

    /* ── Page header ── */
    .page-header {{ margin-bottom: 2rem; }}
    .page-header h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .page-header .subtitle {{
      color: var(--c-muted);
      font-size: 0.875rem;
      margin-top: 0.25rem;
    }}

    /* ── Day card ── */
    .day-card {{
      background: var(--c-surface);
      border: 1px solid var(--c-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      margin-bottom: 1.25rem;
      overflow: hidden;
    }}
    .day-card.good {{ border-left: 4px solid var(--c-good); }}
    .day-card.poor {{ border-left: 4px solid var(--c-poor); }}

    /* card header */
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem 1.25rem 0.75rem;
      gap: 0.75rem;
    }}
    .card-title {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .card-icon {{ font-size: 1.4rem; line-height: 1; }}
    .card-title h2 {{
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: -0.01em;
    }}

    /* badge */
    .card-badge {{
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.65rem;
      border-radius: 999px;
      white-space: nowrap;
    }}
    .card-badge.good {{
      background: var(--c-good-bg);
      color: var(--c-good);
      border: 1px solid var(--c-good-bd);
    }}
    .card-badge.poor {{
      background: var(--c-poor-bg);
      color: var(--c-poor);
      border: 1px solid var(--c-poor-bd);
    }}

    /* card body */
    .card-body {{ padding: 0 1.25rem 1.25rem; }}

    /* probability bar */
    .prob-section {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.85rem;
    }}
    .prob-label {{
      font-size: 0.78rem;
      color: var(--c-muted);
      white-space: nowrap;
      min-width: 130px;
    }}
    .prob-bar-track {{
      flex: 1;
      height: 8px;
      background: var(--c-track);
      border-radius: 999px;
      position: relative;
      overflow: visible;
    }}
    .prob-bar-fill {{
      height: 100%;
      border-radius: 999px;
      transition: width 0.4s ease;
    }}
    .threshold-mark {{
      position: absolute;
      top: -4px;
      width: 2px;
      height: 16px;
      background: var(--c-muted);
      border-radius: 1px;
      transform: translateX(-50%);
    }}
    .prob-value {{
      font-size: 0.875rem;
      font-weight: 700;
      min-width: 36px;
      text-align: right;
    }}

    /* meta chips */
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-bottom: 0.85rem;
    }}
    .meta-chip {{
      font-size: 0.72rem;
      color: var(--c-muted);
      background: var(--c-bg);
      border: 1px solid var(--c-border);
      border-radius: 999px;
      padding: 0.15rem 0.6rem;
    }}

    /* ── Window stats ── */
    .stats-block {{
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
      margin-bottom: 0.85rem;
    }}
    .stats-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.35rem;
    }}
    .stats-label {{
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      min-width: 60px;
      color: var(--c-muted);
    }}
    .stats-chip {{
      font-size: 0.72rem;
      color: var(--c-text);
      background: var(--c-bg);
      border: 1px solid var(--c-border);
      border-radius: 999px;
      padding: 0.1rem 0.5rem;
    }}
    .stats-observed .stats-label {{ color: var(--c-good); }}
    .stats-nwp      .stats-label {{ color: #7c3aed; }}

    /* snapshots collapsible */
    .snapshots summary {{
      cursor: pointer;
      font-size: 0.8rem;
      color: var(--c-muted);
      user-select: none;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }}
    .snapshots summary::before {{
      content: "▸";
      font-size: 0.7rem;
      transition: transform 0.2s;
    }}
    .snapshots[open] summary::before {{ transform: rotate(90deg); }}
    .snapshots summary::-webkit-details-marker {{ display: none; }}

    .snap-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.6rem;
      font-size: 0.8rem;
    }}
    .snap-row {{ border-top: 1px solid var(--c-border); }}
    .snap-row td {{ padding: 0.35rem 0.4rem; }}
    .snap-time  {{ color: var(--c-muted); width: 110px; }}
    .snap-dot   {{ width: 20px; text-align: center; }}
    .snap-good .snap-dot {{ color: var(--c-good); }}
    .snap-poor .snap-dot {{ color: var(--c-poor); }}
    .snap-prob  {{ width: 38px; text-align: right; font-weight: 600; }}
    .snap-bar-cell {{ }}
    .snap-bar {{
      height: 5px;
      background: var(--c-muted);
      border-radius: 999px;
      opacity: 0.5;
    }}
    .snap-good .snap-bar {{ background: var(--c-good); opacity: 0.7; }}
    .snap-poor .snap-bar {{ background: var(--c-poor); opacity: 0.5; }}

    /* ── Wind distribution charts ── */
    .wind-svg .wv-lbl {{
      fill: var(--c-muted);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .wind-svg .wv-grid {{ stroke: var(--c-border); }}

    /* ── Forecast probability trend sparkline ── */
    .dropout-svg .do-lbl {{
      fill: var(--c-muted);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .dropout-svg .do-track {{ stroke: var(--c-border); }}

    /* ── History section ── */
    .history-section {{
      background: var(--c-surface);
      border: 1px solid var(--c-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 1rem 1.25rem;
      margin-bottom: 1.25rem;
    }}
    .history-title {{
      font-size: 0.875rem;
      font-weight: 600;
      margin-bottom: 0.6rem;
    }}
    .history-days {{
      font-weight: 400;
      color: var(--c-muted);
      font-size: 0.8rem;
    }}
    .history-dots {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.15rem;
      margin-bottom: 0.5rem;
      line-height: 1;
      letter-spacing: 0.05em;
    }}
    .history-legend {{
      font-size: 0.72rem;
      color: var(--c-muted);
      margin-bottom: 0.4rem;
    }}
    .history-stats {{
      font-size: 0.78rem;
      color: var(--c-text);
    }}

    /* ── Footer ── */
    .page-footer {{
      margin-top: 2.5rem;
      text-align: center;
      font-size: 0.75rem;
      color: var(--c-muted);
    }}
  </style>
</head>
<body>
  <div class="page-wrap">
    <header class="page-header">
      <h1>⛵ Wind Predictor</h1>
      <p class="subtitle">Sailing conditions forecast · Generated {generated}</p>
    </header>

    {history_section}

    {cards_html}

    <footer class="page-footer">
      <p>Sailing window {window_start}–{window_end} · Wind {wind_min}–{wind_max} kn · Good-day threshold {int(threshold*100)}%</p>
    </footer>
  </div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render predictions.json → index.html")
    parser.add_argument("--predictions", default=os.path.join(_HERE, "predictions.json"))
    parser.add_argument("--config",      default=os.path.join(_HERE, "config.toml"))
    parser.add_argument("--out",         default=os.path.join(_HERE, "index.html"))
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    cfg = load_config(args.config)
    html = build_html(predictions, cfg)

    with open(args.out, "w") as f:
        f.write(html)

    print(f"Written: {args.out}  ({len(predictions)} predictions, "
          f"{len(set(p['predicting_date'] for p in predictions))} days)")


if __name__ == "__main__":
    main()
