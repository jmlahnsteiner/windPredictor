"""
render_html.py — Generate a self-contained predictions page.

Reads forecast snapshots from DB (or optionally a JSON file) and writes index.html.

Usage:
    python render_html.py
    python render_html.py --predictions predictions.json --out index.html
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))

from utils.config import load_config
from utils.db import DEFAULT_SQLITE
from render.charts import prob_trend_svg, wind_svg
from render.data import score_to_hex, window_stats, expected_wind_chips, stats_html


def dir_label(degrees: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[round(degrees / 45) % 8]


# ── Condition-scale colour palette ──────────────────────────────────────────
# (text-colour, background, border) keyed by condition label
_COND_BADGE: dict[str, tuple[str, str, str]] = {
    "No wind":        ("#475569", "#f1f5f9", "#cbd5e1"),
    "Very light":     ("#0369a1", "#e0f2fe", "#bae6fd"),
    "Marginal":       ("#92400e", "#fffbeb", "#fde68a"),
    "Fair":           ("#3d6b1a", "#f0fdf4", "#bbf7d0"),
    "Good":           ("#15803d", "#dcfce7", "#86efac"),
    "Great":          ("#047857", "#d1fae5", "#6ee7b7"),
    "Excellent":      ("#0e7490", "#ecfeff", "#a5f3fc"),
    "Strong / gusty": ("#9a3412", "#fff7ed", "#fed7aa"),
    "Storm":          ("#dc2626", "#fef2f2", "#fecaca"),
}

def _badge_style(label: str) -> str:
    """Inline CSS for a condition-label badge."""
    txt, bg, bd = _COND_BADGE.get(label, ("#475569", "#f8fafc", "#e2e8f0"))
    return f"color:{txt};background:{bg};border:1px solid {bd}"


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
        if "predicting_date" not in p:
            continue
        by_date[p["predicting_date"]].append(p)

    # Classify each date as past (sailing window closed) or active/future.
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    we_h, we_m = map(int, window_end.split(":"))
    window_closed_today = (now.hour, now.minute) >= (we_h, we_m)

    def _is_past(date_str: str) -> bool:
        if date_str < today_str:
            return True
        if date_str == today_str and window_closed_today:
            return True
        return False

    active_dates = sorted(d for d in by_date if not _is_past(d))
    past_dates   = sorted((d for d in by_date if _is_past(d)), reverse=True)
    sorted_dates = active_dates + past_dates   # active first, newest-past last

    # Build day cards HTML — active cards shown directly, past cards in one fold-out
    active_cards_html = ""
    past_cards_html   = ""
    for date_str in sorted_dates:
        snaps = sorted(by_date[date_str], key=lambda x: x["snapshot"])
        # Best (latest) prediction drives the headline
        headline = snaps[-1]
        prob = headline["probability"]
        good = headline["good"]
        is_past_day = _is_past(date_str)

        # Formatted date
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_label = dt.strftime("%A, %-d %B %Y")

        pct          = round(prob * 100)
        c_score      = headline.get("condition_score",  pct)
        c_label      = headline.get("condition_label",  "Good" if good else "Poor")
        c_icon       = headline.get("condition_icon",   "⛵" if good else "🌫")
        c_source     = headline.get("condition_source", "forecast")
        is_extended  = headline.get("is_extended_forecast", False)
        lead_days    = headline.get("lead_days", None)

        if is_past_day:
            status_class   = "past"
            badge_label    = f"Past · {c_label}"
            badge_style    = _badge_style(c_label)
        else:
            status_class   = "good" if good else "poor"
            badge_label    = c_label
            badge_style    = _badge_style(c_label)

        # Condition score gradient bar
        # The track shows a fixed gradient; a mask covers the unfilled portion.
        score_color     = score_to_hex(c_score)
        bar_label       = "Observed conditions" if c_source == "observed" else "Sailing outlook"
        # Zone marker at the "good" threshold on the gradient bar (score ≥ 60 = Good)
        good_zone_left  = 60

        # Snapshot rows (condition label + probability)
        snap_rows = ""
        for s in snaps:
            snap_dt    = datetime.fromisoformat(s["snapshot"])
            snap_time  = snap_dt.strftime("%H:%M")
            s_score    = s.get("condition_score",  round(s["probability"] * 100))
            s_label    = s.get("condition_label",  "Good" if s["good"] else "Poor")
            s_icon     = s.get("condition_icon",   "⛵" if s["good"] else "🌫")
            s_src      = s.get("condition_source", "forecast")
            s_color    = score_to_hex(s_score)
            s_pct      = round(s["probability"] * 100)
            snap_rows += f"""
            <tr class="snap-row">
              <td class="snap-time">Snapshot {snap_time}</td>
              <td class="snap-dot">{s_icon}</td>
              <td class="snap-label" style="color:{s_color}">{s_label}</td>
              <td class="snap-score" style="color:{s_color}">{s_score}</td>
              <td class="snap-bar-cell">
                <div class="snap-bar" style="width:{s_score}%;background:{s_color}"></div>
              </td>
            </tr>"""

        # Expected wind chips (avg wind, gusts, consistency) from observed or NWP
        exp_chips = expected_wind_chips(headline, cfg)
        # Extended forecast note (e.g. "+3 d" shown next to p value)
        lead_note = f" +{lead_days}d" if is_extended and lead_days is not None else ""

        card_body = f"""
      <div class="card-body">
        <div class="cond-section">
          <span class="cond-bar-label">{bar_label}</span>
          <div class="cond-bar-wrap">
            <div class="cond-bar-track">
              <div class="cond-bar-mask" style="left:{c_score}%"></div>
              <div class="cond-zone-mark" style="left:{good_zone_left}%"
                   title="Good threshold (score 60)"></div>
            </div>
            <div class="cond-zone-labels">
              <span>No wind</span><span>Marginal</span><span>Good</span><span>Excellent</span>
            </div>
          </div>
          <span class="cond-score-value" style="color:{score_color}">{c_score}</span>
        </div>

        <div class="meta-row">
          {exp_chips}
          <span class="meta-chip">p={pct}%{lead_note}</span>
        </div>

        {wind_svg(headline.get("window_wind", {}), cfg)}

        {stats_html(headline, cfg)}

        {prob_trend_svg(snaps)}
      </div>"""

        # Card border colour follows condition score for active days
        if is_past_day:
            border_color = "var(--c-border)"
        else:
            border_color = score_to_hex(c_score)

        if is_past_day:
            past_cards_html += f"""
    <details class="past-card-wrap">
      <summary class="past-card-summary">
        <article class="day-card past" style="margin-bottom:0;border-left-color:{border_color}">
          <header class="card-header">
            <div class="card-title">
              <span class="card-icon">{c_icon}</span>
              <h2>{day_label}</h2>
            </div>
            <div class="card-badge" style="{badge_style}">{badge_label}</div>
          </header>
        </article>
      </summary>
      <article class="day-card past past-detail" style="border-left-color:{border_color}">
        {card_body}
      </article>
    </details>"""
        else:
            active_cards_html += f"""
    <article class="day-card {status_class}" style="border-left-color:{border_color}">
      <header class="card-header">
        <div class="card-title">
          <span class="card-icon">{c_icon}</span>
          <h2>{day_label}</h2>
        </div>
        <div class="card-badge" style="{badge_style}">{badge_label}</div>
      </header>
      {card_body}
    </article>"""

    # Wrap all past-day cards in a single collapsible section
    n_past = len(past_dates)
    if past_cards_html:
        past_label = f"Past {n_past} day{'s' if n_past != 1 else ''}"
        past_section = f"""
    <details class="past-week-wrap">
      <summary class="past-week-summary">{past_label}</summary>
      <div class="past-week-inner">
        {past_cards_html}
      </div>
    </details>"""
    else:
        past_section = ""

    cards_html = active_cards_html + past_section

    generated = datetime.now().strftime("%-d %B %Y, %H:%M")

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
    .day-card.past {{ border-left: 4px solid var(--c-border); opacity: 0.72; }}

    /* Past-week outer fold-out */
    .past-week-wrap {{ margin-bottom: 1.25rem; }}
    .past-week-summary {{
      cursor: pointer;
      list-style: none;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--c-muted);
      padding: 0.6rem 0;
      user-select: none;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .past-week-summary::-webkit-details-marker {{ display: none; }}
    .past-week-summary::before {{
      content: "▸";
      font-size: 0.75rem;
      transition: transform 0.2s;
    }}
    .past-week-wrap[open] .past-week-summary::before {{ transform: rotate(90deg); }}
    .past-week-inner {{ padding-left: 0; }}

    /* Past-day collapsible wrapper */
    .past-card-wrap {{ margin-bottom: 1.25rem; }}
    .past-card-wrap[open] .past-card-summary article {{ border-radius: var(--radius) var(--radius) 0 0; margin-bottom: 0; }}
    .past-card-summary {{ list-style: none; cursor: pointer; }}
    .past-card-summary::-webkit-details-marker {{ display: none; }}
    .past-card-summary article {{ margin-bottom: 0; }}
    .past-card-summary .card-header::after {{
      content: "▸";
      font-size: 0.8rem;
      color: var(--c-muted);
      margin-left: auto;
      transition: transform 0.2s;
    }}
    .past-card-wrap[open] .past-card-summary .card-header::after {{ transform: rotate(90deg); }}
    .past-detail {{ border-radius: 0 0 var(--radius) var(--radius); border-top: none; }}

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

    /* card body */
    .card-body {{ padding: 0 1.25rem 1.25rem; }}

    /* ── Condition score bar ── */
    .cond-section {{
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      margin-bottom: 0.85rem;
    }}
    .cond-bar-label {{
      font-size: 0.78rem;
      color: var(--c-muted);
      white-space: nowrap;
      min-width: 130px;
      padding-top: 0.1rem;
    }}
    .cond-bar-wrap {{ flex: 1; }}
    .cond-bar-track {{
      height: 10px;
      border-radius: 999px;
      /* Gradient: gray → amber → green → emerald → cyan */
      background: linear-gradient(to right,
        #94a3b8  0%,
        #fbbf24 28%,
        #22c55e 58%,
        #10b981 78%,
        #06b6d4 100%
      );
      position: relative;
      overflow: visible;
    }}
    /* Mask covers the unfilled (right) portion with the page track colour */
    .cond-bar-mask {{
      position: absolute;
      top: 0; right: 0; bottom: 0;
      background: var(--c-track);
      border-radius: 0 999px 999px 0;
    }}
    .cond-zone-mark {{
      position: absolute;
      top: -4px;
      width: 2px;
      height: 18px;
      background: var(--c-muted);
      opacity: 0.5;
      border-radius: 1px;
      transform: translateX(-50%);
    }}
    .cond-zone-labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 0.2rem;
      font-size: 0.6rem;
      color: var(--c-muted);
    }}
    .cond-score-value {{
      font-size: 1rem;
      font-weight: 700;
      min-width: 36px;
      text-align: right;
      padding-top: 0.05rem;
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
    /* Expected-condition chips carry slightly stronger colour to distinguish
       from secondary info chips like the model probability */
    .exp-chip {{
      color: var(--c-text);
      font-weight: 500;
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
    .snap-row td {{ padding: 0.3rem 0.4rem; vertical-align: middle; }}
    .snap-time  {{ color: var(--c-muted); width: 100px; }}
    .snap-dot   {{ width: 22px; text-align: center; font-size: 1rem; line-height: 1; }}
    .snap-label {{ font-weight: 600; white-space: nowrap; }}
    .snap-score {{ width: 32px; text-align: right; font-weight: 700; font-size: 0.85rem; }}
    .snap-bar-cell {{ }}
    .snap-bar {{
      height: 5px;
      border-radius: 999px;
      opacity: 0.75;
    }}

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
    .hist-dot {{
      display: inline-block;
      width: 14px;
      height: 14px;
      border-radius: 3px;
      vertical-align: middle;
      cursor: default;
    }}
    .hist-pending {{
      background: none;
      border: none;
      color: #94a3b8;
      font-size: 1.2rem;
      line-height: 1;
      width: auto;
      height: auto;
    }}
    .history-dots {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.2rem;
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

    {cards_html}

    <footer class="page-footer">
      <p>Sailing window {window_start}–{window_end} · Wind {wind_min}–{wind_max} kn · Good-day threshold {int(threshold*100)}%</p>
    </footer>
  </div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render forecast snapshots → index.html")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional JSON file override (default: read from DB)",
    )
    parser.add_argument("--config", default=os.path.join(_HERE, "config.toml"))
    parser.add_argument("--out",    default=os.path.join(_HERE, "index.html"))
    args = parser.parse_args()

    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
    else:
        from model.predict import load_forecast_snapshots
        predictions = load_forecast_snapshots()

    cfg = load_config(args.config)
    html = build_html(predictions, cfg)

    with open(args.out, "w") as f:
        f.write(html)

    print(f"Written: {args.out}  ({len(predictions)} predictions, "
          f"{len(set(p['predicting_date'] for p in predictions if 'predicting_date' in p))} days)")


if __name__ == "__main__":
    main()
