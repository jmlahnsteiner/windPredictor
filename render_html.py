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
from render.charts import wind_svg, history_chart_svg  # noqa: F401 (history_chart_svg used via history_html)
from render.data import score_to_hex, window_stats, expected_wind_chips, stats_html, history_html


def _methodology_html(cfg: dict) -> str:
    """Static foldout explaining the numbers and methodology."""
    sailing = cfg.get("sailing", {})
    ws   = sailing.get("window_start", "08:00")
    we   = sailing.get("window_end", "16:00")
    wmin = sailing.get("wind_speed_min", 2)
    wmax = sailing.get("wind_speed_max", 10)
    thr  = cfg.get("prediction", {}).get("min_good_fraction", 0.3)
    return f"""
    <details class="foldout">
      <summary class="foldout-summary">ℹ️ About this forecast</summary>
      <div class="foldout-body">
        <dl class="about-dl">
          <div class="about-row">
            <dt>Condition score (0–100)</dt>
            <dd>Fraction of hourly readings within the sailing window that fall in the target wind range ({wmin}–{wmax} kn) with consistent direction. The gradient bar maps score to quality colour.</dd>
          </div>
          <div class="about-row">
            <dt>Probability (p=X%)</dt>
            <dd>Random Forest model confidence that the sailing window ({ws}–{we}) will have ≥{int(thr * 100)}% of hours with good conditions. Days above this threshold are classified as "good".</dd>
          </div>
          <div class="about-row">
            <dt>Extended forecasts (+1d, +2d)</dt>
            <dd>Days beyond the direct ML target are scaled by ×0.82 and ×0.64 to reflect increasing uncertainty. Display-only — not included in the accuracy history.</dd>
          </div>
          <div class="about-row">
            <dt>Data sources</dt>
            <dd>Observed wind from a local Ecowitt weather station. Wind, gust, cloud cover, and boundary-layer height forecasts from Open-Meteo NWP (no API key required).</dd>
          </div>
          <div class="about-row">
            <dt>History chart</dt>
            <dd>Cyan dashed = model probability each day. Green solid = actual fraction of the sailing window with good wind (recorded after the day passes). The dashed threshold line marks the {int(thr * 100)}% good-day cutoff.</dd>
          </div>
        </dl>
      </div>
    </details>"""


def build_html(
    predictions: list[dict],
    cfg: dict,
    db_path: str | None = None,
) -> str:
    sailing      = cfg.get("sailing", {})
    window_start = sailing.get("window_start", "08:00")
    window_end   = sailing.get("window_end",   "16:00")
    wind_min     = sailing.get("wind_speed_min", 2)
    wind_max     = sailing.get("wind_speed_max", 10)
    threshold    = cfg.get("prediction", {}).get("min_good_fraction", 0.3)

    # Group snapshots by predicting_date
    by_date: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        if "predicting_date" not in p:
            continue
        by_date[p["predicting_date"]].append(p)

    # Classify dates
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

    def _card_data(date_str: str) -> dict:
        snaps   = sorted(by_date[date_str], key=lambda x: x["snapshot"])
        h       = snaps[-1]
        prob    = h["probability"]
        pct     = round(prob * 100)
        c_score = h.get("condition_score", pct)
        c_label = h.get("condition_label", "Good" if h["good"] else "Poor")
        c_icon  = h.get("condition_icon",  "⛵" if h["good"] else "🌫")
        is_ext  = h.get("is_extended_forecast", False)
        lead    = h.get("lead_days", None)
        lead_note = f" +{lead}d" if is_ext and lead is not None else ""
        return dict(
            snaps=snaps, headline=h, pct=pct,
            c_score=c_score, c_label=c_label, c_icon=c_icon,
            lead_note=lead_note,
            score_color=score_to_hex(c_score),
        )

    GOOD_ZONE = 60   # score threshold for "good" zone marker position

    # ── Hero card ─────────────────────────────────────────────────────────────
    hero_date = active_dates[0] if active_dates else (past_dates[0] if past_dates else None)

    if hero_date is None:
        cards_html = '<p class="no-data">No forecast data available.</p>'
    else:
        d = _card_data(hero_date)
        dt = datetime.strptime(hero_date, "%Y-%m-%d")
        day_label = dt.strftime("%A, %-d %B %Y").upper()

        exp_chips = expected_wind_chips(d["headline"], cfg)

        hero_html = f"""
    <div class="hero-card">
      <div class="hero-top">
        <div class="hero-left">
          <div class="hero-date">{day_label}</div>
          <span class="hero-pct" style="color:{d['score_color']}">{d['pct']}%</span>
          <div class="hero-label">{d['c_icon']} {d['c_label']}</div>
        </div>
        <div class="hero-right">
          <div class="hero-window">{window_start}–{window_end}</div>
        </div>
      </div>
      <div class="cond-section">
        <div class="cond-bar-wrap">
          <div class="cond-bar-track">
            <div class="cond-bar-mask" style="left:{d['c_score']}%"></div>
            <div class="cond-zone-mark" style="left:{GOOD_ZONE}%"
                 title="Good threshold (score {GOOD_ZONE})"></div>
          </div>
          <div class="cond-zone-labels">
            <span>No wind</span><span>Marginal</span><span>Good</span><span>Excellent</span>
          </div>
        </div>
        <span class="cond-score-value" style="color:{d['score_color']}">{d['c_score']}</span>
      </div>
      <div class="meta-row">
        {exp_chips}
        <span class="meta-chip">p={d['pct']}%{d['lead_note']}</span>
      </div>
      {wind_svg(d['headline'].get("window_wind", {}), cfg)}
      {stats_html(d['headline'], cfg)}
    </div>"""

        # ── Compact grid (next 2 active days after hero) ──────────────────────
        compact_cards = ""
        for cdate in active_dates[1:3]:
            cd = _card_data(cdate)
            cdt = datetime.strptime(cdate, "%Y-%m-%d")
            short_label = cdt.strftime("%a %-d %b").upper()
            compact_chips = expected_wind_chips(cd["headline"], cfg, compact=True)
            compact_cards += f"""
      <div class="compact-card">
        <div class="compact-date">{short_label}</div>
        <span class="compact-pct" style="color:{cd['score_color']}">{cd['pct']}%</span>
        <div class="compact-label">{cd['c_icon']} {cd['c_label']}</div>
        <div class="cond-section compact-bar">
          <div class="cond-bar-wrap">
            <div class="cond-bar-track">
              <div class="cond-bar-mask" style="left:{cd['c_score']}%"></div>
              <div class="cond-zone-mark" style="left:{GOOD_ZONE}%"></div>
            </div>
          </div>
        </div>
        <div class="meta-row">{compact_chips}</div>
      </div>"""

        compact_grid_html = (
            f'<div class="compact-grid">{compact_cards}</div>'
            if compact_cards else ""
        )

        # ── Past days strip (up to 5, newest first) ──────────────────────────
        past_items = ""
        for pdate in past_dates[:5]:
            pd = _card_data(pdate)
            pdt = datetime.strptime(pdate, "%Y-%m-%d")
            short = pdt.strftime("%a %-d")
            past_items += f"""
      <div class="past-day">
        <span class="past-date">{short}</span>
        <span class="past-icon">{pd['c_icon']}</span>
        <span class="past-score">{pd['c_score']}</span>
      </div>"""

        past_row_html = ""
        if past_items:
            past_row_html = f"""
    <div class="past-row">
      <div class="past-days">{past_items}</div>
    </div>"""

        cards_html = hero_html + compact_grid_html + past_row_html

    history_foldout    = history_html(db_path or DEFAULT_SQLITE)
    methodology_foldout = _methodology_html(cfg)

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
      --c-bg:      #0f1117;
      --c-surface: #1e2436;
      --c-border:  #2d3555;
      --c-text:    #e2e8f0;
      --c-muted:   #94a3b8;
      --c-dim:     #4b5675;
      --c-accent:  #22d3ee;
      --c-good:    #16a34a;
      --c-track:   #1e2436;
      --radius:    8px;
      --shadow:    0 1px 3px rgba(0,0,0,0.4);
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
    .page-header {{ margin-bottom: 1.5rem; }}
    .page-header h1 {{
      font-size: 1.4rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--c-text);
    }}
    .page-header .subtitle {{
      color: var(--c-dim);
      font-size: 0.8rem;
      margin-top: 0.2rem;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}

    .no-data {{
      color: var(--c-muted);
      text-align: center;
      padding: 3rem 0;
      font-size: 0.95rem;
    }}

    /* ── Condition score bar ── */
    .cond-section {{
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      margin-bottom: 0.85rem;
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
      color: var(--c-accent);
      font-weight: 500;
      background: rgba(34,211,238,0.08);
      border-color: rgba(34,211,238,0.2);
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

    /* ── Hero card ── */
    .hero-card {{
      background: var(--c-surface);
      border: 1px solid var(--c-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 10px;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .hero-left {{ flex: 1; }}
    .hero-date {{
      color: var(--c-muted);
      font-size: 11px;
      letter-spacing: .06em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    .hero-pct {{
      display: block;
      font-size: 52px;
      font-weight: 700;
      line-height: 1;
    }}
    .hero-label {{
      color: var(--c-dim);
      font-size: 12px;
      margin-top: 4px;
    }}
    .hero-right {{ text-align: right; flex-shrink: 0; }}
    .hero-window {{
      color: var(--c-muted);
      font-size: 10px;
    }}

    /* ── Compact grid ── */
    .compact-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }}
    @media (max-width: 480px) {{
      .compact-grid {{ grid-template-columns: 1fr; }}
    }}
    .compact-card {{
      background: var(--c-surface);
      border: 1px solid var(--c-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 12px;
    }}
    .compact-date {{
      color: var(--c-muted);
      font-size: 10px;
      letter-spacing: .05em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    .compact-pct {{
      display: block;
      font-size: 36px;
      font-weight: 700;
      line-height: 1;
    }}
    .compact-label {{
      color: var(--c-dim);
      font-size: 11px;
      margin-bottom: 8px;
    }}
    .compact-bar .cond-bar-track {{ height: 4px; }}
    .compact-bar .cond-zone-mark {{ top: -2px; height: 8px; }}
    .compact-bar {{ margin-bottom: 8px; }}

    /* ── Past days strip ── */
    .past-row {{
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--c-border);
    }}
    .past-label {{
      color: var(--c-dim);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .06em;
      white-space: nowrap;
    }}
    .past-days {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .past-day  {{ display: flex; align-items: center; gap: 4px; color: var(--c-dim); font-size: 11px; }}
    .past-score {{ font-weight: 600; }}

    /* ── Foldouts ── */
    .foldout {{
      background: var(--c-surface);
      border: 1px solid var(--c-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      margin-bottom: 8px;
      overflow: hidden;
    }}
    .foldout-summary {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
      padding: 10px 14px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--c-text);
      list-style: none;
      user-select: none;
    }}
    .foldout-summary::-webkit-details-marker {{ display: none; }}
    .foldout-summary::after {{
      content: "›";
      margin-left: auto;
      font-size: 1rem;
      color: var(--c-dim);
      transition: transform .2s;
    }}
    details[open] .foldout-summary::after {{ transform: rotate(90deg); }}
    .foldout-hint {{
      color: var(--c-dim);
      font-weight: 400;
      font-size: 0.75rem;
    }}
    .foldout-body {{
      padding: 8px 14px 12px;
      border-top: 1px solid var(--c-border);
    }}

    /* ── History stat cards ── */
    .hist-stats {{
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .hist-stat {{
      display: flex;
      flex-direction: column;
      align-items: center;
      background: var(--c-bg);
      border: 1px solid var(--c-border);
      border-radius: 6px;
      padding: 8px 16px;
      min-width: 72px;
    }}
    .hist-stat-val {{
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--c-accent);
      line-height: 1;
    }}
    .hist-stat-lbl {{
      font-size: 0.62rem;
      color: var(--c-muted);
      text-transform: uppercase;
      letter-spacing: .05em;
      margin-top: 4px;
      text-align: center;
    }}

    /* ── About / methodology ── */
    .about-dl {{ display: flex; flex-direction: column; gap: 10px; }}
    .about-row {{ display: grid; grid-template-columns: 9rem 1fr; gap: 8px; align-items: baseline; }}
    .about-row dt {{
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--c-muted);
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .about-row dd {{
      font-size: 0.8rem;
      color: var(--c-text);
      line-height: 1.55;
      opacity: 0.85;
    }}
    @media (max-width: 480px) {{
      .about-row {{ grid-template-columns: 1fr; gap: 2px; }}
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
      <p class="subtitle">Generated {generated}</p>
    </header>

    {cards_html}

    {history_foldout}
    {methodology_foldout}

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
    html = build_html(predictions, cfg, db_path=DEFAULT_SQLITE)

    with open(args.out, "w") as f:
        f.write(html)

    print(f"Written: {args.out}  ({len(predictions)} predictions, "
          f"{len(set(p['predicting_date'] for p in predictions if 'predicting_date' in p))} days)")


if __name__ == "__main__":
    main()
