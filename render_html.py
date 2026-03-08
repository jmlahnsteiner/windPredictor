"""
render_html.py — Generate a self-contained predictions page.

Reads predictions.json (and optionally config.toml) and writes index.html.

Usage:
    python render_html.py
    python render_html.py --predictions predictions.json --out index.html
"""

import argparse
import json
import os
import tomllib
from collections import defaultdict
from datetime import datetime

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


def build_html(predictions: list[dict], cfg: dict) -> str:
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
