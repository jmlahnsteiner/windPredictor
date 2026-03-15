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


def expected_wind_chips(headline: dict, cfg: dict) -> str:
    """
    Return chip HTML for expected average wind, gusts, and direction consistency.
    Prefers observed window data (past days) over NWP (future days).
    Returns '' when neither source is available.
    (Previously _expected_wind_chips in render_html.py)
    """
    obs = window_stats(headline.get("window_wind", {}), cfg)
    nwp = headline.get("nwp_forecast", {})
    chips = []

    if obs:
        chips.append(f'<span class="meta-chip exp-chip">💨 avg {obs["mean_kn"]} kn</span>')
        if obs.get("max_gust_kn") is not None:
            chips.append(f'<span class="meta-chip exp-chip">↑ gust {obs["max_gust_kn"]} kn</span>')
        if obs.get("dir_std_deg") is not None:
            chips.append(f'<span class="meta-chip exp-chip">〜 dir ±{obs["dir_std_deg"]}°</span>')
    elif nwp:
        if nwp.get("mean_wind_kn") is not None:
            chips.append(f'<span class="meta-chip exp-chip">💨 avg {nwp["mean_wind_kn"]} kn</span>')
        if nwp.get("max_gust_kn") is not None:
            chips.append(f'<span class="meta-chip exp-chip">↑ gust {nwp["max_gust_kn"]} kn</span>')
        if nwp.get("dir_consistency_deg") is not None:
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
        blh_chip   = f'<span class="stats-chip">BLH {nwp["blh_m"]:,} m</span>' if nwp.get("blh_m") is not None else ""
        if cloud_chip or blh_chip:
            rows.append(
                f'<div class="stats-row stats-nwp">'
                f'<span class="stats-label">NWP</span>'
                f'{cloud_chip}{blh_chip}'
                f'</div>'
            )

    if not rows:
        return ""
    return '<div class="stats-block">' + "".join(rows) + "</div>\n"


def history_html(db_path: str, days: int = 60) -> str:
    """
    Render prediction accuracy history section.
    (Previously _history_html in render_html.py)
    Uses utils/db.backend() instead of importing model.history._backend.
    """
    from utils.db import backend, DEFAULT_SQLITE
    from model.history import load_history, accuracy_summary

    bk = backend()
    actual_path = db_path or DEFAULT_SQLITE
    if bk == "sqlite" and not os.path.exists(actual_path):
        return ""

    df = load_history(db_path=actual_path, days=days)
    if df.empty:
        return ""

    # One row per predicting_date: use the last snapshot of each day
    df = df.sort_values("snapshot_dt").groupby("predicting_date").last().reset_index()
    df = df.sort_values("predicting_date")

    summary = accuracy_summary(db_path=actual_path, days=days)

    dots = ""
    for _, row in df.iterrows():
        pred_good  = int(row["good"])
        ag         = row["actual_good"]
        af         = row["actual_frac"]
        has_outcome = (ag is not None) and (ag == ag)
        date_label  = row["predicting_date"]
        prob_pct    = round(float(row["probability"]) * 100)

        if not has_outcome:
            title = (
                f"{date_label}: predicted {'good' if pred_good else 'poor'} "
                f"({prob_pct}%) — outcome pending"
            )
            dots += (
                f'<span class="hist-dot hist-pending" title="{title}">·</span>'
            )
        else:
            actual_good  = int(ag)
            actual_score = int(round(float(af) * 100)) if (af == af and af is not None) else 0
            fill_color   = score_to_hex(actual_score)
            correct      = pred_good == actual_good
            pred_label   = "good" if pred_good else "poor"
            act_label    = "good" if actual_good else "poor"
            title = (
                f"{date_label}: predicted {pred_label} ({prob_pct}%), "
                f"actual {act_label} (score {actual_score}) "
                f"[{'✓' if correct else '✗'}]"
            )
            border_style = "solid" if correct else "dashed"
            border_color = "#15803d" if correct else "#dc2626"
            dots += (
                f'<span class="hist-dot" '
                f'style="background:{fill_color};border:2px {border_style} {border_color};" '
                f'title="{title}"></span>'
            )

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
        '<span style="display:inline-block;width:12px;height:12px;background:#22c55e;'
        'border:2px solid #15803d;border-radius:2px;vertical-align:middle"></span>'
        ' correct &nbsp;'
        '<span style="display:inline-block;width:12px;height:12px;background:#fbbf24;'
        'border:2px dashed #dc2626;border-radius:2px;vertical-align:middle"></span>'
        ' wrong &nbsp;'
        '<span style="color:#94a3b8;font-size:1.1rem;vertical-align:middle">·</span>'
        ' pending &nbsp;'
        '<span style="font-size:0.68rem;color:var(--c-muted)">colour = actual conditions</span>'
    )

    return f"""
    <section class="history-section">
      <h3 class="history-title">Prediction history <span class="history-days">({days}d)</span></h3>
      <div class="history-dots">{dots}</div>
      <div class="history-legend">{legend}</div>
      {"<div class='history-stats'>" + stats_line + "</div>" if stats_line else ""}
    </section>"""
