"""render/charts.py — SVG chart generators for the forecast page."""
import math
from datetime import datetime


def prob_trend_svg(snaps: list[dict], size: tuple[int, int] | None = None) -> str:
    """
    Compact sparkline showing how sailing probability evolved across snapshots.
    Returns '' when fewer than 2 snapshots exist.
    When size=(w, h) is provided, emits width/height attributes on the <svg> element
    instead of an inline style (useful for fixed-size inline hero sparklines).
    (Previously _prob_trend_svg in render_html.py)
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

    if size is not None:
        w, h = size
        svg_attrs = f'viewBox="0 0 {VW} {VH}" width="{w}" height="{h}" class="dropout-svg" aria-hidden="true"'
    else:
        svg_attrs = (
            f'viewBox="0 0 {VW} {VH}" '
            f'style="width:100%;height:{VH}px;display:block;margin-bottom:.5rem" '
            f'class="dropout-svg" aria-hidden="true"'
        )

    out = [f'<svg {svg_attrs}>']

    out.append(
        f'<text x="{PAD_L - 3}" y="{PAD_T + ch / 2 + 2.5:.1f}" '
        f'font-size="6.5" text-anchor="end" class="do-lbl">Forecast</text>'
    )
    # Threshold dashed line
    hy = ty(thr)
    out.append(
        f'<line x1="{PAD_L}" y1="{hy:.1f}" x2="{VW - PAD_R}" y2="{hy:.1f}" '
        f'stroke="#94a3b8" stroke-width="0.75" stroke-dasharray="3,2" opacity="0.4"/>'
    )
    # Connecting line
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    out.append(
        f'<polyline points="{pts}" fill="none" stroke="#22d3ee" '
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


def wind_svg(window_wind: dict, cfg: dict) -> str:
    """
    Two-panel SVG: wind speed time series + compass rose.
    Returns '' when window data is unavailable.
    (Previously _wind_svg in render_html.py)
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
    VW, VH = 360, 220
    SPLIT  = 230          # left / right panel boundary

    LP_L, LP_R = 28, 6
    LP_T, LP_B = 16, 22   # LP_T: panel title; LP_B: time-axis labels
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
        f'style="width:100%;max-height:{VH}px;display:block;margin-bottom:.75rem" '
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


def _hourly_quality(window_wind: dict, cfg: dict | None) -> list[bool | None]:
    """
    Return one entry per sailing-window hour: True=good, False=off-range, None=no data.
    """
    sc = (cfg or {}).get("sailing", {})
    wind_min = sc.get("wind_speed_min", 2.0)
    wind_max = sc.get("wind_speed_max", 10.0)
    ws_h = int(sc.get("window_start", "08:00").split(":")[0])
    we_h = int(sc.get("window_end",   "16:00").split(":")[0])

    times  = window_wind.get("times", [])
    speeds = window_wind.get("speeds_kn", [])

    by_hour: dict[int, list[float]] = {}
    for t, s in zip(times, speeds):
        h = int(t.split(":")[0])
        if ws_h <= h < we_h:
            by_hour.setdefault(h, []).append(s)

    result: list[bool | None] = []
    for h in range(ws_h, we_h):
        readings = by_hour.get(h)
        if not readings:
            result.append(None)
        else:
            result.append(wind_min <= (sum(readings) / len(readings)) <= wind_max)
    return result


def history_chart_svg(
    rows: list[dict],
    threshold: float = 0.3,
    window_wind_by_date: dict | None = None,
    cfg: dict | None = None,
) -> str:
    """
    Time-series chart: predicted probability vs actual wind quality over past days.
    rows: list of dicts with keys:
        predicting_date (str YYYY-MM-DD), probability (float),
        actual_frac (float | None — None means outcome not yet known).
    window_wind_by_date: optional dict mapping date → window_wind, used to draw
        an intraday hourly-quality strip below the x-axis.
    Returns SVG string, or '' if fewer than 2 rows.
    """
    rows = sorted(rows, key=lambda r: r["predicting_date"])
    n = len(rows)
    if n < 2:
        return ""

    STRIP_H   = 10   # height of the intraday strip below date labels
    VW, VH    = 620, 144   # +14 vs original 130 to fit the strip
    PAD_L, PAD_R, PAD_T, PAD_B = 36, 72, 14, 22
    cw = VW - PAD_L - PAD_R
    ch = VH - STRIP_H - PAD_T - PAD_B   # chart area height unchanged

    CHART_BOTTOM = PAD_T + ch          # y where chart area ends (= start of x-axis)
    LABEL_Y      = CHART_BOTTOM + PAD_B - 2    # date label baseline (same relative position)
    STRIP_Y      = LABEL_Y + 3                 # intraday strip starts just below labels

    def tx(i: int) -> float:
        return PAD_L + (i / max(n - 1, 1)) * cw

    def ty(v: float) -> float:
        return PAD_T + ch * (1.0 - min(max(v, 0.0), 1.0))

    out = [
        f'<svg viewBox="0 0 {VW} {VH}" '
        f'style="width:100%;height:{VH}px;display:block;overflow:visible" '
        f'aria-hidden="true">'
    ]

    # Horizontal grid + y-axis labels at 0%, threshold, 100%
    for v, label in [(0.0, "0%"), (threshold, f"{int(threshold * 100)}%"), (1.0, "100%")]:
        gy = ty(v)
        dash = 'stroke-dasharray="4,2"' if v == threshold else ""
        out.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{VW - PAD_R}" y2="{gy:.1f}" '
            f'stroke="#2d3555" stroke-width="0.75" {dash}/>'
        )
        out.append(
            f'<text x="{PAD_L - 4}" y="{gy + 3:.1f}" font-size="7" text-anchor="end" '
            f'fill="#4b5675" font-family="sans-serif">{label}</text>'
        )
    # Threshold label on the right
    out.append(
        f'<text x="{VW - PAD_R + 4}" y="{ty(threshold) + 3:.1f}" font-size="6.5" '
        f'fill="#4b5675" font-family="sans-serif">threshold</text>'
    )

    # Actual fraction line (green) — only points with known outcomes
    actual_coords = [
        (tx(i), ty(float(r["actual_frac"])))
        for i, r in enumerate(rows)
        if r.get("actual_frac") is not None
    ]
    if len(actual_coords) >= 2:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in actual_coords)
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="#16a34a" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.75"/>'
        )
    for i, r in enumerate(rows):
        af = r.get("actual_frac")
        if af is not None:
            c = "#22c55e" if float(af) >= threshold else "#64748b"
            out.append(
                f'<circle cx="{tx(i):.1f}" cy="{ty(float(af)):.1f}" r="2.5" '
                f'fill="{c}" opacity="0.85"/>'
            )

    # Predicted probability line (cyan, dashed)
    pred_pts = " ".join(
        f"{tx(i):.1f},{ty(float(r['probability'])):.1f}" for i, r in enumerate(rows)
    )
    out.append(
        f'<polyline points="{pred_pts}" fill="none" stroke="#22d3ee" '
        f'stroke-width="1.5" stroke-dasharray="5,2" stroke-linecap="round" opacity="0.8"/>'
    )
    for i, r in enumerate(rows):
        py = ty(float(r["probability"]))
        out.append(
            f'<circle cx="{tx(i):.1f}" cy="{py:.1f}" r="2" '
            f'fill="#22d3ee" opacity="0.65"/>'
        )

    # X-axis ticks + sparse date labels (up to ~6)
    step = max(1, n // 5)
    shown_idx = set(range(0, n, step)) | {n - 1}
    for i, r in enumerate(rows):
        x = tx(i)
        out.append(
            f'<line x1="{x:.1f}" y1="{CHART_BOTTOM:.1f}" x2="{x:.1f}" y2="{CHART_BOTTOM + 3:.1f}" '
            f'stroke="#2d3555" stroke-width="0.5"/>'
        )
        if i in shown_idx:
            dt_obj = datetime.strptime(r["predicting_date"], "%Y-%m-%d")
            label = dt_obj.strftime("%-d %b")
            anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
            out.append(
                f'<text x="{x:.1f}" y="{LABEL_Y:.1f}" font-size="7" text-anchor="{anchor}" '
                f'fill="#4b5675" font-family="sans-serif">{label}</text>'
            )

    # Intraday hourly-quality strip
    if window_wind_by_date:
        SEG_W, SEG_H, SEG_GAP = 2.5, 8.0, 0.5
        for i, r in enumerate(rows):
            # Only show the strip when a full outcome has been recorded,
            # so the bars reflect complete window data rather than a partial snapshot.
            if r.get("actual_frac") is None:
                continue
            ww = window_wind_by_date.get(r["predicting_date"])
            if not ww:
                continue
            hourly = _hourly_quality(ww, cfg)
            if not hourly:
                continue
            nh      = len(hourly)
            total_w = nh * SEG_W + (nh - 1) * SEG_GAP
            x0      = tx(i) - total_w / 2
            for j, quality in enumerate(hourly):
                sx = x0 + j * (SEG_W + SEG_GAP)
                color = "#22c55e" if quality is True else ("#334155" if quality is False else "#1e2436")
                out.append(
                    f'<rect x="{sx:.1f}" y="{STRIP_Y:.1f}" width="{SEG_W}" height="{SEG_H}" '
                    f'fill="{color}" rx="0.5"/>'
                )

    # Legend (top-left inside chart area)
    lx, ly = PAD_L + 6, PAD_T + 9
    out.append(
        f'<line x1="{lx}" y1="{ly}" x2="{lx + 14}" y2="{ly}" '
        f'stroke="#22d3ee" stroke-width="1.5" stroke-dasharray="5,2"/>'
    )
    out.append(
        f'<text x="{lx + 17}" y="{ly + 3}" font-size="6.5" fill="#94a3b8" '
        f'font-family="sans-serif">Predicted</text>'
    )
    out.append(
        f'<line x1="{lx + 68}" y1="{ly}" x2="{lx + 82}" y2="{ly}" '
        f'stroke="#16a34a" stroke-width="1.8"/>'
    )
    out.append(
        f'<text x="{lx + 85}" y="{ly + 3}" font-size="6.5" fill="#94a3b8" '
        f'font-family="sans-serif">Actual</text>'
    )

    out.append("</svg>")
    return "\n".join(out)
