# Wind Predictor — Dark Dashboard UI Redesign

**Date:** 2026-03-15
**Status:** Approved

## Goal

Redesign `index.html` to use a dark dashboard aesthetic matching the Ecowitt WS View dashboard, with a hero-card + grid layout that prioritises the most immediate day.

## Visual Reference

The Ecowitt WS View dashboard (dark navy background, card grid, large numbers, teal/cyan accents). The page is **always-dark** — the existing `@media (prefers-color-scheme: dark)` override block and the light-mode `:root` defaults are both removed and replaced by a single forced-dark `:root`.

---

## Layout

```
┌─────────────────────────────────────────────┐
│  ⛵ WIND PREDICTOR · 15 MARCH 2026, 15:46   │  ← page header
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  HERO CARD — next/most immediate day        │
└─────────────────────────────────────────────┘

┌───────────────────┐  ┌───────────────────┐
│  COMPACT CARD     │  │  COMPACT CARD     │  ← 2-up grid
└───────────────────┘  └───────────────────┘

┌─────────────────────────────────────────────┐
│  PAST DAYS ROW (dimmed strip)               │
└─────────────────────────────────────────────┘

  Sailing window 08:00–16:00 · 2–12 kn · 30%
```

---

## Colour Tokens

Replace the entire existing `:root` block (both light defaults and the `@media` dark override) with:

```css
:root {
  --c-bg:      #0f1117;
  --c-surface: #1e2436;
  --c-border:  #2d3555;
  --c-text:    #e2e8f0;
  --c-muted:   #94a3b8;
  --c-dim:     #4b5675;
  --c-accent:  #22d3ee;   /* NEW token */
  --c-good:    #16a34a;   /* retained — used by .stats-observed .stats-label */
  --c-track:   #1e2436;   /* retained — used by .cond-bar-mask background */
  --radius:    8px;        /* retained — used by .history-section, .day-card */
  --shadow:    0 1px 3px rgba(0,0,0,0.4);  /* retained — updated for dark theme */
}
```

`--c-accent` is the only new token; all others already exist in the code and are simply assigned new values.

Condition colours (score 0–100) are unchanged from the current `score_to_hex()` palette.

---

## Condition Badges (`_COND_BADGE` / `_badge_style`)

Replace the `_COND_BADGE` dict with dark-theme versions. Each entry becomes:
```python
"Good":      ("#22c55e",  "rgba(34,197,94,0.12)",   "rgba(34,197,94,0.25)"),
"Excellent": ("#06b6d4",  "rgba(6,182,212,0.12)",   "rgba(6,182,212,0.25)"),
"Marginal":  ("#fbbf24",  "rgba(251,191,36,0.12)",  "rgba(251,191,36,0.25)"),
"No wind":   ("#94a3b8",  "rgba(148,163,184,0.10)", "rgba(148,163,184,0.20)"),
"Poor":      ("#f472b6",  "rgba(244,114,182,0.12)", "rgba(244,114,182,0.25)"),
```
Tuple is `(text_hex, bg_rgba, border_rgba)`. The existing `_badge_style()` function reads all three and emits the inline style — **no signature change needed**. No hex-to-rgb conversion is required; the rgba values are provided literally above.

---

## Hero Card

**Definition:** `active_dates[0]` where "active" means the sailing window has not yet closed.
**Fallback:** If `active_dates` is empty, use the most recent past date. If `by_date` is entirely empty, render a `<p class="no-data">No forecast data available.</p>` in place of the cards section.

**HTML skeleton:**

```html
<div class="hero-card">
  <div class="hero-top">
    <div class="hero-left">
      <div class="hero-date">SATURDAY, 15 MARCH</div>
      <span class="hero-pct" style="color:{score_color}">{pct}%</span>
      <div class="hero-label">{c_icon} {c_label}</div>
    </div>
    <div class="hero-right">
      <div class="hero-window">08:00 – 16:00</div>
      <div class="hero-sparkline">{prob_trend_svg(snaps)}</div>
    </div>
  </div>
  <div class="cond-section"> ... </div>   <!-- existing bar markup, unchanged -->
  <div class="meta-row"> ... </div>        <!-- chips row -->
  <div class="stats-block"> ... </div>     <!-- secondary stats, if present -->
</div>
```

**Key CSS (new classes):**
```css
.hero-card { background:var(--c-surface); border:1px solid var(--c-border); border-radius:8px; padding:16px; margin-bottom:10px; }
.hero-top  { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px; }
.hero-date { color:var(--c-muted); font-size:11px; letter-spacing:.06em; text-transform:uppercase; margin-bottom:4px; }
.hero-pct  { display:block; font-size:52px; font-weight:700; line-height:1; }
.hero-label{ color:var(--c-dim); font-size:12px; margin-top:4px; }
.hero-right{ text-align:right; }
.hero-window{ color:var(--c-muted); font-size:10px; margin-bottom:6px; }
.hero-sparkline svg { width:90px; height:30px; }
```

---

## Compact Cards (2-up grid)

`active_dates[1]` and `active_dates[2]`. If only one exists, render one card; if none, skip the grid row entirely.

**HTML skeleton:**
```html
<div class="compact-grid">
  <div class="compact-card">
    <div class="compact-date">SUN 16 MAR</div>
    <span class="compact-pct" style="color:{score_color}">{pct}%</span>
    <div class="compact-label">{c_icon} {c_label}</div>
    <div class="cond-section compact-bar"> ... </div>
    <div class="meta-row">{compact_chips}</div>
  </div>
  ...
</div>
```

**Key CSS:**
```css
.compact-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px; }
.compact-card { background:var(--c-surface); border:1px solid var(--c-border); border-radius:8px; padding:12px; }
.compact-date { color:var(--c-muted); font-size:10px; letter-spacing:.05em; text-transform:uppercase; margin-bottom:4px; }
.compact-pct  { display:block; font-size:36px; font-weight:700; line-height:1; }
.compact-label{ color:var(--c-dim); font-size:11px; margin-bottom:8px; }
.compact-bar .cond-bar-track { height:4px; }   /* override the global 10px */
@media (max-width:480px) { .compact-grid { grid-template-columns:1fr; } }
```

**Chip suppression:** Call `expected_wind_chips(headline, cfg, compact=True)`. When `compact=True`, `expected_wind_chips()` skips the dir-consistency chip. The inline `p=X%` chip is also omitted at the call site (only emitted for the hero card). This requires adding `compact: bool = False` to `expected_wind_chips()` in `render/data.py`.

---

## Past Days Row

Replaces the existing `<details>` fold-out entirely. Shows the **5 most recent past dates**, **newest-first left-to-right**.

**HTML:**
```html
<div class="past-row">
  <span class="past-label">Past</span>
  <div class="past-days">
    <div class="past-day">
      <span class="past-date">Sat 8</span>
      <span class="past-icon">⛵</span>
      <span class="past-score">72</span>
    </div>
    <!-- ... up to 5 entries -->
  </div>
</div>
```

**CSS:**
```css
.past-row   { display:flex; align-items:center; gap:1rem; margin-top:8px; padding-top:8px; border-top:1px solid var(--c-border); }
.past-label { color:var(--c-dim); font-size:10px; text-transform:uppercase; letter-spacing:.06em; white-space:nowrap; }
.past-days  { display:flex; flex-wrap:wrap; gap:12px; }
.past-day   { display:flex; align-items:center; gap:4px; color:var(--c-dim); font-size:11px; }
.past-score { font-weight:600; }
```

Emoji icons are rendered at natural size; no forced colour.

---

## Charts (`render/charts.py` — `prob_trend_svg`)

Change these hardcoded colour literals:

| Element | Current value | New value |
|---------|--------------|-----------|
| Connecting line stroke | `#16a34a` | `#22d3ee` (accent) at `opacity="0.7"` |
| Latest-value dot fill | `#16a34a` | `#22d3ee` |
| Threshold dashed line stroke | `#16a34a` | `#94a3b8` (literal, not CSS var — SVG stroke attributes accept literals) |
| Per-dot colours | `score_to_hex(score)` | unchanged |

The current `prob_trend_svg()` emits the `<svg>` element with an inline `style="width:100%;height:38px"` attribute. Inline styles take precedence over embedded `<style>` rules, so CSS on `.hero-sparkline svg` would be overridden. Fix: add an optional `size: tuple[int,int] | None = None` parameter to `prob_trend_svg()`. When provided (e.g. `size=(90, 30)`), the function sets `width="{w}" height="{h}"` directly on the `<svg>` element (removing the inline `style`). When `None`, the existing `style="width:100%;height:{VH}px"` behaviour is preserved for the full-width usage elsewhere. The hero card calls `prob_trend_svg(snaps, size=(90, 30))`; all other call sites pass no `size` argument. No changes to `VW`, `VH`, or `viewBox` constants.

---

## Files Changed

| File | Change |
|------|--------|
| `render_html.py` | Replace `:root` + `@media` dark block with single always-dark `:root`; update `_COND_BADGE`; new hero/compact/past-row layout and CSS; remove fold-out past-days block |
| `render/charts.py` | `prob_trend_svg()` — update 3 colour literals as above |
| `render/data.py` | `expected_wind_chips()` — add `compact: bool = False`; skip dir chip when True |

---

## What Stays the Same

- All Python data pipeline code
- `score_to_hex()` colour mapping
- Wind chip content values (avg, gust, dir, p values)
- Condition gradient bar design (gradient stops, zone marker, good-threshold marker at 60%)
- `_badge_style()` function signature
- `prob_trend_svg()` function signature
- Footer parameters line
