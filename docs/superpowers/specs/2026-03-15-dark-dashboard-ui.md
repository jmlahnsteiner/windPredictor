# Wind Predictor — Dark Dashboard UI Redesign

**Date:** 2026-03-15
**Status:** Approved

## Goal

Redesign `index.html` to use a dark dashboard aesthetic matching the Ecowitt WS View dashboard, with a hero-card + grid layout that prioritises the most immediate day.

## Visual Reference

The Ecowitt WS View dashboard (dark navy background, card grid, large numbers, teal/cyan accents) is the inspiration. The wind predictor page adapts this to a forecast context rather than a real-time sensor view.

## Layout

```
┌─────────────────────────────────────────────┐
│  ⛵ WIND PREDICTOR · 15 MARCH 2026, 15:46   │  ← page header
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  HERO CARD — next/most immediate day        │  ← large, full-detail
│  Big % · condition bar · chips · sparkline  │
│  Forecast secondary stats                   │
└─────────────────────────────────────────────┘

┌───────────────────┐  ┌───────────────────┐
│  COMPACT CARD     │  │  COMPACT CARD     │  ← 2-up grid, next 2 days
│  % · bar · chips  │  │  % · bar · chips  │
└───────────────────┘  └───────────────────┘

┌─────────────────────────────────────────────┐
│  PAST DAYS ROW (dimmed, smaller text)       │  ← compact, muted
└─────────────────────────────────────────────┘

  Sailing window 08:00–16:00 · 2–12 kn · 30%  ← footer
```

## Colour Tokens

| Token | Value | Use |
|-------|-------|-----|
| `--c-bg` | `#0f1117` | Page background |
| `--c-surface` | `#1e2436` | Card background |
| `--c-border` | `#2d3555` | Card border |
| `--c-text` | `#e2e8f0` | Primary text |
| `--c-muted` | `#94a3b8` | Secondary labels |
| `--c-dim` | `#4b5675` | Tertiary / past days |
| `--c-accent` | `#22d3ee` | Teal accent (chips, sparkline dot) |

Condition colours (score 0–100) are unchanged from the current `score_to_hex()` palette (grey → light-blue → yellow → green → emerald → cyan).

## Hero Card

The first active predicting date (today if window is open; tomorrow otherwise).

**Contents:**
- Date label (`SATURDAY, 15 MARCH`) — small, muted, uppercase
- Probability — giant number (`52px`, coloured by `score_to_hex`)
- Condition label + icon below the number
- Sailing window time range — top-right, small muted
- Probability sparkline SVG — top-right, under the time range
- Condition gradient bar (same design as current, full width)
- Wind chips row: avg wind · gust · dir consistency · p=X%
- Secondary stats row (only when present): `Forecast  cloud X% · boundary layer Y m`
  — or `Observed  X% in range · max Y kn` when window_wind data is available

**Left/right split:** date + big % + label on the left; time range + sparkline on the right.

## Compact Cards (2-up grid)

The next two active days after the hero.

**Contents:**
- Short date label (`SUN 16 MAR`) — small, muted
- Probability — large number (`36px`, coloured by score)
- Condition icon + label
- Condition gradient bar (thinner, 4px)
- Wind chips (avg, gust only — no dir or p chip to save space)

No secondary stats row, no sparkline.

**Grid:** `grid-template-columns: 1fr 1fr`. On narrow viewports (< 480 px) stacks to single column.

## Past Days Row

All dates where the sailing window has already closed (date < today, or today and window_closed).

**Displayed as a single horizontal strip** below the 2-up grid, one entry per past day, left-to-right newest-first. If more than 5 past days exist, show only the 5 most recent.

**Per entry:**
- Short date (`Sat 8`) — very small, `--c-dim`
- Icon (⛵ or 🌫)
- Score number — small, `--c-dim`

No bars, no chips. The strip is visually separated from the main cards by a subtle top border.

## Removed Elements

- Prediction history section (already removed)
- Snapshots dropdown (already removed)
- Past-days fold-out (replaced by past days row)

## Responsive Behaviour

- Hero card: always full width
- 2-up grid: side-by-side ≥ 480 px; single column below
- Past days row: wraps naturally

## Files Changed

| File | Change |
|------|--------|
| `render_html.py` | New CSS tokens, new card layout logic, past-days row renderer |
| `render/charts.py` | `prob_trend_svg()` — update stroke colour to `--c-accent` (#22d3ee) |
| `render/data.py` | No changes needed |

## What Stays the Same

- All Python data pipeline code (predict.py, weather_store.py, etc.)
- `score_to_hex()` colour mapping
- Wind chip content (avg, gust, dir, p values)
- Condition gradient bar design (same gradient, same zone marker)
- Footer parameters line
