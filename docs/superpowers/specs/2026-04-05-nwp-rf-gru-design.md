# Design: NWP-enriched RF + GRU experiment

**Date:** 2026-04-05  
**Branch:** `feature/nwp-gru-model`  
**Status:** Approved for implementation

---

## Problem

The current Random Forest predicts purely from local weather station readings. It has no
access to synoptic-scale context: cloud cover forecasts, boundary layer height, expected
direct radiation, or the NWP wind forecast for the target sailing window. Open-Meteo NWP
data is already fetched at runtime but only for display — it is never fed to the model.

Additionally, hand-crafted rolling statistics (3h/6h/12h/24h aggregates) may not fully
capture temporal patterns in the raw time series that predict sailing conditions.

---

## Goals

1. Feed NWP forecast features into the Random Forest so it learns from synoptic context.
2. Build a GRU sequence model as a parallel experiment to evaluate whether temporal
   sequences contain signal that hand-crafted features miss.
3. Training window stays at 2 years (targets require actual station readings; cannot extend).

---

## Architecture Overview

```
Phase 1 — NWP-enriched RF (production baseline)
  input/open_meteo_historical.py   fetch ERA5 historical NWP (archive API)
  input/nwp_store.py               upsert/load nwp_readings table (Supabase / SQLite)
  model/features.py                extract_snapshot_features gains optional nwp_df param
  model/train.py                   loads nwp_df alongside station df, passes to features
  model/predict.py                 passes live NWP forecast df to feature extraction
  supabase/schema_nwp.sql          nwp_readings table DDL

Phase 2 — GRU experiment (parallel, not yet production)
  model/features_sequence.py       hourly resample → (T=24, F) sequence tensors
  model/gru_model.py               PyTorch GRU architecture
  model/train_gru.py               time-series CV, saves weights + comparison metrics
  model/weights_gru.pt             saved GRU weights (gitignored)
  docs/gru_eval.md                 comparison results written after training
```

---

## Phase 1: NWP-enriched Random Forest

### Data store

New table `nwp_readings` with schema:

```sql
CREATE TABLE IF NOT EXISTS nwp_readings (
    timestamp        TEXT PRIMARY KEY,
    temperature      REAL,
    wind_speed       REAL,
    wind_direction   REAL,
    wind_gust        REAL,
    cloud_cover      REAL,
    blh              REAL,
    direct_radiation REAL
);
```

Hourly resolution. SQLite fallback identical to Supabase schema (same pattern as
`weather_readings`). Data is deterministic given location — safe to re-fetch if lost.

### Historical backfill

`input/open_meteo_historical.py`:
- Fetches from `https://archive-api.open-meteo.com/v1/archive`
- Same variables as the existing forecast API
- **Must pass `wind_speed_unit=kn` and `timezone=auto`** — identical to `fetch_forecast`
  so historical and live NWP are on the same scale and timezone
- Fetches in 90-day chunks to avoid timeout
- Stores tz-aware timestamps (matching station data timezone via `timezone=auto`)
- CLI: `python input/open_meteo_historical.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]`
- Upserts via `input/nwp_store.py`

### nwp_store.py interface

```python
def upsert_nwp_readings(df: pd.DataFrame) -> int:
    """Upsert NWP rows into nwp_readings. Returns rows written."""

def load_nwp_readings(
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load NWP readings from DB. Returns tz-aware DatetimeIndex DataFrame.
    Same signature pattern as load_weather_readings()."""
```

DataFrame schema: columns `temperature`, `wind_speed`, `wind_direction`, `wind_gust`,
`cloud_cover`, `boundary_layer_height`, `direct_radiation` — identical to the
`_COL_NAMES` mapping in `open_meteo.py`. Timestamps stored as TEXT in DB (ISO 8601 with
offset); `load_nwp_readings` is responsible for parsing back to tz-aware DatetimeIndex
(same pattern as `load_weather_readings` in `weather_store.py`).

### NWP features

Signature change: `extract_snapshot_features(df, snap_dt, nwp_df=None, cfg=None)`.

For the snapshot predicting `target_date`, extract the NWP sailing window slice
(window_start–window_end on target_date) and compute:

| Feature | Description |
|---|---|
| `nwp_wind_speed_mean` | Mean NWP wind speed in window (kn) |
| `nwp_wind_speed_max` | Max NWP wind speed in window (kn) |
| `nwp_wind_gust_max` | Max NWP gust in window (kn) |
| `nwp_wind_dir_sin` | Circular mean direction — sin component |
| `nwp_wind_dir_cos` | Circular mean direction — cos component |
| `nwp_dir_consistency` | Circular std of forecast direction (°) |
| `nwp_cloud_cover_mean` | Mean cloud cover in window (%) |
| `nwp_blh_mean` | Mean boundary layer height in window (m) |
| `nwp_direct_radiation_mean` | Mean direct solar radiation in window (W/m²) |

When `nwp_df` is None or `cfg` is None, all `nwp_*` features are NaN — the RF imputes
with training-time medians, so the model degrades gracefully if NWP is unavailable.

**Call sites that must be updated:**
- `build_training_pairs` (features.py:343) — pass `nwp_df=nwp_df, cfg=cfg`
- `predict_snapshot` (predict.py:131) — pass `nwp_df=nwp_df, cfg=cfg`
- Extended-forecast entries in `predict_now` reuse `result["probability"]` directly and
  do not call `extract_snapshot_features`, so no change needed there.

**Model retraining requirement:** Adding `nwp_*` columns changes the `feature_names` list
saved in `model/weights.joblib`. The existing weights file is incompatible with the new
feature set. `train.py` must be re-run after Phase 1 is implemented. The `feature_medians`
dict saved in the bundle must include medians for all `nwp_*` columns (computed from the
training set, not inferred at prediction time).

### Training changes

`build_training_pairs(df, cfg, nwp_df=None)` — loads NWP df optionally, passes to
`extract_snapshot_features`. `train.py` loads NWP from `nwp_store.load_nwp_readings()`.

### Prediction changes

`predict_snapshot(df, snap_dt, bundle, cfg, nwp_df=None)` — passes live NWP df to
`extract_snapshot_features`. `predict_now` already fetches `nwp_df` via `fetch_forecast`;
it passes this df down rather than using it only for display enrichment.

---

## Phase 2: GRU Experiment

### Sequence representation

`model/features_sequence.py`:
- Resample station data to 1h (mean aggregation)
- For each snapshot, take the 24 hours ending at `snap_dt`
- Per-timestep features: `wind_speed`, `wind_dir_sin`, `wind_dir_cos`, `temperature`,
  `pressure_relative`, `humidity`, `solar` (7 channels; NaN → 0 + mask)
- Static context appended to GRU final hidden state: NWP sailing window stats (9 features
  from Phase 1) + `sin_doy`, `cos_doy`, `snapshot_hour` (3 features) = 12 context features

### GRU architecture (`model/gru_model.py`)

```
Input: (B, T=24, F=7)
GRU(7 → 64, num_layers=2, dropout=0.3)
→ take last hidden state (B, 64)
→ concat static context (B, 64+12=76)
→ Linear(76, 32) → ReLU → Dropout(0.3)
→ Linear(32, 1) → Sigmoid
```

Loss: BCELoss with class weights (mirror RF's `class_weight="balanced"`).

### Training + evaluation (`model/train_gru.py`)

- Time-series CV: 5 temporal folds (no shuffle — future must not leak into past)
- **RF training is also updated to use temporal folds** (replacing `StratifiedKFold` with
  `TimeSeriesSplit`) so both models are evaluated on the same split strategy for a fair
  comparison. Shuffled CV leaks future data into past folds; temporal CV is more correct
  for this problem regardless of the GRU experiment.
- Metrics: ROC-AUC, precision, recall, F1 (same for both models)
- Early stopping on validation loss (patience=10)
- Saves best weights to `model/weights_gru.pt`
- Writes `docs/gru_eval.md` with side-by-side RF vs GRU metrics

### Promotion criteria

GRU replaces RF in production only if it shows ≥3% ROC-AUC improvement on held-out folds
AND precision/recall are not degraded. Otherwise RF with NWP features remains in production.

---

## Deploy pipeline changes

`deploy.py`: add a `--backfill-nwp` flag that runs `open_meteo_historical.py` for any
gaps in `nwp_readings`. Normal pipeline runs auto-fetch the current NWP forecast (already
done) — the only change is passing it to the model.

GitHub Actions: no changes needed for Phase 1 inference (NWP already fetched). Phase 2
(GRU training) runs locally only for now.

---

## What does NOT change

- `render_html.py` and all render code — NWP display enrichment already works
- `forecast_snapshots` table schema — payload JSON gains extra `nwp_*` feature keys
  which are ignored by the renderer
- `predictions` / `outcomes` history tables
- GitHub Pages deployment workflow
