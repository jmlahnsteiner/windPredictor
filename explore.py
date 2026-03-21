"""
explore.py — Visualise historical data and trained model.

Reads weather readings from Supabase (or local.db when SUPABASE_DB_URL is
unset) and model/weights.joblib. All plots are saved to PNG; nothing is
displayed interactively (safe for headless Pi use).

Usage:
    python explore.py                   # all panels → exploration.png
    python explore.py --data            # raw data overview only
    python explore.py --model           # feature importances only
    python explore.py --out report.png  # custom output path
"""

import argparse
import os
import sys

import matplotlib.path as _mpath
import numpy as np


# ── matplotlib deepcopy fix for Python 3.14 ──────────────────────────────────
def _fixed_path_deepcopy(self, memo=None):
    p = _mpath.Path(
        vertices=np.array(self._vertices),
        codes=np.array(self._codes) if self._codes is not None else None,
        _interpolation_steps=self._interpolation_steps,
        readonly=False,
    )
    if memo is not None:
        memo[id(self)] = p
    return p


_mpath.Path.__deepcopy__ = _fixed_path_deepcopy
_mpath.Path.deepcopy = _fixed_path_deepcopy
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.features import _circular_std, compute_daily_target
from input.weather_store import load_weather_readings
from utils.config import load_config

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "config.toml")

# ── global style ──────────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
RED    = "#DD4444"
GREEN  = "#2CA02C"
ORANGE = "#FF7F0E"
GREY   = "#888888"

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titleweight":   "semibold",
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.7,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8F8F8",
})


def _date_axis(ax: plt.Axes, span_days: int) -> None:
    """Apply a non-overlapping date formatter based on the data span."""
    if span_days > 365:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    elif span_days > 90:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_daily_fraction(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Good-sailing fraction per day over the full dataset."""
    daily = compute_daily_target(df, cfg)
    dates = [pd.Timestamp(d) for d in daily.index]
    s = pd.Series(daily.values, index=dates)
    roll = s.rolling("30d").mean()

    ax.fill_between(dates, daily.values, alpha=0.18, color=BLUE)
    ax.plot(dates, daily.values, linewidth=0.6, color=BLUE, alpha=0.5)
    ax.plot(roll.index, roll.values, color=RED, linewidth=2, label="30-day mean", zorder=3)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Daily good-sailing fraction")
    ax.set_xlabel("")
    ax.legend()

    span = (dates[-1] - dates[0]).days if len(dates) > 1 else 1
    _date_axis(ax, span)


def plot_monthly_average(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Average good-sailing rate by calendar month."""
    daily = compute_daily_target(df, cfg)
    monthly = daily.groupby([d.month for d in daily.index]).mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    palette = [GREEN if v >= cfg["prediction"]["min_good_fraction"] else BLUE
               for v in monthly.values]
    bars = ax.bar(range(len(monthly)), monthly.values, color=palette,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([month_names[m - 1] for m in monthly.index])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(cfg["prediction"]["min_good_fraction"], color=GREY,
               linestyle="--", linewidth=0.8, label="Good-day threshold")
    ax.set_title("Good-sailing rate by month")
    ax.legend()


def plot_wind_speed_dist(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Wind speed distribution during the sailing window, good vs bad days."""
    sc = cfg["sailing"]
    daily = compute_daily_target(df, cfg)
    min_good = cfg["prediction"]["min_good_fraction"]

    good_dates = {d for d, v in daily.items() if v >= min_good}
    bad_dates  = {d for d, v in daily.items() if v <  min_good}

    window  = df.between_time(sc["window_start"], sc["window_end"])
    date_of = window.index.normalize().map(lambda t: t.date())
    good_ws = window[date_of.isin(good_dates)]["wind_speed"].dropna()
    bad_ws  = window[date_of.isin(bad_dates) ]["wind_speed"].dropna()

    bins = np.linspace(0, 25, 35)
    ax.hist(bad_ws,  bins=bins, alpha=0.55, color=RED,   label="Poor day", density=True)
    ax.hist(good_ws, bins=bins, alpha=0.55, color=GREEN, label="Good day", density=True)
    ax.axvspan(sc["wind_speed_min"], sc["wind_speed_max"],
               alpha=0.08, color=GREEN, label="Target range")
    ax.set_xlabel("Wind speed (knots)")
    ax.set_ylabel("Density")
    ax.set_title("Wind speed — good vs poor days")
    ax.legend()


def plot_wind_consistency_dist(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Wind direction consistency (circular std) during the sailing window."""
    sc = cfg["sailing"]
    min_good = cfg["prediction"]["min_good_fraction"]
    daily = compute_daily_target(df, cfg)
    good_dates = {d for d, v in daily.items() if v >= min_good}
    bad_dates  = {d for d, v in daily.items() if v <  min_good}

    window = df.between_time(sc["window_start"], sc["window_end"])
    consistency = (
        window["wind_direction"]
        .rolling("2h", min_periods=3)
        .apply(_circular_std, raw=False)
    )
    date_map = consistency.index.normalize().map(lambda t: t.date())
    good_c = consistency[date_map.isin(good_dates)].dropna()
    bad_c  = consistency[date_map.isin(bad_dates) ].dropna()

    bins = np.linspace(0, 90, 35)
    ax.hist(bad_c,  bins=bins, alpha=0.55, color=RED,   label="Poor day", density=True)
    ax.hist(good_c, bins=bins, alpha=0.55, color=GREEN, label="Good day", density=True)
    ax.axvline(sc["wind_dir_consistency_max"], color=GREY,
               linestyle="--", linewidth=1, label="Threshold")
    ax.set_xlabel("Wind direction circular std (°)")
    ax.set_ylabel("Density")
    ax.set_title("Wind consistency — good vs poor days")
    ax.legend()


def plot_pressure_anomaly(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """28-day pressure anomaly with good-day markers."""
    min_good = cfg["prediction"]["min_good_fraction"]
    daily = compute_daily_target(df, cfg)

    daily_pressure = df["pressure_relative"].resample("D").mean()
    baseline = daily_pressure.rolling("28D", min_periods=7).mean()
    anomaly  = daily_pressure - baseline

    ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax.fill_between(anomaly.index, anomaly.values,
                    where=anomaly.values >= 0, alpha=0.25, color=BLUE,  interpolate=True)
    ax.fill_between(anomaly.index, anomaly.values,
                    where=anomaly.values <  0, alpha=0.25, color=RED,   interpolate=True)
    ax.plot(anomaly.index, anomaly.values, linewidth=0.9, color=BLUE)

    good_dates = [pd.Timestamp(d) for d, v in daily.items() if v >= min_good]
    if good_dates:
        y_top = anomaly.max() * 1.1
        ax.scatter(good_dates, [y_top] * len(good_dates),
                   marker="|", color=GREEN, s=40, linewidths=1,
                   label="Good sailing day", zorder=3)

    ax.set_ylabel("hPa (vs 28-day mean)")
    ax.set_title("Pressure anomaly")
    ax.legend()

    span = (anomaly.index[-1] - anomaly.index[0]).days
    _date_axis(ax, span)


def plot_feature_importance(model_path: str, ax: plt.Axes) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    if not os.path.exists(model_path):
        ax.text(0.5, 0.5, "No trained model found.\nRun model/train.py first.",
                ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_title("Feature importances")
        return

    bundle = joblib.load(model_path)
    importance = (
        pd.Series(bundle["model"].feature_importances_, index=bundle["feature_names"])
        .sort_values()
    )

    colors = []
    for n in importance.index:
        if n in ("sin_doy", "cos_doy"):
            colors.append(ORANGE)
        elif n == "data_density":
            colors.append(GREY)
        else:
            colors.append(BLUE)

    importance.plot(kind="barh", ax=ax, color=colors, edgecolor="none", width=0.7)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Feature importances")
    ax.tick_params(axis="y", labelsize=7.5)

    # Compact legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=BLUE,   label="Weather signal"),
        Patch(color=ORANGE, label="Seasonal"),
        Patch(color=GREY,   label="Data density"),
    ], loc="lower right")


# ---------------------------------------------------------------------------
# Composite figures
# ---------------------------------------------------------------------------

def figure_data(df: pd.DataFrame, cfg: dict, out: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_daily_fraction(df, cfg, axes[0, 0])
    plot_monthly_average(df, cfg, axes[0, 1])
    plot_wind_speed_dist(df, cfg, axes[1, 0])
    plot_wind_consistency_dist(df, cfg, axes[1, 1])
    fig.suptitle("Historical sailing conditions", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def figure_model(df: pd.DataFrame, cfg: dict, model_path: str, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_pressure_anomaly(df, cfg, axes[0])
    plot_feature_importance(model_path, axes[1])
    fig.suptitle("Model diagnostics", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def figure_all(df: pd.DataFrame, cfg: dict, model_path: str, out: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_daily_fraction(df, cfg, axes[0, 0])
    plot_monthly_average(df, cfg, axes[0, 1])
    plot_pressure_anomaly(df, cfg, axes[0, 2])
    plot_wind_speed_dist(df, cfg, axes[1, 0])
    plot_wind_consistency_dist(df, cfg, axes[1, 1])
    plot_feature_importance(model_path, axes[1, 2])
    fig.suptitle("windPredictor — exploration", fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise windPredictor data and model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data",  action="store_true", help="Data overview only (4 panels)")
    group.add_argument("--model", action="store_true", help="Model diagnostics only (2 panels)")
    parser.add_argument("--out",    default=None,           help="Output PNG path")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.toml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    root       = os.path.dirname(os.path.abspath(args.config))
    model_path = os.path.join(root, cfg["paths"]["model_file"])

    df = load_weather_readings()
    if df.empty:
        print("No weather readings found. Run input/stitcher.py first.")
        sys.exit(1)
    print(f"Loaded {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})")

    if args.data:
        out = args.out or os.path.join(root, "exploration_data.png")
        figure_data(df, cfg, out)
    elif args.model:
        out = args.out or os.path.join(root, "exploration_model.png")
        figure_model(df, cfg, model_path, out)
    else:
        out = args.out or os.path.join(root, "exploration.png")
        figure_all(df, cfg, model_path, out)


if __name__ == "__main__":
    main()
