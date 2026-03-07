"""
explore.py — Visualise historical data and trained model.

Reads from data.parquet and model/weights.joblib. All plots are saved
to PNG; nothing is displayed interactively (safe for headless Pi use).

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
import tomllib


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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.features import _anomaly, compute_daily_target

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "config.toml")

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Individual plot functions — each accepts (df, cfg, ax) and draws one panel
# ---------------------------------------------------------------------------


def plot_daily_fraction(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Good-sailing fraction per day over the full dataset."""
    daily = compute_daily_target(df, cfg)
    dates = [pd.Timestamp(d) for d in daily.index]
    ax.fill_between(dates, daily.values, alpha=0.4)
    ax.plot(dates, daily.values, linewidth=0.8)
    # 30-day rolling mean
    s = pd.Series(daily.values, index=dates)
    ax.plot(
        s.rolling("30d").mean(), color="tab:red", linewidth=1.5, label="30-day mean"
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Daily good-sailing fraction")
    ax.set_xlabel("")
    ax.legend(fontsize=8)


def plot_monthly_average(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Average good-sailing rate by calendar month."""
    daily = compute_daily_target(df, cfg)
    monthly = daily.groupby([d.month for d in daily.index]).mean()
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    bars = ax.bar(monthly.index, monthly.values, color=sns.color_palette("muted"))
    ax.set_xticks(monthly.index)
    ax.set_xticklabels([month_names[m - 1] for m in monthly.index], fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Average good-sailing rate by month")


def plot_wind_speed_dist(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Wind speed distribution during the sailing window, good vs bad days."""
    sc = cfg["sailing"]
    daily = compute_daily_target(df, cfg)
    min_good = cfg["prediction"]["min_good_fraction"]

    good_dates = {d for d, v in daily.items() if v >= min_good}
    bad_dates = {d for d, v in daily.items() if v < min_good}

    window = df.between_time(sc["window_start"], sc["window_end"])
    good_ws = window[window.index.normalize().map(lambda t: t.date()).isin(good_dates)][
        "wind_speed"
    ].dropna()
    bad_ws = window[window.index.normalize().map(lambda t: t.date()).isin(bad_dates)][
        "wind_speed"
    ].dropna()

    bins = np.linspace(0, 25, 30)
    ax.hist(bad_ws, bins=bins, alpha=0.6, label="Poor day", density=True)
    ax.hist(good_ws, bins=bins, alpha=0.6, label="Good day", density=True)
    ax.axvline(sc["wind_speed_min"], color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(sc["wind_speed_max"], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Wind speed (knots)")
    ax.set_title("Wind speed — good vs poor sailing days")
    ax.legend(fontsize=8)


def plot_wind_consistency_dist(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """Wind direction consistency (circular std) during sailing window."""
    from model.features import _circular_std

    sc = cfg["sailing"]
    min_good = cfg["prediction"]["min_good_fraction"]
    daily = compute_daily_target(df, cfg)
    good_dates = {d for d, v in daily.items() if v >= min_good}
    bad_dates = {d for d, v in daily.items() if v < min_good}

    window = df.between_time(sc["window_start"], sc["window_end"])
    consistency = (
        window["wind_direction"]
        .rolling("2h", min_periods=3)
        .apply(_circular_std, raw=False)
    )
    date_map = consistency.index.normalize().map(lambda t: t.date())
    good_c = consistency[date_map.isin(good_dates)].dropna()
    bad_c = consistency[date_map.isin(bad_dates)].dropna()

    bins = np.linspace(0, 90, 30)
    ax.hist(bad_c, bins=bins, alpha=0.6, label="Poor day", density=True)
    ax.hist(good_c, bins=bins, alpha=0.6, label="Good day", density=True)
    ax.axvline(
        cfg["sailing"]["wind_dir_consistency_max"],
        color="grey",
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_xlabel("Wind direction circular std (°)")
    ax.set_title("Wind consistency — good vs poor sailing days")
    ax.legend(fontsize=8)


def plot_pressure_anomaly(df: pd.DataFrame, cfg: dict, ax: plt.Axes) -> None:
    """28-day pressure anomaly with good-day markers."""
    min_good = cfg["prediction"]["min_good_fraction"]
    daily = compute_daily_target(df, cfg)

    # Resample pressure to daily mean, then compute 28-day anomaly
    daily_pressure = df["pressure_relative"].resample("D").mean()
    baseline = daily_pressure.rolling("28D", min_periods=7).mean()
    anomaly = daily_pressure - baseline

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.fill_between(anomaly.index, anomaly.values, alpha=0.3)
    ax.plot(anomaly.index, anomaly.values, linewidth=0.8, label="Pressure anomaly")

    # Overlay good days as tick marks at the top
    good_dates = [pd.Timestamp(d) for d, v in daily.items() if v >= min_good]
    if good_dates:
        ax.scatter(
            good_dates,
            [anomaly.max() * 1.05] * len(good_dates),
            marker="|",
            color="tab:green",
            s=30,
            linewidths=0.8,
            label="Good sailing day",
            zorder=3,
        )

    ax.set_ylabel("hPa (departure from 28-day mean)")
    ax.set_title("Pressure anomaly")
    ax.legend(fontsize=8)


def plot_feature_importance(model_path: str, ax: plt.Axes) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    if not os.path.exists(model_path):
        ax.text(
            0.5,
            0.5,
            "No trained model found.\nRun model/train.py first.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Feature importances")
        return

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    names = bundle["feature_names"]

    importance = pd.Series(clf.feature_importances_, index=names).sort_values()
    colors = [
        "tab:red" if n in ("sin_doy", "cos_doy") else "tab:blue"
        for n in importance.index
    ]
    importance.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Feature importances  (red = seasonal)")
    ax.tick_params(axis="y", labelsize=7)


# ---------------------------------------------------------------------------
# Composite figures
# ---------------------------------------------------------------------------


def figure_data(df: pd.DataFrame, cfg: dict, out: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_daily_fraction(df, cfg, axes[0, 0])
    plot_monthly_average(df, cfg, axes[0, 1])
    plot_wind_speed_dist(df, cfg, axes[1, 0])
    plot_wind_consistency_dist(df, cfg, axes[1, 1])
    fig.suptitle("Historical sailing conditions", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def figure_model(df: pd.DataFrame, cfg: dict, model_path: str, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_pressure_anomaly(df, cfg, axes[0])
    plot_feature_importance(model_path, axes[1])
    fig.suptitle("Model diagnostics", fontsize=13, y=1.01)
    fig.tight_layout()
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
    fig.suptitle("windPredictor — exploration", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise windPredictor data and model."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--data", action="store_true", help="Data overview only (4 panels)"
    )
    group.add_argument(
        "--model", action="store_true", help="Model diagnostics only (2 panels)"
    )
    parser.add_argument(
        "--out", default=None, help="Output PNG path (auto-named if omitted)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    root = os.path.dirname(os.path.abspath(args.config))
    parquet_path = os.path.join(root, cfg["paths"]["data_parquet"])
    model_path = os.path.join(root, cfg["paths"]["model_file"])

    if not os.path.exists(parquet_path):
        print(f"No data found at {parquet_path}. Run input/stitcher.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(
        f"Loaded {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})"
    )

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
