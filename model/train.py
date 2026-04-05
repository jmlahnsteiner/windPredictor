"""
model/train.py — Train the sailing forecast model and save weights.

Usage:
    python model/train.py             # uses config.toml
    python model/train.py my.toml
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_training_pairs
from utils.config import load_config

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "..", "config.toml")


def train(config_path: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config_path)
    root = os.path.dirname(os.path.abspath(config_path))

    # --- Load data ---
    from input.weather_store import load_weather_readings
    print("Loading all weather readings from database …")
    df = load_weather_readings()
    if df.empty:
        print("ERROR: No weather data found. Run the scraper and stitch first.")
        return
    print(f"Loaded {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})")

    # --- Load NWP data ---
    print("Loading NWP readings from database …")
    from input.nwp_store import load_nwp_readings
    nwp_df = load_nwp_readings()
    if nwp_df.empty:
        print("  [!] No NWP data found — run open_meteo_historical.py to backfill.")
        print("  Proceeding with station features only (nwp_* features will be NaN).")
        nwp_df = None
    else:
        print(f"  NWP readings: {len(nwp_df):,} rows  "
              f"({nwp_df.index.min().date()} → {nwp_df.index.max().date()})")

    # --- Build training pairs ---
    X, y = build_training_pairs(df, cfg, nwp_df=nwp_df)

    if X.empty:
        days = df.index.normalize().nunique()
        print(f"\nNot enough data to build training pairs.")
        print(f"  Have {days} day(s) — need at least 2.")
        print("  Run the scraper to collect more historical data, then re-stitch.")
        return

    print(f"\nTraining pairs : {len(X)}")
    print(f"Good days (y=1): {y.sum()}  /  {len(y)}")
    print(f"Features       : {list(X.columns)}")

    # Fill NaN with column median (robust to missing sensors)
    X = X.fillna(X.median(numeric_only=True))

    # --- Train ---
    mc = cfg["model"]
    clf = RandomForestClassifier(
        n_estimators=mc["n_estimators"],
        max_depth=mc["max_depth"],
        min_samples_leaf=mc["min_samples_leaf"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation only when we have enough samples
    if len(X) >= 10 and y.nunique() > 1:
        n_splits = min(5, y.value_counts().min())
        cv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        print(f"\nCV ROC-AUC : {scores.mean():.3f} ± {scores.std():.3f}  (k={n_splits}, temporal)")
    else:
        print("\n(Skipping cross-validation — too few samples)")

    clf.fit(X, y)

    importance = (
        pd.Series(clf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop feature importances:")
    print(importance.to_string())

    # --- Save ---
    model_path = os.path.join(root, cfg["paths"]["model_file"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_names": list(X.columns),
            "feature_medians": X.median(numeric_only=True).to_dict(),
            "config": cfg,
            "trained_on": str(pd.Timestamp.now()),
            "n_training_samples": len(X),
        },
        model_path,
    )
    print(f"\nSaved: {model_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    train(config_path)
