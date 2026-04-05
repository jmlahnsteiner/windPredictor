"""
model/train_gru.py — Train and evaluate the GRU sailing forecast model.

Compares GRU against the NWP-enriched RF on identical temporal folds.
Writes evaluation results to docs/gru_eval.md.

Usage:
    python model/train_gru.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from input.nwp_store import load_nwp_readings
from input.weather_store import load_weather_readings
from model.features import build_training_pairs
from model.features_sequence import build_sequence_training_pairs
from model.gru_model import SailingGRU
from utils.config import load_config

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "..", "config.toml")
WEIGHTS_PATH = os.path.join(_HERE, "weights_gru.pt")
REPORT_PATH  = os.path.join(_HERE, "..", "docs", "gru_eval.md")


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_fold(
    seqs_tr: np.ndarray,
    ctxs_tr: np.ndarray,
    labs_tr: np.ndarray,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> SailingGRU:
    model = SailingGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = _class_weights(labs_tr)

    seq_t = torch.tensor(seqs_tr, dtype=torch.float32)
    ctx_t = torch.tensor(ctxs_tr, dtype=torch.float32)
    lab_t = torch.tensor(labs_tr, dtype=torch.float32)

    n = len(seq_t)
    best_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            out = model(seq_t[idx], ctx_t[idx]).squeeze(1)
            weights = torch.where(lab_t[idx] == 1, pos_weight, torch.ones(1))
            loss = (nn.BCELoss(reduction="none")(out, lab_t[idx]) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model: SailingGRU, seqs: np.ndarray, ctxs: np.ndarray,
             labs: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        probs = model(
            torch.tensor(seqs, dtype=torch.float32),
            torch.tensor(ctxs, dtype=torch.float32),
        ).squeeze(1).numpy()

    preds = (probs >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(labs, probs) if len(set(labs)) > 1 else float("nan")
    }
    for name, fn in [("precision", precision_score), ("recall", recall_score),
                     ("f1", f1_score)]:
        try:
            metrics[name] = fn(labs, preds, zero_division=0)
        except Exception:
            metrics[name] = float("nan")
    return metrics


def evaluate_rf(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    cfg = load_config(DEFAULT_CONFIG)
    mc = cfg["model"]
    clf = RandomForestClassifier(
        n_estimators=mc["n_estimators"],
        max_depth=mc["max_depth"],
        min_samples_leaf=mc["min_samples_leaf"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    X_filled = X.fillna(X.median(numeric_only=True))
    cv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for tr, va in cv.split(X_filled):
        clf.fit(X_filled.iloc[tr], y.iloc[tr])
        probs = clf.predict_proba(X_filled.iloc[va])[:, 1]
        preds = (probs >= 0.5).astype(int)
        y_va = y.iloc[va].to_numpy()
        m = {"roc_auc": roc_auc_score(y_va, probs) if len(set(y_va)) > 1 else float("nan")}
        for name, fn in [("precision", precision_score), ("recall", recall_score),
                         ("f1", f1_score)]:
            m[name] = fn(y_va, preds, zero_division=0)
        fold_metrics.append(m)

    return {k: float(np.nanmean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]}


def write_report(gru_metrics: dict, rf_metrics: dict, n_samples: int) -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        "# GRU vs RF Evaluation Report\n",
        f"N training samples: {n_samples}  |  CV: 5-fold TimeSeriesSplit\n\n",
        "| Metric | RF (NWP-enriched) | GRU |",
        "|--------|-------------------|-----|",
    ]
    for k in ["roc_auc", "precision", "recall", "f1"]:
        rf_v  = rf_metrics.get(k, float("nan"))
        gru_v = gru_metrics.get(k, float("nan"))
        lines.append(f"| {k} | {rf_v:.3f} | {gru_v:.3f} |")

    verdict = (
        "**GRU shows ≥3% ROC-AUC improvement — candidate for production promotion.**"
        if gru_metrics.get("roc_auc", 0) - rf_metrics.get("roc_auc", 0) >= 0.03
        else "**RF remains the recommended production model.**"
    )
    lines += ["", verdict, ""]
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


def main():
    cfg = load_config(DEFAULT_CONFIG)

    print("Loading station data …")
    df = load_weather_readings()
    print(f"  {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})")

    print("Loading NWP data …")
    nwp_df = load_nwp_readings()
    if nwp_df.empty:
        print("  [!] No NWP data — context vector will be zeros")
        nwp_df = None
    else:
        print(f"  {len(nwp_df):,} NWP rows")

    print("\nBuilding sequence training pairs …")
    sequences, contexts, labels = build_sequence_training_pairs(df, cfg, nwp_df=nwp_df)
    print(f"  Sequences: {sequences.shape}  Labels: {labels.shape}  "
          f"Good: {labels.sum()}/{len(labels)}")

    if len(sequences) < 10:
        print("ERROR: not enough training pairs")
        sys.exit(1)

    n_splits = min(5, int(labels.sum()))
    cv = TimeSeriesSplit(n_splits=n_splits)
    print(f"\nGRU temporal CV ({n_splits} folds) …")
    gru_fold_metrics = []
    for fold, (tr, va) in enumerate(cv.split(sequences)):
        print(f"  Fold {fold+1}/{n_splits}  train={len(tr)}  val={len(va)}", flush=True)
        model = train_fold(sequences[tr], contexts[tr], labels[tr])
        m = evaluate(model, sequences[va], contexts[va], labels[va])
        gru_fold_metrics.append(m)
        print(f"    ROC-AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}")

    gru_metrics = {k: float(np.nanmean([m[k] for m in gru_fold_metrics]))
                   for k in gru_fold_metrics[0]}
    print(f"\nGRU mean ROC-AUC: {gru_metrics['roc_auc']:.3f}")

    print("\nBuilding RF training pairs for comparison …")
    X, y = build_training_pairs(df, cfg, nwp_df=nwp_df)
    print(f"  RF pairs: {len(X)}")
    rf_metrics = evaluate_rf(X, y, n_splits=n_splits)
    print(f"RF mean ROC-AUC: {rf_metrics['roc_auc']:.3f}")

    print("\nTraining final GRU on all data …")
    final_model = train_fold(sequences, contexts, labels, epochs=200, patience=20)
    torch.save(final_model.state_dict(), WEIGHTS_PATH)
    print(f"Saved GRU weights: {WEIGHTS_PATH}")

    write_report(gru_metrics, rf_metrics, n_samples=len(sequences))

    print("\nSummary:")
    for k in ["roc_auc", "precision", "recall", "f1"]:
        print(f"  {k:12s}  RF={rf_metrics[k]:.3f}  GRU={gru_metrics[k]:.3f}")


if __name__ == "__main__":
    main()
