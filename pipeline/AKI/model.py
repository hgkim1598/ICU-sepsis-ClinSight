from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def normalize_stay_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_xgb_predictions(xgb_path: str | Path) -> pd.DataFrame:
    xgb_path = Path(xgb_path)
    df = pd.read_csv(xgb_path)

    rename_map = {
        "pred_prob": "xgb_prob",
        "aki_within_48h": "y_true",
    }
    df = df.rename(columns=rename_map)

    required_candidates = {"stay_id", "xgb_prob"}
    missing = [c for c in required_candidates if c not in df.columns]
    if missing:
        raise ValueError(f"XGB prediction file missing required columns: {missing}")

    if "y_true" not in df.columns:
        raise ValueError("XGB prediction file must contain 'y_true' or 'aki_within_48h'.")

    df["stay_id"] = normalize_stay_id(df["stay_id"])
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["xgb_prob"] = pd.to_numeric(df["xgb_prob"], errors="coerce")

    df = df[["stay_id", "y_true", "xgb_prob"]].drop_duplicates(subset=["stay_id"]).copy()
    df = df.dropna(subset=["stay_id", "y_true", "xgb_prob"]).copy()

    return df


def load_gru_predictions(gru_path: str | Path) -> pd.DataFrame:
    gru_path = Path(gru_path)
    df = pd.read_csv(gru_path)

    rename_map = {
        "pred_prob": "gru_prob",
    }
    df = df.rename(columns=rename_map)

    required_candidates = {"stay_id", "gru_prob"}
    missing = [c for c in required_candidates if c not in df.columns]
    if missing:
        raise ValueError(f"GRU prediction file missing required columns: {missing}")

    df["stay_id"] = normalize_stay_id(df["stay_id"])
    df["gru_prob"] = pd.to_numeric(df["gru_prob"], errors="coerce")

    df = df[["stay_id", "gru_prob"]].drop_duplicates(subset=["stay_id"]).copy()
    df = df.dropna(subset=["stay_id", "gru_prob"]).copy()

    return df


def merge_prediction_tables(
    xgb_df: pd.DataFrame,
    gru_df: pd.DataFrame,
) -> pd.DataFrame:
    common_ids = set(xgb_df["stay_id"].dropna().tolist()) & set(gru_df["stay_id"].dropna().tolist())
    if len(common_ids) == 0:
        raise ValueError("No overlapping stay_id values between XGB and GRU predictions.")

    merged = xgb_df[xgb_df["stay_id"].isin(common_ids)].copy()
    merged = merged.merge(gru_df, on="stay_id", how="left")
    merged = merged.dropna(subset=["y_true", "xgb_prob", "gru_prob"]).copy()

    merged["y_true"] = merged["y_true"].astype(int)

    return merged


def weighted_ensemble(
    df: pd.DataFrame,
    weight_xgb: float = 0.5,
    weight_gru: float = 0.5,
) -> pd.DataFrame:
    out = df.copy()
    out["ensemble_prob"] = weight_xgb * out["xgb_prob"] + weight_gru * out["gru_prob"]
    return out


def apply_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out["pred_label"] = (out["ensemble_prob"] >= threshold).astype(int)
    return out


def build_ensemble_dataframe(
    xgb_path: str | Path,
    gru_path: str | Path,
    weight_xgb: float = 0.5,
    weight_gru: float = 0.5,
    threshold: float = 0.5,
) -> pd.DataFrame:
    xgb_df = load_xgb_predictions(xgb_path)
    gru_df = load_gru_predictions(gru_path)

    df = merge_prediction_tables(xgb_df, gru_df)
    df = weighted_ensemble(df, weight_xgb=weight_xgb, weight_gru=weight_gru)
    df = apply_threshold(df, threshold=threshold)

    return df


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }
    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, dict]:
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_f1 = -1.0

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)

    metrics = evaluate_predictions(y_true, y_prob, threshold=best_threshold)
    return best_threshold, metrics