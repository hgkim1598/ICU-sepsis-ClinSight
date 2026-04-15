from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent

XGB_PATH = BASE_DIR / "eicu_xgb_external_full_predictions.csv"
GRU_PATH = BASE_DIR / "eicu_gru_external_predictions_clean.csv"

OUT_PATH = BASE_DIR / "eicu_weighted_ensemble_predictions.csv"
OUT_METRICS = BASE_DIR / "eicu_weighted_ensemble_metrics.txt"


def normalize_stay_id(series: pd.Series) -> pd.Series:
    # 문자열/정수 섞임 문제 방지
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    if len(y_true) == 0:
        raise ValueError("평가할 샘플이 0개입니다. merge 결과가 비어 있습니다.")
    if len(np.unique(y_true)) < 2:
        raise ValueError(f"y_true 클래스가 하나뿐입니다: {np.unique(y_true)}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[max(best_idx - 1, 0)]) if len(thresholds) > 0 else 0.5
    pred = (y_prob >= best_threshold).astype(int)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "best_f1": float(np.nanmax(f1_scores)),
        "best_threshold": float(best_threshold),
        "precision_at_best_f1": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_best_f1": float(recall_score(y_true, pred, zero_division=0)),
    }


def main():
    print("[1/5] 예측 파일 로드")
    xgb = pd.read_csv(XGB_PATH)
    gru = pd.read_csv(GRU_PATH)

    print("xgb shape:", xgb.shape)
    print("gru shape:", gru.shape)
    print("xgb cols :", xgb.columns.tolist())
    print("gru cols :", gru.columns.tolist())

    print("\n[2/5] 컬럼명 / 타입 정리")

    # XGB
    xgb = xgb.rename(columns={
        "pred_prob": "xgb_prob",
        "aki_within_48h": "y_true",
    })

    # GRU
    gru = gru.rename(columns={
        "pred_prob": "gru_prob",
    })

    if "stay_id" not in xgb.columns or "stay_id" not in gru.columns:
        raise ValueError("두 파일 모두 stay_id 컬럼이 필요함")

    xgb["stay_id"] = normalize_stay_id(xgb["stay_id"])
    gru["stay_id"] = normalize_stay_id(gru["stay_id"])

    if "y_true" not in xgb.columns:
        raise ValueError("XGB 파일에 y_true 또는 aki_within_48h 컬럼이 필요함")

    xgb["y_true"] = pd.to_numeric(xgb["y_true"], errors="coerce")
    xgb["xgb_prob"] = pd.to_numeric(xgb["xgb_prob"], errors="coerce")
    gru["gru_prob"] = pd.to_numeric(gru["gru_prob"], errors="coerce")

    xgb = xgb[["stay_id", "y_true", "xgb_prob"]].drop_duplicates(subset=["stay_id"]).copy()
    gru = gru[["stay_id", "gru_prob"]].drop_duplicates(subset=["stay_id"]).copy()

    # stay_id 결측 제거
    xgb = xgb.dropna(subset=["stay_id", "y_true", "xgb_prob"]).copy()
    gru = gru.dropna(subset=["stay_id", "gru_prob"]).copy()

    print("xgb unique stays:", xgb["stay_id"].nunique())
    print("gru unique stays:", gru["stay_id"].nunique())

    common = set(xgb["stay_id"].dropna().tolist()) & set(gru["stay_id"].dropna().tolist())
    print("common stays:", len(common))

    if len(common) == 0:
        # 디버깅용 샘플 출력
        print("xgb stay_id sample:", xgb["stay_id"].head().tolist())
        print("gru stay_id sample:", gru["stay_id"].head().tolist())
        raise ValueError("XGB와 GRU 사이에 겹치는 stay_id가 0개입니다.")

    print("\n[3/5] merge + ensemble")
    df = xgb[xgb["stay_id"].isin(common)].copy()
    df = df.merge(gru, on="stay_id", how="left")

    df = df.dropna(subset=["y_true", "xgb_prob", "gru_prob"]).copy()
    df["y_true"] = df["y_true"].astype(int)

    print("merged shape:", df.shape)
    print("merged positive rate:", df["y_true"].mean())

    if len(df) == 0:
        raise ValueError("merge 후 유효한 샘플이 0개입니다.")

    # 0.5 + 0.5 weighted ensemble
    df["ensemble_prob"] = 0.5 * df["xgb_prob"] + 0.5 * df["gru_prob"]

    print("\n[4/5] 평가")
    metrics = evaluate_binary(df["y_true"].values, df["ensemble_prob"].values)

    print("\n[5/5] 저장")
    df.to_csv(OUT_PATH, index=False)

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(metrics)
    print("Saved:", OUT_PATH)
    print("Saved:", OUT_METRICS)


if __name__ == "__main__":
    main()