"""
ARDS 모델 학습 스크립트
- 데이터: v6_master_win24_h48_conservative (parquet)
- 출력: joblib 아티팩트 (XGBoost + calibrator + features + threshold)

사용법:
    python train.py --data-dir ./datasets/dataset-6/v6_master_win24_h48_conservative \
                    --output-dir ./models/ards
"""
import argparse
import json
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, roc_auc_score,
)

from ards_model import build_xgb_model, calibrate_model

ARTIFACT_FILENAME = "artifact__v6_master_win24_h48_conservative__XGBoost__full.joblib"


def load_splits(data_dir):
    data_dir = Path(data_dir)
    splits = {}
    for sp in ["train", "val", "test"]:
        splits[sp] = pd.read_parquet(data_dir / f"{sp}.parquet")
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return splits, meta


def choose_threshold(y_val, val_prob, lo=0.10, hi=0.80, step=0.05):
    best_thr, best_f1 = 0.30, 0.0
    for thr in np.arange(lo, hi + step, step):
        pred = (val_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def main(data_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits, meta = load_splits(data_dir)
    features = meta["feature_columns"]

    X_train = splits["train"][features]
    y_train = splits["train"]["label"].astype(int)
    X_val = splits["val"][features]
    y_val = splits["val"]["label"].astype(int)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    spw = neg / max(pos, 1.0)

    # 학습
    model = build_xgb_model(scale_pos_weight=spw)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 캘리브레이션
    calibrator = calibrate_model(model, X_val, y_val)

    # threshold 선정
    val_prob = calibrator.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_prob)

    # 아티팩트 저장
    artifact = {
        "base_model": model,
        "calibrator": calibrator,
        "features": features,
        "threshold": threshold,
        "model_info": {
            "dataset": "v6_master_win24_h48_conservative",
            "model": "XGBoost",
            "track": "conservative",
            "window_h": 24,
            "horizon_h": 48,
        }
    }
    save_path = output_dir / ARTIFACT_FILENAME
    joblib.dump(artifact, save_path)
    print(f"아티팩트 저장: {save_path}")
    print(f"  threshold: {threshold}")
    print(f"  val AUROC: {roc_auc_score(y_val, val_prob):.4f}")
    print(f"  val AUPRC: {average_precision_score(y_val, val_prob):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="./models/ards")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
