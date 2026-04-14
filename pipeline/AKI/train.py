from __future__ import annotations

import json
from pathlib import Path

from model import build_ensemble_dataframe, find_best_threshold


BASE_DIR = Path(__file__).resolve().parent

# 네가 실제로 쓰던 파일명 기준
XGB_PATH = BASE_DIR / "eicu_xgb_external_full_predictions.csv"
GRU_PATH = BASE_DIR / "eicu_gru_external_predictions_clean.csv"

OUT_PRED = BASE_DIR / "eicu_weighted_ensemble_predictions.csv"
OUT_METRICS = BASE_DIR / "eicu_weighted_ensemble_metrics.txt"
OUT_CONFIG = BASE_DIR / "weighted_ensemble_config.json"


def main() -> None:
    df = build_ensemble_dataframe(
        xgb_path=XGB_PATH,
        gru_path=GRU_PATH,
        weight_xgb=0.5,
        weight_gru=0.5,
        threshold=0.5,
    )

    best_threshold, metrics = find_best_threshold(
        y_true=df["y_true"].values,
        y_prob=df["ensemble_prob"].values,
    )

    df["pred_label_best"] = (df["ensemble_prob"] >= best_threshold).astype(int)
    df.to_csv(OUT_PRED, index=False)

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write(f"best_threshold={best_threshold:.6f}\n")
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")

    config = {
        "weight_xgb": 0.5,
        "weight_gru": 0.5,
        "best_threshold": best_threshold,
        "xgb_path": str(XGB_PATH.name),
        "gru_path": str(GRU_PATH.name),
        "prediction_output": str(OUT_PRED.name),
        "metrics_output": str(OUT_METRICS.name),
    }
    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Saved:")
    print(f"- {OUT_PRED}")
    print(f"- {OUT_METRICS}")
    print(f"- {OUT_CONFIG}")


if __name__ == "__main__":
    main()