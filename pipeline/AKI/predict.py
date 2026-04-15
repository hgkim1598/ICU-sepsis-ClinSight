from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to aki_seq_X_v7.npy")
    parser.add_argument("--output_path", type=str, default="final_prediction.csv")
    parser.add_argument("--threshold", type=float, default=0.5)

    # local fallback
    parser.add_argument("--gru_model_path", type=str, default="gru_v_final.h5")
    parser.add_argument("--xgb_model_path", type=str, default="xgb_v8_final.pkl")

    # optional local JSON output
    parser.add_argument("--json_output_path", type=str, default="aki_result_final.json")
    return parser.parse_args()


def load_s3_bytes(bucket: str, key: str) -> bytes:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def load_gru_model(local_path: str, bucket: str | None, key: str | None) -> tf.keras.Model:
    if bucket and key:
        raw = load_s3_bytes(bucket, key)
        tmp_path = "/tmp/gru_v_final.h5"
        with open(tmp_path, "wb") as f:
            f.write(raw)
        return tf.keras.models.load_model(tmp_path)

    return tf.keras.models.load_model(local_path)


def load_xgb_model(local_path: str, bucket: str | None, key: str | None):
    if bucket and key:
        raw = load_s3_bytes(bucket, key)
        return joblib.load(io.BytesIO(raw))

    return joblib.load(local_path)


def build_common_features(X_raw: np.ndarray):
    # X_raw: (N, T, F)
    mask = np.isnan(X_raw).astype(float)
    X = np.nan_to_num(X_raw, nan=0.0)

    delta = np.diff(X, axis=1)
    delta = np.concatenate(
        [np.zeros((X.shape[0], 1, X.shape[2])), delta],
        axis=1,
    )

    mean_feat = np.mean(X, axis=1, keepdims=True)
    mean_feat = np.repeat(mean_feat, X.shape[1], axis=1)

    std_feat = np.std(X, axis=1, keepdims=True)
    std_feat = np.repeat(std_feat, X.shape[1], axis=1)

    return X, delta, mean_feat, std_feat, mask


def build_gru_input(X_raw: np.ndarray) -> np.ndarray:
    X, delta, mean_feat, std_feat, _ = build_common_features(X_raw)
    X_gru = np.concatenate([X, delta, mean_feat, std_feat], axis=2)  # 52 features
    return X_gru


def build_xgb_input(X_raw: np.ndarray) -> np.ndarray:
    X, delta, mean_feat, std_feat, mask = build_common_features(X_raw)
    X_xgb = np.concatenate([X, delta, mean_feat, std_feat, mask], axis=2)  # 65 features
    X_flat = X_xgb.reshape(X_xgb.shape[0], -1)
    return X_flat


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent

    input_path = Path(args.input_path)
    if not input_path.is_absolute():
        input_path = base_dir / input_path

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    json_output_path = Path(args.json_output_path)
    if not json_output_path.is_absolute():
        json_output_path = base_dir / json_output_path

    # S3 settings
    use_s3 = os.getenv("USE_S3", "false").lower() == "true"
    bucket = os.getenv("S3_BUCKET")
    model_prefix = os.getenv("MODEL_PREFIX", "").strip("/")

    gru_key = f"{model_prefix}/gru_v_final.h5" if use_s3 and bucket and model_prefix else None
    xgb_key = f"{model_prefix}/xgb_v8_final.pkl" if use_s3 and bucket and model_prefix else None

    gru_local = str((base_dir / args.gru_model_path).resolve())
    xgb_local = str((base_dir / args.xgb_model_path).resolve())

    gru_model = load_gru_model(
        local_path=gru_local,
        bucket=bucket if use_s3 else None,
        key=gru_key,
    )
    xgb_model = load_xgb_model(
        local_path=xgb_local,
        bucket=bucket if use_s3 else None,
        key=xgb_key,
    )

    print("모델 로드 완료")

    X_raw = np.load(input_path)
    print(f"데이터 로드 완료: {X_raw.shape}")

    X_gru = build_gru_input(X_raw)
    X_xgb = build_xgb_input(X_raw)

    print(f"GRU input: {X_gru.shape}")
    print(f"XGB input: {X_xgb.shape}")

    gru_prob = gru_model.predict(X_gru, verbose=0).ravel()
    xgb_prob = xgb_model.predict_proba(X_xgb)[:, 1]

    final_prob = 0.5 * gru_prob + 0.5 * xgb_prob
    final_pred = (final_prob >= args.threshold).astype(int)

    df = pd.DataFrame({
        "sample_id": np.arange(len(final_prob)),
        "gru_prob": gru_prob,
        "xgb_prob": xgb_prob,
        "ensemble_prob": final_prob,
        "pred": final_pred,
    })
    df.to_csv(output_path, index=False)

    first_result = {
        "aki": {
            "probability": float(final_prob[0]),
            "prediction": int(final_pred[0]),
        }
    }
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(first_result, f, indent=2, ensure_ascii=False)

    print("Saved:")
    print(f"- {output_path}")
    print(f"- {json_output_path}")


if __name__ == "__main__":
    main()