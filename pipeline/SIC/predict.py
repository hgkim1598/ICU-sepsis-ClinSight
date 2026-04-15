"""
SIC (Sepsis-Induced Coagulopathy) 예측 추론 모듈
─────────────────────────────────────────────────────────────────────────────
모델  : BiLSTM + XGBoost → Stacking Logistic Regression (5-fold 앙상블)
입력  : vital_df (MAP, PF ratio) + lab_df (lactate, creatinine, bilirubin,
        wbc, rdw, aptt) + patient_meta
출력  : {"sic": {"probability": float, "shap": [{"feature": str, "value": float}]}}

변경 이력
---------
v1 (2026-04):
  - 초기 버전 작성 (mortality / ARDS predict.py 인터페이스 통일)
  - S3 / 로컬 모델 로드 지원
  - scaler.pkl 로드 후 X_seq / X_xgb 스케일링 적용
v2 (2026-04):
  - padding_mask 피처 제거 (input_size 42 → 41)
  - eICU 외부 검증 결과 반영하여 재학습된 모델 대응
"""

import json
import os
import pickle
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb

# ── 환경 설정 ──────────────────────────────────────────────────────────────
S3_BUCKET        = os.getenv("S3_BUCKET",        "say2-1team")
MODEL_PREFIX     = os.getenv("MODEL_PREFIX_SIC",  "Final_model/saved_models/sic")
USE_S3           = os.getenv("USE_S3", "true").lower() == "true"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH_SIC", "./models/sic")

N_FOLDS = 5
SEQ_LEN = 48        # 48 시간 시퀀스
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM 하이퍼파라미터 (config.yaml 기준)
_LSTM_CFG = dict(
    input_size    = 41,    # X_seq.shape[2] — padding_mask 제거 후 42 → 41
    hidden_size   = 128,
    num_layers    = 2,
    dropout       = 0.3,
    bidirectional = True,
)

# ── 피처 정의 ──────────────────────────────────────────────────────────────
# config.yaml features.timeseries 순서와 동일 (41개, padding_mask 제거)
TS_FEATURES = [
    "hours_from_onset",
    "lactate", "creatinine", "bilirubin_total", "wbc", "rdw", "map",
    "map_mask", "creatinine_mask", "wbc_mask", "rdw_mask", "aptt_mask",
    "lactate_mask", "bilirubin_total_mask",
    "pf_ratio",
    "map_last", "map_trend", "aptt_last", "aptt_trend",
    "lactate_last", "lactate_trend", "creatinine_last", "creatinine_trend",
    "bilirubin_total_last", "bilirubin_total_trend",
    "wbc_last", "wbc_trend", "rdw_last", "rdw_trend",
    "pf_ratio_last", "pf_ratio_trend",
    "map_min", "map_mean", "aptt_max", "lactate_max", "creatinine_max",
    "bilirubin_total_max", "wbc_max", "wbc_min", "rdw_mean", "pf_ratio_min",
]

# config.yaml features.static 순서 (9개)
STATIC_FEATURES = [
    "age", "sex_male",
    "flag_liver_failure", "flag_ckd", "flag_coagulopathy",
    "flag_diabetes", "flag_immunosuppression", "flag_chf", "flag_septic_shock_hx",
]

# vital / lab 원시 컬럼 (전처리 시 사용)
_VITAL_COLS = ["map", "pf_ratio"]
_LAB_COLS   = ["lactate", "creatinine", "bilirubin_total", "wbc", "rdw", "aptt"]
_RAW_COLS   = _VITAL_COLS + _LAB_COLS


# ── LSTM 모델 정의 ─────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """config.yaml 의 lstm 섹션과 동일한 BiLSTM 분류기."""

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_size * num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)        # (B, T, out_dim)
        h   = output[:, -1, :]          # last timestep
        h   = self.dropout(h)
        return self.fc(h).squeeze(-1)   # (B,) logits


# ── 모델 전역 캐시 (최초 1회 로드) ────────────────────────────────────────
_lstm_models:    list | None = None
_xgb_models:     list | None = None
_meta_model:     object | None = None
_xgb_feat_names: list | None = None
_scaler_bundle:  dict | None = None


def _load_models() -> tuple:
    """S3 또는 로컬에서 모든 모델 파일과 scaler를 로드합니다."""
    global _xgb_feat_names

    lstm_models, xgb_models = [], []

    if USE_S3:
        s3 = boto3.client("s3")

        for k in range(1, N_FOLDS + 1):
            # LSTM fold_k
            obj = s3.get_object(
                Bucket=S3_BUCKET,
                Key=f"{MODEL_PREFIX}/fold_{k}/best_model.pt",
            )
            state = torch.load(BytesIO(obj["Body"].read()), map_location=device,
                                weights_only=True)
            model = LSTMClassifier(**_LSTM_CFG)
            model.load_state_dict(state)
            model.to(device).eval()
            lstm_models.append(model)

            # XGB fold_k
            obj = s3.get_object(
                Bucket=S3_BUCKET,
                Key=f"{MODEL_PREFIX}/fold_{k}/model.json",
            )
            tmp_path = f"/tmp/sic_xgb_fold_{k}.json"
            with open(tmp_path, "wb") as f:
                f.write(obj["Body"].read())
            clf_xgb = xgb.XGBClassifier()
            clf_xgb.load_model(tmp_path)
            xgb_models.append(clf_xgb)

        # XGB feature names (fold_1 기준)
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{MODEL_PREFIX}/fold_1/feature_names.json",
        )
        _xgb_feat_names = json.loads(obj["Body"].read().decode())

        # Meta LR
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{MODEL_PREFIX}/meta/meta_model.pkl",
        )
        meta_model = pickle.loads(obj["Body"].read())

        # Scaler bundle
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{MODEL_PREFIX}/scaler.pkl",
        )
        scaler_bundle = pickle.loads(obj["Body"].read())

    else:
        base = Path(LOCAL_MODEL_PATH)

        for k in range(1, N_FOLDS + 1):
            state = torch.load(
                base / f"fold_{k}" / "best_model.pt",
                map_location=device,
                weights_only=True,
            )
            model = LSTMClassifier(**_LSTM_CFG)
            model.load_state_dict(state)
            model.to(device).eval()
            lstm_models.append(model)

            clf_xgb = xgb.XGBClassifier()
            clf_xgb.load_model(str(base / f"fold_{k}" / "model.json"))
            xgb_models.append(clf_xgb)

        with open(base / "fold_1" / "feature_names.json") as f:
            _xgb_feat_names = json.load(f)

        with open(base / "meta" / "meta_model.pkl", "rb") as f:
            meta_model = pickle.load(f)

        with open(base / "scaler.pkl", "rb") as f:
            scaler_bundle = pickle.load(f)

    return lstm_models, xgb_models, meta_model, scaler_bundle


def _get_models() -> tuple:
    global _lstm_models, _xgb_models, _meta_model, _scaler_bundle
    if _lstm_models is None:
        print("[SIC] 모델 로드 중...")
        _lstm_models, _xgb_models, _meta_model, _scaler_bundle = _load_models()
        print(
            f"[SIC] 모델 로드 완료 "
            f"(fold × {N_FOLDS}, XGB features: {len(_xgb_feat_names)})"
        )
    return _lstm_models, _xgb_models, _meta_model, _scaler_bundle


# ── 전처리 헬퍼 ────────────────────────────────────────────────────────────
def _build_raw_timeseries(vital_df, lab_df, onset: pd.Timestamp) -> pd.DataFrame:
    """vital_df + lab_df를 48-step 시간 그리드에 정렬합니다.

    Returns
    -------
    df : pd.DataFrame (48행, TS_FEATURES 컬럼 포함)
    """
    window_start = onset - pd.Timedelta(hours=SEQ_LEN - 1)
    slots = pd.date_range(start=window_start, periods=SEQ_LEN, freq="1h")
    df = pd.DataFrame({"charttime": slots})
    df["hours_from_onset"] = (df["charttime"] - onset).dt.total_seconds() / 3600

    # vital 병합 (1시간 평균)
    if vital_df is not None and len(vital_df) > 0:
        v = vital_df.copy()
        v["slot"] = pd.to_datetime(v["charttime"]).dt.floor("h")
        for col in _VITAL_COLS:
            if col in v.columns:
                agg = (
                    v.groupby("slot")[col]
                    .mean()
                    .reset_index()
                    .rename(columns={"slot": "charttime"})
                )
                df = df.merge(agg, on="charttime", how="left")

    # lab 병합 (1시간 평균, forward fill 최대 12h)
    if lab_df is not None and len(lab_df) > 0:
        l = lab_df.copy()
        l["slot"] = pd.to_datetime(l["charttime"]).dt.floor("h")
        for col in _LAB_COLS:
            if col in l.columns:
                agg = (
                    l.groupby("slot")[col]
                    .mean()
                    .reset_index()
                    .rename(columns={"slot": "charttime"})
                )
                df = df.merge(agg, on="charttime", how="left")

    # 없는 컬럼은 NaN으로 초기화
    for col in _RAW_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # forward fill
    df[["map", "pf_ratio"]] = df[["map", "pf_ratio"]].ffill(limit=2)
    for col in _LAB_COLS:
        df[[col]] = df[[col]].ffill(limit=12)

    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """마스크, rolling last/trend/min/max/mean 피처를 추가합니다."""

    # 마스크 (1 = 해당 타임스텝에서 값 없음)
    df["map_mask"]             = df["map"].isna().astype(float)
    df["creatinine_mask"]      = df["creatinine"].isna().astype(float)
    df["wbc_mask"]             = df["wbc"].isna().astype(float)
    df["rdw_mask"]             = df["rdw"].isna().astype(float)
    df["aptt_mask"]            = df["aptt"].isna().astype(float)
    df["lactate_mask"]         = df["lactate"].isna().astype(float)
    df["bilirubin_total_mask"] = df["bilirubin_total"].isna().astype(float)

    # NaN → 0 (마스크로 표시됨)
    for col in _RAW_COLS:
        df[col] = df[col].fillna(0.0)

    # rolling last / trend (각 타임스텝까지의 누적)
    _rolling_cols = [
        "map", "aptt", "lactate", "creatinine",
        "bilirubin_total", "wbc", "rdw", "pf_ratio",
    ]
    for col in _rolling_cols:
        s = df[col].replace(0.0, np.nan)
        df[f"{col}_last"]  = s.ffill().fillna(0.0)
        df[f"{col}_trend"] = (s.ffill() - s.bfill()).fillna(0.0)

    # expanding aggregations (각 타임스텝까지 누적 통계)
    def _exp(col):
        return df[col].replace(0.0, np.nan)

    df["map_min"]             = _exp("map").expanding().min().fillna(0.0)
    df["map_mean"]            = _exp("map").expanding().mean().fillna(0.0)
    df["aptt_max"]            = _exp("aptt").expanding().max().fillna(0.0)
    df["lactate_max"]         = _exp("lactate").expanding().max().fillna(0.0)
    df["creatinine_max"]      = _exp("creatinine").expanding().max().fillna(0.0)
    df["bilirubin_total_max"] = _exp("bilirubin_total").expanding().max().fillna(0.0)
    df["wbc_max"]             = _exp("wbc").expanding().max().fillna(0.0)
    df["wbc_min"]             = _exp("wbc").expanding().min().fillna(0.0)
    df["rdw_mean"]            = _exp("rdw").expanding().mean().fillna(0.0)
    df["pf_ratio_min"]        = _exp("pf_ratio").expanding().min().fillna(0.0)

    # 누락 파생 컬럼 보완
    for col in TS_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def _build_x_seq(df: pd.DataFrame, scaler_bundle: dict) -> torch.Tensor:
    """48 × 41 배열 → scaler 적용 → (1, 48, 41) 텐서.

    mask 피처(인덱스 dl_cont_idx 외)는 스케일링에서 제외됩니다.
    """
    X = df[TS_FEATURES].values.astype(np.float32)   # (48, 41)

    scaler_dl   = scaler_bundle["scaler_dl"]
    cont_idx    = scaler_bundle["dl_cont_idx"]       # 연속형 피처 인덱스 목록

    X[:, cont_idx] = scaler_dl.transform(X[:, cont_idx]).astype(np.float32)
    np.nan_to_num(X, nan=0.0, copy=False)

    return torch.tensor(X).unsqueeze(0).to(device)   # (1, 48, 41)


def _build_x_xgb(
    df: pd.DataFrame,
    patient_meta: dict,
    scaler_bundle: dict,
) -> np.ndarray:
    """static + ts 집계 → XGB 입력 벡터 (1, 173) + scaler 적용."""

    # static 피처
    static = {
        "age":                    float(patient_meta.get("age", 0)),
        "sex_male":               float(patient_meta.get("sex_male",
                                        patient_meta.get("gender", 0))),
        "flag_liver_failure":     float(patient_meta.get("flag_liver_failure", 0)),
        "flag_ckd":               float(patient_meta.get("flag_ckd", 0)),
        "flag_coagulopathy":      float(patient_meta.get("flag_coagulopathy", 0)),
        "flag_diabetes":          float(patient_meta.get("flag_diabetes", 0)),
        "flag_immunosuppression": float(patient_meta.get("flag_immunosuppression", 0)),
        "flag_chf":               float(patient_meta.get("flag_chf", 0)),
        "flag_septic_shock_hx":   float(patient_meta.get("flag_septic_shock_hx", 0)),
    }

    # ts 집계 (48 스텝 × {mean, std, min, max})
    ts_df   = df[TS_FEATURES]
    agg_df  = ts_df.agg(["mean", "std", "min", "max"])   # (4, 41)
    ts_agg  = {
        f"{feat}_{stat}": float(agg_df.loc[stat, feat])
        for feat in TS_FEATURES
        for stat in ["mean", "std", "min", "max"]
    }

    row = {**static, **ts_agg}

    # feature_names.json 순서에 맞게 정렬
    x_raw = np.array(
        [row.get(f, 0.0) for f in _xgb_feat_names],
        dtype=np.float32,
    ).reshape(1, -1)
    np.nan_to_num(x_raw, nan=0.0, copy=False)

    # scaler_xgb 적용 (xgb_feat_cols 위치만)
    scaler_xgb   = scaler_bundle["scaler_xgb"]
    xgb_feat_cols = scaler_bundle["xgb_feat_cols"]  # 스케일링 대상 컬럼명 목록

    # 스케일링 대상 인덱스 (feature_names.json 기준)
    feat_idx = {f: i for i, f in enumerate(_xgb_feat_names)}
    scale_idx = [feat_idx[f] for f in xgb_feat_cols if f in feat_idx]

    x_scale = x_raw[:, scale_idx]
    x_raw[:, scale_idx] = scaler_xgb.transform(x_scale).astype(np.float32)
    np.nan_to_num(x_raw, nan=0.0, copy=False)

    return x_raw   # (1, 173)


# ── 메인 추론 함수 ─────────────────────────────────────────────────────────
def predict_sic(vital_df: pd.DataFrame, lab_df: pd.DataFrame, patient_meta: dict) -> dict:
    """패혈증 유발 응고장애(SIC) 발생 확률 예측.

    Parameters
    ----------
    vital_df : pd.DataFrame
        컬럼: charttime, map (평균동맥압 mmHg), pf_ratio (PaO2/FiO2)
    lab_df : pd.DataFrame
        컬럼: charttime, lactate, creatinine, bilirubin_total, wbc, rdw, aptt
    patient_meta : dict
        필수:
          age (int), onset_time 또는 sepsis_onset_time (datetime)
          sex_male (1=남, 0=여) 또는 gender
        선택 (comorbidity flags, 없으면 0으로 처리):
          flag_liver_failure, flag_ckd, flag_coagulopathy, flag_diabetes,
          flag_immunosuppression, flag_chf, flag_septic_shock_hx

    Returns
    -------
    dict :
        {
            "sic": {
                "probability": 0.412,
                "shap": [
                    {"feature": "lactate_max_mean", "value": 0.0821},
                    ...
                ]
            }
        }
    """
    # onset_time 키 호환 (mortality 인터페이스와 동일)
    if "sepsis_onset_time" in patient_meta and "onset_time" not in patient_meta:
        patient_meta = {**patient_meta, "onset_time": patient_meta["sepsis_onset_time"]}

    onset = pd.Timestamp(patient_meta["onset_time"])

    lstm_models, xgb_models, meta_model, scaler_bundle = _get_models()

    # ── 전처리 ────────────────────────────────────────────────────────────
    df = _build_raw_timeseries(vital_df, lab_df, onset)
    df = _add_derived_features(df)

    x_seq = _build_x_seq(df, scaler_bundle)           # (1, 48, 41) tensor
    x_xgb = _build_x_xgb(df, patient_meta, scaler_bundle)  # (1, 173) ndarray

    # ── LSTM 앙상블 예측 (5-fold 평균) ───────────────────────────────────
    p_lstm = 0.0
    for model in lstm_models:
        with torch.no_grad():
            p_lstm += float(torch.sigmoid(model(x_seq)).cpu().numpy()[0])
    p_lstm /= N_FOLDS

    # ── XGB 앙상블 예측 (5-fold 평균) ────────────────────────────────────
    p_xgb = 0.0
    for clf in xgb_models:
        p_xgb += float(clf.predict_proba(x_xgb)[0, 1])
    p_xgb /= N_FOLDS

    # ── Meta LR 최종 예측 ─────────────────────────────────────────────
    S          = np.array([[p_lstm, p_xgb]], dtype=np.float32)
    prob_final = float(meta_model.predict_proba(S)[0, 1])

    # ── SHAP (XGB fold_1 기준) ────────────────────────────────────────
    explainer   = shap.TreeExplainer(xgb_models[0])
    shap_values = explainer.shap_values(x_xgb)[0]
    shap_list   = sorted(
        [
            {"feature": feat, "value": round(float(val), 4)}
            for feat, val in zip(_xgb_feat_names, shap_values)
        ],
        key=lambda d: abs(d["value"]),
        reverse=True,
    )

    return {
        "sic": {
            "probability": round(prob_final, 4),
            "shap": shap_list,
        }
    }


# ── 배포용 모델 파일 생성 도우미 ───────────────────────────────────────────
def prepare_artifacts_for_deploy(
    checkpoint_dir: str,
    scaler_pkl_path: str,
    output_dir: str = "./models/sic",
) -> None:
    """학습 완료 후 S3/로컬 배포를 위한 모델 파일을 output_dir에 복사합니다.

    Parameters
    ----------
    checkpoint_dir : str
        models/lstm_stacking_lr/checkpoints/ 경로
    scaler_pkl_path : str
        data/dataset/scaler.pkl 경로
    output_dir : str
        배포용 모델 파일 저장 경로
    """
    import shutil

    src  = Path(checkpoint_dir)
    dst  = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for k in range(1, N_FOLDS + 1):
        fold_src = src / f"fold_{k}"
        fold_dst = dst / f"fold_{k}"
        fold_dst.mkdir(exist_ok=True)
        for fname in ["best_model.pt", "model.json", "feature_names.json"]:
            shutil.copy2(fold_src / fname, fold_dst / fname)

    meta_dst = dst / "meta"
    meta_dst.mkdir(exist_ok=True)
    shutil.copy2(src / "meta" / "meta_model.pkl", meta_dst / "meta_model.pkl")
    shutil.copy2(scaler_pkl_path, dst / "scaler.pkl")

    print(f"[SIC] 배포용 파일 복사 완료 → {dst}")
    print(f"  fold_1~{N_FOLDS}/best_model.pt, model.json, feature_names.json")
    print(f"  meta/meta_model.pkl")
    print(f"  scaler.pkl")
