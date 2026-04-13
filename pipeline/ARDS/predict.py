# predict_ards.py
"""
ARDS 조기예측 추론 모듈
- 모델: XGBoost (calibrated via Platt Scaling)
- 대표 조합: master sepsis cohort + 24h window + 48h horizon + conservative feature
- 입력: onset 이전 24h 내 활력징후 + 혈액검사 + 환자 기본정보
- 출력: {"ards": {"probability": float, "shap": [{"feature": str, "value": float}]}}

변경 이력
---------
v2 (2025-06):
  - 인터페이스를 mortality predict.py와 통일 (vital_df + lab_df 2개로 통합)
    → bg 컬럼(lactate, ph, bicarbonate)은 lab_df 안에서 자동 분리
  - patient_meta 키 호환: gender / gender_bin 모두 수용
  - patient_meta 키 호환: onset_time / sepsis_onset_time 모두 수용
  - threshold 키 호환: threshold / threshold_from_val 모두 수용
  - XGBoost는 NaN을 자체 처리하므로 별도 imputation 불필요 (의도된 설계)
"""

import os
import boto3
import joblib
import numpy as np
import pandas as pd
import shap
from io import BytesIO

# ── 설정 ─────────────────────────────────────────────────────
S3_BUCKET        = os.getenv('S3_BUCKET', 'say2-1team')
MODEL_PREFIX     = os.getenv('MODEL_PREFIX', 'Final_model/saved_models/ards')
USE_S3           = os.getenv('USE_S3', 'true').lower() == 'true'
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './models/ards')

ARTIFACT_FILENAME = 'artifact__v6_master_win24_h48_conservative__XGBoost__full.joblib'

# 관측 윈도우: onset 이전 24시간
WINDOW_H = 24

# ── 피처 정의 ─────────────────────────────────────────────────
# conservative track: label-adjacent 변수(po2, fio2, pf_ratio, peep) 제외
STAT_RULES = {
    "spo2":          ["last", "mean", "trend", "min"],
    "resp_rate":     ["last", "mean", "trend", "max"],
    "heart_rate":    ["last", "mean", "trend", "max"],
    "mbp":           ["last", "trend", "min"],
    "sbp":           ["last", "trend", "min"],
    "temperature":   ["last", "trend", "max"],
    "lactate":       ["last", "trend", "max", "missing"],
    "ph":            ["last", "trend", "min", "missing"],
    "bicarbonate":   ["last", "trend", "min", "missing"],
    "creatinine":    ["last", "trend"],
    "bun":           ["last", "trend"],
    "wbc":           ["last", "trend"],
    "platelet":      ["last", "trend"],
}

# 피처 → 원본 컬럼명 매핑
COL_MAP = {
    "spo2": "spo2", "resp_rate": "resp_rate", "heart_rate": "heart_rate",
    "mbp": "mbp", "sbp": "sbp", "temperature": "temperature",
    "lactate": "lactate", "ph": "ph", "bicarbonate": "bicarbonate",
    "creatinine": "creatinine", "bun": "bun", "wbc": "wbc", "platelet": "platelet",
}

# 피처 → 데이터 소스 구분 (vital / bg / lab)
SOURCE_MAP = {
    "spo2": "vital", "resp_rate": "vital", "heart_rate": "vital",
    "mbp": "vital", "sbp": "vital", "temperature": "vital",
    "lactate": "bg", "ph": "bg", "bicarbonate": "bg",
    "creatinine": "lab", "bun": "lab", "wbc": "lab", "platelet": "lab",
}

# 최종 피처 컬럼 순서 (age, gender_bin + 41개 summary stats = 43개)
FEAT_COLS = ["age", "gender_bin"]
for _feat_name, _stats in STAT_RULES.items():
    for _stat in _stats:
        FEAT_COLS.append(f"{_feat_name}_{_stat}")

# bg(혈액가스) 컬럼 목록 — lab_df에서 자동 분리할 때 사용
_BG_COLS = {"lactate", "ph", "bicarbonate"}


# ── 모델 로드 ─────────────────────────────────────────────────
_artifact = None

def _load_artifact():
    if USE_S3:
        s3 = boto3.client('s3')
        key = f'{MODEL_PREFIX}/{ARTIFACT_FILENAME}'
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        artifact = joblib.load(BytesIO(obj['Body'].read()))
    else:
        path = os.path.join(LOCAL_MODEL_PATH, ARTIFACT_FILENAME)
        artifact = joblib.load(path)
    return artifact

def _get_artifact():
    global _artifact
    if _artifact is None:
        print("[ARDS] 모델 로드 중...")
        _artifact = _load_artifact()
        print(f"[ARDS] 모델 로드 완료 (피처 {len(_artifact['features'])}개)")
    return _artifact


# ── 전처리 함수 ───────────────────────────────────────────────
def _extract_stats(values, stat_list):
    """시계열 값 배열에서 summary statistics를 추출한다."""
    out = {stat: np.nan for stat in stat_list}
    vals = pd.Series(values).dropna()

    if len(vals) == 0:
        if "missing" in stat_list:
            out["missing"] = 1.0
        return out

    if "last" in stat_list:
        out["last"] = float(vals.iloc[-1])
    if "mean" in stat_list:
        out["mean"] = float(vals.mean())
    if "trend" in stat_list:
        out["trend"] = float(vals.iloc[-1] - vals.iloc[0]) if len(vals) > 1 else 0.0
    if "min" in stat_list:
        out["min"] = float(vals.min())
    if "max" in stat_list:
        out["max"] = float(vals.max())
    if "missing" in stat_list:
        out["missing"] = 0.0

    return out


def _resolve_onset(patient_meta):
    """patient_meta에서 onset 시점을 가져온다 (키 이름 호환)."""
    for key in ['onset_time', 'sepsis_onset_time']:
        if key in patient_meta and patient_meta[key] is not None:
            return pd.Timestamp(patient_meta[key])
    raise KeyError("patient_meta에 'onset_time' 또는 'sepsis_onset_time' 키가 필요합니다.")


def _resolve_gender(patient_meta):
    """patient_meta에서 성별 값을 가져온다 (키 이름 호환)."""
    # gender_bin(0=F, 1=M) 우선, 없으면 gender 사용
    if 'gender_bin' in patient_meta:
        return patient_meta['gender_bin']
    if 'gender' in patient_meta:
        return patient_meta['gender']
    return np.nan


def _split_bg_from_lab(lab_df):
    """
    lab_df에서 bg(혈액가스) 컬럼을 분리한다.
    파이프라인에서 vital_df + lab_df 2개만 넘겨줘도 내부에서 처리.

    Returns: (bg_df, pure_lab_df)
    """
    bg_cols_present = [c for c in _BG_COLS if c in lab_df.columns]
    lab_only_cols = [c for c in lab_df.columns if c not in _BG_COLS or c == 'charttime']

    if bg_cols_present:
        bg_df = lab_df[['charttime'] + bg_cols_present].copy()
    else:
        bg_df = pd.DataFrame(columns=['charttime'])

    pure_lab_df = lab_df[lab_only_cols].copy()
    return bg_df, pure_lab_df


def preprocess(vital_df, lab_df, patient_meta):
    """
    환자 1명의 원시 데이터를 모델 입력 벡터(1×43)로 변환한다.
    lab_df 안에 bg 컬럼(lactate, ph, bicarbonate)이 있으면 자동 분리한다.

    Parameters
    ----------
    vital_df : pd.DataFrame
        컬럼: charttime, spo2, resp_rate, heart_rate, mbp, sbp, temperature
    lab_df : pd.DataFrame
        컬럼: charttime, creatinine, bun, wbc, platelet
        (+ 선택적으로 lactate, ph, bicarbonate 포함 가능)
    patient_meta : dict
        필수 keys: age, onset_time(또는 sepsis_onset_time)
        성별: gender_bin(또는 gender)

    Returns
    -------
    np.ndarray : shape (1, 43)
    """
    onset = _resolve_onset(patient_meta)
    win_start = onset - pd.Timedelta(hours=WINDOW_H)
    gender_val = _resolve_gender(patient_meta)

    # lab_df에서 bg 컬럼 자동 분리
    bg_df, pure_lab_df = _split_bg_from_lab(lab_df)

    # 윈도우 내 데이터 필터링
    source_dfs = {
        "vital": vital_df[(vital_df['charttime'] >= win_start) & (vital_df['charttime'] < onset)],
        "bg":    bg_df[(bg_df['charttime'] >= win_start) & (bg_df['charttime'] < onset)] if len(bg_df) > 0 else bg_df,
        "lab":   pure_lab_df[(pure_lab_df['charttime'] >= win_start) & (pure_lab_df['charttime'] < onset)],
    }

    feats = {
        "age": patient_meta.get('age', np.nan),
        "gender_bin": gender_val,
    }

    for feat_name, stat_list in STAT_RULES.items():
        source_key = SOURCE_MAP[feat_name]
        source_col = COL_MAP[feat_name]
        df = source_dfs[source_key]

        if source_col in df.columns:
            values = df.sort_values('charttime')[source_col].values
        else:
            values = np.array([])

        stats = _extract_stats(values, stat_list)
        for stat_name, val in stats.items():
            feats[f"{feat_name}_{stat_name}"] = val

    # XGBoost는 NaN을 자체 처리하므로 별도 imputation 불필요
    x = np.array([feats.get(c, np.nan) for c in FEAT_COLS], dtype=np.float32).reshape(1, -1)
    return x


# ── 메인 추론 함수 ────────────────────────────────────────────
def predict_ards(vital_df, lab_df, patient_meta):
    """
    패혈증 환자의 48시간 내 ARDS 발생 확률 예측

    인터페이스는 mortality predict.py와 동일: (vital_df, lab_df, patient_meta)
    lab_df 안에 bg 컬럼(lactate, ph, bicarbonate)이 포함되어 있으면 자동 분리.

    Parameters
    ----------
    vital_df : pd.DataFrame
        컬럼: charttime, spo2, resp_rate, heart_rate, mbp, sbp, temperature
    lab_df : pd.DataFrame
        컬럼: charttime, creatinine, bun, wbc, platelet
        (+ 선택적: lactate, ph, bicarbonate)
    patient_meta : dict
        필수: age, onset_time(또는 sepsis_onset_time)
        성별: gender_bin(또는 gender)

    Returns
    -------
    dict :
        {
            "ards": {
                "probability": 0.342,
                "threshold": 0.30,
                "prediction": 1,
                "shap": [
                    {"feature": "lactate_missing", "value": 0.0821},
                    ...
                ]
            }
        }
    """
    artifact = _get_artifact()
    base_model = artifact['base_model']
    calibrator = artifact['calibrator']
    # threshold 키 호환: 노트북에서 "threshold_from_val"로 저장된 경우에도 정상 동작
    threshold  = artifact.get('threshold', artifact.get('threshold_from_val', 0.30))
    features   = artifact['features']

    # 전처리
    x = preprocess(vital_df, lab_df, patient_meta)

    # Calibrated 확률 산출
    prob = float(calibrator.predict_proba(x)[0, 1])

    # SHAP 계산 (base_model 기준 — TreeExplainer 사용)
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(x)[0]
    shap_list = sorted(
        [
            {"feature": feat, "value": round(float(val), 4)}
            for feat, val in zip(FEAT_COLS, shap_values)
        ],
        key=lambda d: abs(d["value"]),
        reverse=True,
    )

    return {
        "ards": {
            "probability": round(prob, 4),
            "threshold": threshold,
            "prediction": int(prob >= threshold),
            "shap": shap_list,
        }
    }


# ── 모델 파일 생성 도우미 ──────────────────────────────────────
def save_artifact_for_deploy(base_model, calibrator, features, threshold=0.30, save_path=None):
    """
    학습 노트북에서 이 함수를 호출하여 배포용 .joblib 파일을 생성한다.
    """
    artifact = {
        "base_model": base_model,
        "calibrator": calibrator,
        "features": features,
        "threshold": threshold,
        "model_info": {
            "dataset": "v6_master_win24_h48_conservative",
            "model": "XGBoost",
            "track": "conservative",
            "window_h": WINDOW_H,
            "horizon_h": 48,
        }
    }
    if save_path is None:
        save_path = ARTIFACT_FILENAME
    joblib.dump(artifact, save_path)
    print(f"[ARDS] 아티팩트 저장 완료: {save_path}")
    return save_path
