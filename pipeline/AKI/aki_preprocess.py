import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from aki_config import RAW_COLS, SEQ_LEN, N_RAW


def _build_raw_matrix(vital_ts, lab_df, patient_meta):
    """
    최근 SEQ_LEN(12) 시간 슬롯 기준으로 raw 피처 행렬 생성
    Returns: np.ndarray (SEQ_LEN, N_RAW)
    """
    onset  = pd.Timestamp(patient_meta['sepsis_onset_time'])
    intime = pd.Timestamp(patient_meta['intime'])
    window_end   = onset + pd.Timedelta(hours=42)
    window_start = window_end - pd.Timedelta(hours=SEQ_LEN)
    window_start = max(window_start, intime)

    slots = pd.date_range(start=window_start.floor('h'), periods=SEQ_LEN, freq='1h')
    ts = pd.DataFrame({'slot': slots})

    # vital 병합
    vital = vital_ts.copy()
    vital['slot'] = pd.to_datetime(vital['charttime']).dt.floor('h')
    vital_cols = [c for c in ['heart_rate', 'mbp', 'resp_rate', 'temperature', 'spo2', 'glucose_vital']
                  if c in vital.columns]
    vital_agg = vital.groupby('slot')[vital_cols].mean().reset_index()
    ts = ts.merge(vital_agg, on='slot', how='left')

    # lab 병합
    lab = lab_df.copy()
    lab['slot'] = pd.to_datetime(lab['charttime']).dt.floor('h')
    lab_cols = [c for c in ['creatinine', 'bun', 'bicarbonate', 'sodium', 'potassium', 'glucose', 'urine_output']
                if c in lab.columns]
    lab_agg = lab.groupby('slot')[lab_cols].mean().reset_index()
    ts = ts.merge(lab_agg, on='slot', how='left')

    # 컬럼 순서 맞추기
    for col in RAW_COLS:
        if col not in ts.columns:
            ts[col] = np.nan

    return ts[RAW_COLS].values.astype(np.float64)  # (SEQ_LEN, N_RAW)


def preprocess_gru(vital_ts, lab_df, patient_meta):
    """
    GRU 입력: raw + delta + mean + std (mask 없음)
    Returns: np.ndarray (1, SEQ_LEN, N_RAW * 4)
    """
    X_raw = _build_raw_matrix(vital_ts, lab_df, patient_meta)

    # NaN → 0
    X = np.nan_to_num(X_raw, nan=0.0)

    delta = np.diff(X, axis=0)
    delta = np.concatenate([np.zeros((1, N_RAW)), delta], axis=0)

    mean_feat = np.mean(X, axis=0, keepdims=True).repeat(SEQ_LEN, axis=0)
    std_feat  = np.std(X,  axis=0, keepdims=True).repeat(SEQ_LEN, axis=0)

    X_gru = np.concatenate([X, delta, mean_feat, std_feat], axis=1)  # (SEQ_LEN, 52)

    return X_gru[np.newaxis, :, :].astype(np.float32)  # (1, SEQ_LEN, 52)


def preprocess_xgb(vital_ts, lab_df, patient_meta):
    """
    XGBoost 입력: raw + delta + mean + std + mask → flatten
    Returns: np.ndarray (1, SEQ_LEN * N_RAW * 5)
    """
    X_raw = _build_raw_matrix(vital_ts, lab_df, patient_meta)

    mask = np.isnan(X_raw).astype(np.float64)

    # forward fill
    df = pd.DataFrame(X_raw, columns=RAW_COLS)
    df = df.ffill()
    # 남은 NaN → 0 (train median 대체는 학습 시 적용된 값 없으므로 0으로)
    X = df.values

    delta = np.diff(X, axis=0)
    delta = np.concatenate([np.zeros((1, N_RAW)), delta], axis=0)

    mean_feat = np.mean(X, axis=0, keepdims=True).repeat(SEQ_LEN, axis=0)
    std_feat  = np.std(X,  axis=0, keepdims=True).repeat(SEQ_LEN, axis=0)

    X_xgb = np.concatenate([X, delta, mean_feat, std_feat, mask], axis=1)  # (SEQ_LEN, 65)
    X_flat = X_xgb.reshape(1, -1).astype(np.float32)  # (1, 780)

    return X_flat