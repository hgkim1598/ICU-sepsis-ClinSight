import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from sic_config import (
    TS_VALUE_COLS, TS_MASK_COLS, TS_DERIVED_COLS,
    STATIC_COLS, FFILL_LIMITS, SEQ_LEN
)


def _calc_pf_ratio(df):
    """pao2 / (fio2 / 100), 둘 다 없으면 NaN"""
    if 'pao2' in df.columns and 'fio2' in df.columns:
        return df['pao2'] / (df['fio2'] / 100).replace(0, np.nan)
    return pd.Series(np.nan, index=df.index)


def _calc_derived(ts):
    """last, trend, 누적 통계 파생 피처 계산"""
    for col in TS_VALUE_COLS:
        if col not in ts.columns:
            ts[col] = 0.0
        ts[f'{col}_last']  = ts[col].shift(1).fillna(0.0)
        ts[f'{col}_trend'] = (ts[col] - ts[f'{col}_last']).fillna(0.0)

    ts['map_min']            = ts['map'].expanding().min()
    ts['map_mean']           = ts['map'].expanding().mean()
    ts['aptt_max']           = ts['aptt'].expanding().max() if 'aptt' in ts.columns else 0.0
    ts['lactate_max']        = ts['lactate'].expanding().max() if 'lactate' in ts.columns else 0.0
    ts['creatinine_max']     = ts['creatinine'].expanding().max() if 'creatinine' in ts.columns else 0.0
    ts['bilirubin_total_max']= ts['bilirubin_total'].expanding().max() if 'bilirubin_total' in ts.columns else 0.0
    ts['wbc_max']            = ts['wbc'].expanding().max() if 'wbc' in ts.columns else 0.0
    ts['wbc_min']            = ts['wbc'].expanding().min() if 'wbc' in ts.columns else 0.0
    ts['rdw_mean']           = ts['rdw'].expanding().mean() if 'rdw' in ts.columns else 0.0
    ts['pf_ratio_min']       = ts['pf_ratio'].expanding().min() if 'pf_ratio' in ts.columns else 0.0

    return ts


def preprocess_timeseries(vital_ts, lab_df, patient_meta):
    """
    Returns:
        tensor: (1, SEQ_LEN, INPUT_DIM)
        n_slots: int
    """
    onset = pd.Timestamp(patient_meta['sepsis_onset_time'])
    intime = pd.Timestamp(patient_meta['intime'])
    window_start = max(onset - pd.Timedelta(hours=6), intime)
    window_end   = onset + pd.Timedelta(hours=41)

    slots = pd.date_range(start=window_start.floor('h'), end=window_end.floor('h'), freq='1h')
    ts = pd.DataFrame({'slot': slots})

    # vital 병합
    vital = vital_ts.copy()
    vital['slot'] = pd.to_datetime(vital['charttime']).dt.floor('h')
    vital_agg = vital.groupby('slot')[
        [c for c in ['map', 'pao2', 'fio2'] if c in vital.columns]
    ].mean().reset_index()
    ts = ts.merge(vital_agg, on='slot', how='left')

    # lab 병합
    lab = lab_df.copy()
    lab['slot'] = pd.to_datetime(lab['charttime']).dt.floor('h')
    lab_cols = [c for c in ['creatinine', 'wbc', 'rdw', 'aptt', 'lactate', 'bilirubin_total'] if c in lab.columns]
    lab_agg = lab.groupby('slot')[lab_cols].mean().reset_index()
    ts = ts.merge(lab_agg, on='slot', how='left')

    # pf_ratio 계산
    ts['pf_ratio'] = _calc_pf_ratio(ts)

    # ffill
    for col, limit in FFILL_LIMITS.items():
        if col in ts.columns:
            ts[col] = ts[col].ffill(limit=limit)

    # mask 생성 (ffill 후 기준)
    for col in ['map', 'creatinine', 'wbc', 'rdw', 'aptt', 'lactate', 'bilirubin_total']:
        ts[f'{col}_mask'] = ts[col].isna().astype(int) if col in ts.columns else 1

    # 파생 피처
    ts = _calc_derived(ts)

    # 결측 → 0
    all_cols = TS_VALUE_COLS + TS_MASK_COLS + TS_DERIVED_COLS
    for col in all_cols:
        if col not in ts.columns:
            ts[col] = 0.0
    ts[all_cols] = ts[all_cols].fillna(0.0)

    # SEQ_LEN 맞추기
    x = ts[all_cols].values.astype(np.float32)
    if len(x) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(x), x.shape[1]), dtype=np.float32)
        x = np.vstack([pad, x])
    else:
        x = x[-SEQ_LEN:]

    n_slots = int(ts['map'].notna().sum()) if 'map' in ts.columns else 0

    return torch.tensor(x).unsqueeze(0), n_slots


def preprocess_static(patient_meta):
    """
    Returns:
        x: np.ndarray (1, len(STATIC_COLS))
    """
    row = {
        'age':                   patient_meta.get('age', 0),
        'sex_male':               patient_meta.get('gender', 0),
        'flag_liver_failure':     patient_meta.get('flag_liver_failure', 0),
        'flag_ckd':               patient_meta.get('flag_ckd', 0),
        'flag_coagulopathy':      patient_meta.get('flag_coagulopathy', 0),
        'flag_diabetes':          patient_meta.get('flag_diabetes', 0),
        'flag_immunosuppression': patient_meta.get('flag_immunosuppression', 0),
        'flag_chf':               patient_meta.get('flag_chf', 0),
        'flag_septic_shock_hx':   patient_meta.get('flag_septic_shock_hx', 0),
    }
    x = np.array([row[c] for c in STATIC_COLS], dtype=np.float32).reshape(1, -1)
    return x