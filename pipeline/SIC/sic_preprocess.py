import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from sic_config import TS_VALUE_COLS, FFILL_LIMITS, SEQ_LEN


def _calc_pf_ratio(df):
    if 'pao2' in df.columns and 'fio2' in df.columns:
        return df['pao2'] / (df['fio2'] / 100).replace(0, np.nan)
    return pd.Series(np.nan, index=df.index)


def _calc_derived(ts):
    BASE_COLS = ['map', 'aptt', 'lactate', 'creatinine', 'bilirubin_total', 'wbc', 'rdw', 'pao2fio2ratio']
    for col in BASE_COLS:
        if col not in ts.columns:
            ts[col] = 0.0
        ts[f'{col}_last']  = ts[col].shift(1).fillna(0.0)
        ts[f'{col}_trend'] = (ts[col] - ts[f'{col}_last']).fillna(0.0)

    ts['map_min']             = ts['map'].expanding().min()
    ts['map_mean']            = ts['map'].expanding().mean()
    ts['aptt_max']            = ts['aptt'].expanding().max() if 'aptt' in ts.columns else 0.0
    ts['lactate_max']         = ts['lactate'].expanding().max() if 'lactate' in ts.columns else 0.0
    ts['creatinine_max']      = ts['creatinine'].expanding().max() if 'creatinine' in ts.columns else 0.0
    ts['bilirubin_total_max'] = ts['bilirubin_total'].expanding().max() if 'bilirubin_total' in ts.columns else 0.0
    ts['wbc_max']             = ts['wbc'].expanding().max() if 'wbc' in ts.columns else 0.0
    ts['wbc_min']             = ts['wbc'].expanding().min() if 'wbc' in ts.columns else 0.0
    ts['rdw_mean']            = ts['rdw'].expanding().mean() if 'rdw' in ts.columns else 0.0
    ts['pf_ratio_min']        = ts['pao2fio2ratio'].expanding().min() if 'pao2fio2ratio' in ts.columns else 0.0
    return ts


def preprocess_timeseries(vital_ts, lab_df, patient_meta):
    onset  = pd.Timestamp(patient_meta['sepsis_onset_time'])
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

    # hours_from_onset
    ts['hours_from_onset'] = [
        (slot - onset).total_seconds() / 3600 for slot in ts['slot']
    ]

    # lab 병합
    lab = lab_df.copy()
    lab['slot'] = pd.to_datetime(lab['charttime']).dt.floor('h')
    lab_cols = [c for c in ['creatinine', 'wbc', 'rdw', 'aptt', 'lactate', 'bilirubin_total'] if c in lab.columns]
    lab_agg = lab.groupby('slot')[lab_cols].mean().reset_index()
    ts = ts.merge(lab_agg, on='slot', how='left')

    # pf_ratio
    ts['pao2fio2ratio'] = _calc_pf_ratio(ts)

    # ffill
    for col, limit in FFILL_LIMITS.items():
        if col in ts.columns:
            ts[col] = ts[col].ffill(limit=limit)

    # mask
    for col in ['map', 'creatinine', 'wbc', 'rdw', 'aptt', 'lactate', 'bilirubin_total']:
        ts[f'{col}_mask'] = ts[col].isna().astype(int) if col in ts.columns else 1

    # 파생 피처
    ts = _calc_derived(ts)

    # 결측 → 0
    for col in TS_VALUE_COLS:
        if col not in ts.columns:
            ts[col] = 0.0
    ts[TS_VALUE_COLS] = ts[TS_VALUE_COLS].fillna(0.0)

    # SEQ_LEN 맞추기
    x = ts[TS_VALUE_COLS].values.astype(np.float32)
    if len(x) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(x), x.shape[1]), dtype=np.float32)
        x = np.vstack([pad, x])
    else:
        x = x[-SEQ_LEN:]

    n_slots = int(ts['map'].notna().sum()) if 'map' in ts.columns else 0

    return torch.tensor(x).unsqueeze(0), n_slots, ts


def preprocess_static(patient_meta, ts_df=None):
    static = {
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

    XGB_TS_COLS = [
        'map_last', 'map_trend', 'aptt_last', 'aptt_trend',
        'lactate_last', 'lactate_trend', 'creatinine_last', 'creatinine_trend',
        'bilirubin_total_last', 'bilirubin_total_trend', 'wbc_last', 'wbc_trend',
        'rdw_last', 'rdw_trend', 'pf_ratio_last', 'pf_ratio_trend',
        'map_min', 'map_mean', 'aptt_max', 'lactate_max', 'creatinine_max',
        'bilirubin_total_max', 'wbc_max', 'wbc_min', 'rdw_mean', 'pf_ratio_min',
        'aptt_mean', 'aptt_std', 'aptt_min',
    ]

    ts_stats = {}
    if ts_df is not None:
        for col in XGB_TS_COLS:
            stat = col.rsplit('_', 1)[1]
            if col in ts_df.columns:
                if stat == 'last':
                    ts_stats[col] = ts_df[col].dropna().iloc[-1] if not ts_df[col].dropna().empty else 0.0
                elif stat == 'mean':
                    ts_stats[col] = ts_df[col].mean()
                elif stat == 'min':
                    ts_stats[col] = ts_df[col].min()
                elif stat == 'max':
                    ts_stats[col] = ts_df[col].max()
                elif stat == 'std':
                    ts_stats[col] = ts_df[col].std()
                elif stat == 'trend':
                    ts_stats[col] = ts_df[col].dropna().iloc[-1] if not ts_df[col].dropna().empty else 0.0
                else:
                    ts_stats[col] = 0.0
            else:
                ts_stats[col] = 0.0

    FEAT_ORDER = list(static.keys()) + XGB_TS_COLS
    row = {**static, **ts_stats}
    x = np.array([row.get(c, 0.0) for c in FEAT_ORDER], dtype=np.float32).reshape(1, -1)
    np.nan_to_num(x, nan=0.0, copy=False)
    return x