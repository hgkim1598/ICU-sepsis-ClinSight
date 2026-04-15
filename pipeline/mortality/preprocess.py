import numpy as np
import pandas as pd
import torch
from scipy import stats

from config import TS_COLS, MASK_COLS, FEAT_COLS, SEQ_LEN


# ── 헬퍼 함수 ─────────────────────────────────────────────────
def _calc_slope(series):
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return stats.linregress(np.arange(len(s)), s.values).slope

def _calc_last(series):
    s = series.dropna()
    return s.iloc[-1] if len(s) > 0 else np.nan

def _calc_missing_flag(series):
    return int(series.isna().all())

def _calc_diff(series):
    s = series.dropna()
    return (s.iloc[-1] - s.iloc[0]) if len(s) >= 2 else np.nan


# ── 시계열 전처리 ─────────────────────────────────────────────
def preprocess_timeseries(vital_ts, patient_meta):
    slots = pd.date_range(
        start=pd.Timestamp(patient_meta['window_start_vital']).floor('h'),
        end=pd.Timestamp(patient_meta['window_end']).floor('h'),
        freq='1h'
    )
    ts = pd.DataFrame({'slot': slots})
    vital = vital_ts.copy()
    vital['slot'] = pd.to_datetime(vital['charttime']).dt.floor('h')
    agg = vital.groupby('slot')[TS_COLS].mean().reset_index()
    ts  = ts.merge(agg, on='slot', how='left')

    for col in ['heart_rate','mbp','sbp','dbp','resp_rate','spo2']:
        ts[col] = ts[col].ffill(limit=1)
    ts['temperature']   = ts['temperature'].ffill(limit=4)
    ts['gcs']           = ts['gcs'].ffill(limit=6)
    ts['pao2fio2ratio'] = ts['pao2fio2ratio'].ffill(limit=12)

    for col in TS_COLS:
        ts[f'{col}_mask'] = ts[col].isna().astype(int)

    vals  = ts[TS_COLS].values.astype(np.float32)
    masks = ts[MASK_COLS].values.astype(np.float32)
    np.nan_to_num(vals, nan=0.0, copy=False)
    x = np.concatenate([vals, masks], axis=1)

    if len(x) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(x), 18), dtype=np.float32)
        x = np.vstack([pad, x])
    else:
        x = x[-SEQ_LEN:]

    n_vital_slots = int(agg['heart_rate'].notna().sum())

    return torch.tensor(x).unsqueeze(0), n_vital_slots


# ── 정적 피처 전처리 ──────────────────────────────────────────
def preprocess_static(vital_ts, lab_df, patient_meta):
    ws_v = patient_meta['window_start_vital']
    ws_l = patient_meta['window_start_lab']
    we   = patient_meta['window_end']

    v = vital_ts[(vital_ts['charttime'] >= ws_v) & (vital_ts['charttime'] <= we)]
    l = lab_df[(lab_df['charttime'] >= ws_l) & (lab_df['charttime'] <= we)]

    feats   = {}
    imputed = {}

    def set_feat(key, val, series):
        feats[key]   = val
        imputed[key] = bool(series.dropna().empty)

    for col, stats_list in [
        ('heart_rate',  ['last','min','max']),
        ('mbp',         ['last','min','slope']),
        ('sbp',         ['last','min','slope']),
        ('dbp',         ['last','min','slope']),
        ('resp_rate',   ['last','max','slope']),
        ('spo2',        ['last','min']),
        ('temperature', ['min','max']),
    ]:
        s = v[col] if col in v.columns else pd.Series(dtype=float)
        for stat in stats_list:
            if stat == 'last':
                set_feat(f'{col}_last',  _calc_last(s),  s)
            elif stat == 'min':
                set_feat(f'{col}_min',   s.min(),        s)
            elif stat == 'max':
                set_feat(f'{col}_max',   s.max(),        s)
            elif stat == 'slope':
                set_feat(f'{col}_slope', _calc_slope(s), s)

    set_feat('gcs_last', _calc_last(v['gcs']), v['gcs'])
    set_feat('gcs_min',  v['gcs'].min(),       v['gcs'])
    feats['gcs_missing_flag']   = _calc_missing_flag(v['gcs'])
    imputed['gcs_missing_flag'] = False

    for urine_col in ['urine_last','urine_min','urine_diff']:
        feats[urine_col]   = np.nan
        imputed[urine_col] = True

    for col, stats_list in [
        ('lactate',     ['last','max','slope']),
        ('creatinine',  ['last','max','slope']),
        ('bun',         ['last','slope']),
        ('sodium',      ['last','min','max']),
        ('potassium',   ['last','min','max']),
        ('glucose',     ['last','min','max']),
        ('bicarbonate', ['last','min']),
    ]:
        s = l[col] if col in l.columns else pd.Series(dtype=float)
        for stat in stats_list:
            if stat == 'last':
                set_feat(f'{col}_last',  _calc_last(s),  s)
            elif stat == 'min':
                set_feat(f'{col}_min',   s.min(),        s)
            elif stat == 'max':
                set_feat(f'{col}_max',   s.max(),        s)
            elif stat == 'slope':
                set_feat(f'{col}_slope', _calc_slope(s), s)

    s = v['pao2fio2ratio'] if 'pao2fio2ratio' in v.columns else pd.Series(dtype=float)
    set_feat('pao2fio2_last',  _calc_last(s),  s)
    set_feat('pao2fio2_min',   s.min(),        s)
    set_feat('pao2fio2_slope', _calc_slope(s), s)

    for col in ['albumin', 'wbc', 'platelet', 'hemoglobin', 'bilirubin_total']:
        s        = l[col] if col in l.columns else pd.Series(dtype=float)
        is_empty = s.dropna().empty

        if col == 'albumin':
            set_feat('albumin_min', s.min() if not is_empty else np.nan, s)
            feats['albumin_missing_flag']   = _calc_missing_flag(s)
            imputed['albumin_missing_flag'] = False
        elif col == 'wbc':
            set_feat('wbc_last',  _calc_last(s),  s)
            set_feat('wbc_min',   s.min(),        s)
            set_feat('wbc_max',   s.max(),        s)
            set_feat('wbc_slope', _calc_slope(s), s)
        elif col == 'platelet':
            set_feat('platelet_last',  _calc_last(s),  s)
            set_feat('platelet_min',   s.min(),        s)
            set_feat('platelet_slope', _calc_slope(s), s)
        elif col == 'hemoglobin':
            set_feat('hemoglobin_last', _calc_last(s), s)
            set_feat('hemoglobin_min',  s.min(),       s)
            set_feat('hemoglobin_diff', _calc_diff(s), s)
        elif col == 'bilirubin_total':
            set_feat('bilirubin_min', s.min() if not is_empty else np.nan, s)
            set_feat('bilirubin_max', s.max() if not is_empty else np.nan, s)
            feats['bilirubin_missing_flag']   = _calc_missing_flag(s)
            imputed['bilirubin_missing_flag'] = False

    feats['age']    = patient_meta['age']
    feats['gender'] = patient_meta['gender']
    imputed['age']    = False
    imputed['gender'] = False

    x = np.array([feats[c] for c in FEAT_COLS], dtype=np.float32).reshape(1, -1)
    np.nan_to_num(x, nan=0.0, copy=False)

    n_lab_measurements = int(l.shape[0])

    return x, n_lab_measurements, imputed, feats