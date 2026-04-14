# predict.py
import os
import boto3
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import shap
from io import BytesIO
from scipy import stats
from datetime import datetime

# ── 설정 ─────────────────────────────────────────────────────
S3_BUCKET        = os.getenv('S3_BUCKET', 'say2-1team')
MODEL_PREFIX     = os.getenv('MODEL_PREFIX', 'Final_model/saved_models')
USE_S3           = os.getenv('USE_S3', 'true').lower() == 'true'
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './models')
THRESHOLD        = 0.21

TS_COLS   = ['heart_rate','mbp','sbp','dbp','resp_rate','spo2','temperature','gcs','pao2fio2ratio']
MASK_COLS = [f'{c}_mask' for c in TS_COLS]
FEAT_COLS = [
    'heart_rate_last','heart_rate_min','heart_rate_max',
    'mbp_last','mbp_min','mbp_slope',
    'sbp_last','sbp_min','sbp_slope',
    'dbp_last','dbp_min','dbp_slope',
    'resp_rate_last','resp_rate_max','resp_rate_slope',
    'spo2_last','spo2_min',
    'temperature_min','temperature_max',
    'gcs_last','gcs_min','gcs_missing_flag',
    'urine_last','urine_min','urine_diff',
    'lactate_last','lactate_max','lactate_slope',
    'creatinine_last','creatinine_max','creatinine_slope',
    'bun_last','bun_slope',
    'sodium_last','sodium_min','sodium_max',
    'potassium_last','potassium_min','potassium_max',
    'glucose_last','glucose_min','glucose_max',
    'bicarbonate_last','bicarbonate_min',
    'pao2fio2_last','pao2fio2_min','pao2fio2_slope',
    'albumin_min','albumin_missing_flag',
    'wbc_last','wbc_min','wbc_max','wbc_slope',
    'platelet_last','platelet_min','platelet_slope',
    'hemoglobin_last','hemoglobin_min','hemoglobin_diff',
    'bilirubin_min','bilirubin_max','bilirubin_missing_flag',
    'age','gender'
]
SEQ_LEN = 48
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── BiLSTM 정의 ───────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.classifier(h).squeeze(-1)


# ── 모델 로드 ─────────────────────────────────────────────────
def _load_models():
    if USE_S3:
        s3 = boto3.client('s3')

        # BiLSTM (기존 모델)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/bilstm_best.pt')
        state = torch.load(BytesIO(obj['Body'].read()), map_location=device)

        # XGBoost (기존 모델)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/xgb_stacking.json')
        with open('/tmp/xgb_stacking.json', 'wb') as f:
            f.write(obj['Body'].read())

        # Stacking LR (OOF 버전)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/OOF/stacking_lr_oof.pkl')
        with open('/tmp/stacking_lr_oof.pkl', 'wb') as f:
            f.write(obj['Body'].read())

    else:
        state = torch.load(f'{LOCAL_MODEL_PATH}/bilstm_best.pt', map_location=device)
        import shutil
        shutil.copy(f'{LOCAL_MODEL_PATH}/xgb_stacking.json',   '/tmp/xgb_stacking.json')
        shutil.copy(f'{LOCAL_MODEL_PATH}/OOF/stacking_lr_oof.pkl', '/tmp/stacking_lr_oof.pkl')

    bilstm = BiLSTM().to(device)
    bilstm.load_state_dict(state)
    bilstm.eval()

    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model('/tmp/xgb_stacking.json')

    lr = joblib.load('/tmp/stacking_lr_oof.pkl')

    return bilstm, clf_xgb, lr

# 모델 전역 캐싱
_bilstm, _clf_xgb, _lr = None, None, None

def _get_models():
    global _bilstm, _clf_xgb, _lr
    if _bilstm is None:
        print("모델 로드 중...")
        _bilstm, _clf_xgb, _lr = _load_models()
        print("모델 로드 완료")
    return _bilstm, _clf_xgb, _lr


# ── 전처리 함수 ───────────────────────────────────────────────
def _calc_slope(series):
    s = series.dropna()
    if len(s) < 2: return np.nan
    return stats.linregress(np.arange(len(s)), s.values).slope

def _calc_last(series):
    s = series.dropna()
    return s.iloc[-1] if len(s) > 0 else np.nan

def _calc_missing_flag(series):
    return int(series.isna().all())

def _calc_diff(series):
    s = series.dropna()
    return (s.iloc[-1] - s.iloc[0]) if len(s) >= 2 else np.nan

def _preprocess_timeseries(vital_ts, patient_meta):
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

    return torch.tensor(x).unsqueeze(0)

def _preprocess_static(vital_ts, lab_df, patient_meta):
    ws_v = patient_meta['window_start_vital']
    ws_l = patient_meta['window_start_lab']
    we   = patient_meta['window_end']

    v = vital_ts[(vital_ts['charttime'] >= ws_v) & (vital_ts['charttime'] <= we)]
    l = lab_df[(lab_df['charttime'] >= ws_l) & (lab_df['charttime'] <= we)]

    feats = {}
    for col, stats_list in [
        ('heart_rate', ['last','min','max']),
        ('mbp',        ['last','min','slope']),
        ('sbp',        ['last','min','slope']),
        ('dbp',        ['last','min','slope']),
        ('resp_rate',  ['last','max','slope']),
        ('spo2',       ['last','min']),
        ('temperature',['min','max']),
    ]:
        s = v[col] if col in v.columns else pd.Series(dtype=float)
        for stat in stats_list:
            if stat == 'last':    feats[f'{col}_last']  = _calc_last(s)
            elif stat == 'min':   feats[f'{col}_min']   = s.min()
            elif stat == 'max':   feats[f'{col}_max']   = s.max()
            elif stat == 'slope': feats[f'{col}_slope'] = _calc_slope(s)

    feats['gcs_last']         = _calc_last(v['gcs'])
    feats['gcs_min']          = v['gcs'].min()
    feats['gcs_missing_flag'] = _calc_missing_flag(v['gcs'])
    feats['urine_last']       = np.nan
    feats['urine_min']        = np.nan
    feats['urine_diff']       = np.nan

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
            if stat == 'last':    feats[f'{col}_last']  = _calc_last(s)
            elif stat == 'min':   feats[f'{col}_min']   = s.min()
            elif stat == 'max':   feats[f'{col}_max']   = s.max()
            elif stat == 'slope': feats[f'{col}_slope'] = _calc_slope(s)

    feats['pao2fio2_last']          = _calc_last(v['pao2fio2ratio'])
    feats['pao2fio2_min']           = v['pao2fio2ratio'].min()
    feats['pao2fio2_slope']         = _calc_slope(v['pao2fio2ratio'])
    feats['albumin_min']            = l['albumin'].min()           if 'albumin'       in l.columns else np.nan
    feats['albumin_missing_flag']   = _calc_missing_flag(l['albumin']) if 'albumin'   in l.columns else 1
    feats['wbc_last']               = _calc_last(l['wbc'])         if 'wbc'           in l.columns else np.nan
    feats['wbc_min']                = l['wbc'].min()               if 'wbc'           in l.columns else np.nan
    feats['wbc_max']                = l['wbc'].max()               if 'wbc'           in l.columns else np.nan
    feats['wbc_slope']              = _calc_slope(l['wbc'])        if 'wbc'           in l.columns else np.nan
    feats['platelet_last']          = _calc_last(l['platelet'])    if 'platelet'      in l.columns else np.nan
    feats['platelet_min']           = l['platelet'].min()          if 'platelet'      in l.columns else np.nan
    feats['platelet_slope']         = _calc_slope(l['platelet'])   if 'platelet'      in l.columns else np.nan
    feats['hemoglobin_last']        = _calc_last(l['hemoglobin'])  if 'hemoglobin'    in l.columns else np.nan
    feats['hemoglobin_min']         = l['hemoglobin'].min()        if 'hemoglobin'    in l.columns else np.nan
    feats['hemoglobin_diff']        = _calc_diff(l['hemoglobin'])  if 'hemoglobin'    in l.columns else np.nan
    feats['bilirubin_min']          = l['bilirubin_total'].min()   if 'bilirubin_total' in l.columns else np.nan
    feats['bilirubin_max']          = l['bilirubin_total'].max()   if 'bilirubin_total' in l.columns else np.nan
    feats['bilirubin_missing_flag'] = _calc_missing_flag(l['bilirubin_total']) if 'bilirubin_total' in l.columns else 1
    feats['age']                    = patient_meta['age']
    feats['gender']                 = patient_meta['gender']

    x = np.array([feats[c] for c in FEAT_COLS], dtype=np.float32).reshape(1, -1)
    np.nan_to_num(x, nan=0.0, copy=False)
    return x


# ── 메인 추론 함수 ────────────────────────────────────────────
def predict_mortality(vital_ts, lab_df, patient_meta):
    """
    패혈증 환자 사망률 예측

    Parameters
    ----------
    vital_ts : pd.DataFrame
        컬럼: charttime, heart_rate, mbp, sbp, dbp, resp_rate,
               spo2, temperature, gcs, pao2fio2ratio
    lab_df : pd.DataFrame
        컬럼: charttime, lactate, creatinine, bun, sodium, potassium,
               glucose, bicarbonate, albumin, wbc, platelet,
               hemoglobin, bilirubin_total
    patient_meta : dict
        keys: age, gender, intime, sepsis_onset_time,
              window_start_vital, window_start_lab, window_end

    Returns
    -------
    dict : { mortality: { probability, prediction, threshold, shap } }
    """
    bilstm, clf_xgb, lr = _get_models()

    x_ts     = _preprocess_timeseries(vital_ts, patient_meta).to(device)
    x_static = _preprocess_static(vital_ts, lab_df, patient_meta)

    with torch.no_grad():
        prob_lstm = torch.sigmoid(bilstm(x_ts)).cpu().numpy()[0]

    prob_xgb   = clf_xgb.predict_proba(x_static)[0, 1]
    S          = np.array([[prob_lstm, prob_xgb]])
    prob_final = lr.predict_proba(S)[0, 1]

    explainer   = shap.TreeExplainer(clf_xgb)
    shap_values = explainer.shap_values(x_static)[0]
    shap_list   = [
        {'feature': feat, 'value': round(float(val), 4)}
        for feat, val in zip(FEAT_COLS, shap_values)
    ]

    return {
        'mortality': {
            'probability': round(float(prob_final), 4),
            'prediction':  int(prob_final >= THRESHOLD),
            'threshold':   THRESHOLD,
            'shap':        shap_list
        }
    }


# ── 테스트 실행 ───────────────────────────────────────────────
if __name__ == '__main__':
    import json
    from datetime import datetime, timedelta

    intime       = datetime(2024, 1, 1, 8, 0)
    sepsis_onset = datetime(2024, 1, 1, 14, 0)

    patient_meta = {
        'age': 68,
        'gender': 1,
        'intime': intime,
        'sepsis_onset_time': sepsis_onset,
        'window_start_vital': max(sepsis_onset - timedelta(hours=6), intime),
        'window_start_lab':   sepsis_onset - timedelta(hours=6),
        'window_end':         sepsis_onset + timedelta(hours=42),
    }

    timestamps = pd.date_range(
        start=patient_meta['window_start_vital'],
        end=patient_meta['window_end'], freq='1h'
    )
    n = len(timestamps)
    np.random.seed(42)

    vital_ts = pd.DataFrame({
        'charttime':     timestamps,
        'heart_rate':    np.random.normal(95, 10, n).clip(60, 140),
        'mbp':           np.random.normal(65, 8, n).clip(45, 100),
        'sbp':           np.random.normal(105, 12, n).clip(70, 160),
        'dbp':           np.random.normal(60, 8, n).clip(40, 100),
        'resp_rate':     np.random.normal(22, 4, n).clip(10, 35),
        'spo2':          np.random.normal(94, 3, n).clip(80, 100),
        'temperature':   np.random.normal(38.2, 0.5, n).clip(36, 40),
        'gcs':           np.random.choice([13, 14, 15], n).astype(float),
        'pao2fio2ratio': np.random.normal(220, 50, n).clip(100, 400),
    })

    lab_times = sorted(np.random.choice(
        pd.date_range(patient_meta['window_start_lab'],
                      patient_meta['window_end'], freq='2h').tolist(),
        size=12, replace=False
    ))
    lab_df = pd.DataFrame({
        'charttime':       lab_times,
        'lactate':         np.random.normal(3.2, 1.0, 12).clip(0.5, 8.0),
        'creatinine':      np.random.normal(1.8, 0.5, 12).clip(0.5, 5.0),
        'bun':             np.random.normal(28, 8, 12).clip(5, 60),
        'sodium':          np.random.normal(138, 4, 12).clip(125, 150),
        'potassium':       np.random.normal(4.1, 0.5, 12).clip(3.0, 6.0),
        'glucose':         np.random.normal(145, 30, 12).clip(70, 300),
        'bicarbonate':     np.random.normal(20, 3, 12).clip(12, 30),
        'albumin':         np.random.normal(2.8, 0.4, 12).clip(1.5, 4.5),
        'wbc':             np.random.normal(14, 4, 12).clip(2, 30),
        'platelet':        np.random.normal(180, 60, 12).clip(50, 400),
        'hemoglobin':      np.random.normal(9.5, 1.5, 12).clip(6, 14),
        'bilirubin_total': np.random.normal(1.8, 0.8, 12).clip(0.3, 8.0),
    })

    result = predict_mortality(vital_ts, lab_df, patient_meta)
    print(json.dumps(result, indent=2))