import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import shap

from datetime import datetime, timezone

from config import THRESHOLD, FEAT_COLS, FEAT_UNITS
from loader import get_models
from preprocess import preprocess_timeseries, preprocess_static
from history import load_latest, save_result, compute_changes

CLINICAL_REFERENCE = {
    'ventilation': {
        'unit': 'binary',
        'usual_range': '0=미사용, 1=기계환기 중',
        'risk_value': None,
    },
    'norepinephrine': {
        'unit': 'mcg/kg/min',
        'usual_range': '0.01–0.25',
        'risk_value': None,
    },
    'dopamine': {
        'unit': 'mcg/kg/min',
        'usual_range': '1–10',
        'risk_value': None,
    },
    'dobutamine': {
        'unit': 'mcg/kg/min',
        'usual_range': '2–20',
        'risk_value': None,
    },
    'epinephrine': {
        'unit': 'mcg/kg/min',
        'usual_range': '0.01–0.5',
        'risk_value': None,
    },
}




def _last_val(df, col):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.iloc[-1]), 4) if len(s) > 0 else None




def _safe_float(val):
    if val is None:
        return None
    f = float(val)
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, 4)

def predict_mortality(
    vital_ts,
    lab_df,
    patient_meta: dict,
    patient_id: str | None = None,
) -> dict:
    bilstm, clf_xgb, lr = get_models()

    # ── 전처리 ───────────────────────────────────────────────
    x_ts, n_vital_slots                          = preprocess_timeseries(vital_ts, patient_meta)
    x_static, n_lab_measurements, imputed, feats = preprocess_static(vital_ts, lab_df, patient_meta)
    x_ts = x_ts.to(next(bilstm.parameters()).device)

    # ── 추론 ─────────────────────────────────────────────────
    with torch.no_grad():
        prob_lstm = torch.sigmoid(bilstm(x_ts)).cpu().numpy()[0]

    prob_xgb   = clf_xgb.predict_proba(x_static)[0, 1]
    S          = np.array([[prob_lstm, prob_xgb]])
    prob_final = float(lr.predict_proba(S)[0, 1])

    # ── SHAP ─────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(clf_xgb)
    shap_values = explainer.shap_values(x_static)[0]
    shap_idx    = {f: i for i, f in enumerate(FEAT_COLS)}

    # ── 이력 로드 + 변화값 계산 ───────────────────────────────
    previous = load_latest(patient_id) if patient_id else None
    changes  = compute_changes(feats, previous)

    # ── feature_values 구성 (age 제외) ───────────────────────
    feature_values = []
    for feat in FEAT_COLS:
        if feat in ( 'age', 'gender'):
            continue
        raw_val = feats.get(feat)
        ch      = changes.get(feat, {})
        feature_values.append({
            'feature':          feat,
            'raw_value':  _safe_float(raw_val),
            'shap_value': _safe_float(shap_values[shap_idx[feat]]),
            'unit':             FEAT_UNITS.get(feat, ''),
            'is_imputed':       imputed.get(feat, False),
            'change': _safe_float(ch.get('change')),
            'change_direction': ch.get('change_direction', 'unknown'),
        })
    top_features = sorted(
        [f for f in feature_values if f['shap_value'] is not None],
        key=lambda x: abs(x['shap_value']),
        reverse=True
    )[:3]


   
    clinical_indicators = {
    feat: {
        'value':     _last_val(vital_ts if feat == 'ventilation' else lab_df, feat),
        'reference': CLINICAL_REFERENCE.get(feat, {}),
    }
    for feat in ['ventilation', 'norepinephrine', 'dopamine', 'dobutamine', 'epinephrine']
      } 
        
    
    result = {
        'mortality': {
            'probability':    round(prob_final, 4),
            'prediction':     int(prob_final >= THRESHOLD),
            'threshold':      THRESHOLD,
            'inference_time': datetime.now(timezone.utc).isoformat(),
            'clinical_indicators': clinical_indicators,
            'data_quality': {
                'n_vital_slots':      n_vital_slots,
                'n_lab_measurements': n_lab_measurements,
                'is_reliable':        n_vital_slots >= 6,
            },
            'feature_values': feature_values,
             'top_features':   top_features,
        }
    }

    if patient_id:
        save_result(patient_id, result)

    return result