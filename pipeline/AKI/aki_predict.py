import sys, os
import shap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datetime import datetime, timezone

from aki_config import THRESHOLD
from aki_loader import get_models
from aki_preprocess import preprocess_gru, preprocess_xgb


def _safe_float(val):
    if val is None:
        return None
    f = float(val)
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, 4)

def _last_val(df, col):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.iloc[-1]), 4) if len(s) > 0 else None


def _calc_aki_risk_value(feat, value, gender=None):
    if value is None:
        return None
    if feat == 'lactate':
        return 0.5 <= value <= 2.0
    if feat == 'spo2':
        return 95 <= value <= 100
    if feat == 'gcs':
        return value == 15
    if feat == 'sbp':
        return 90 <= value <= 120
    if feat == 'wbc':
        return 4.0 <= value <= 11.0  # K/uL
    if feat == 'hemoglobin':
        if gender == 1:  # 남성
            return 13 <= value <= 17
        else:  # 여성
            return 12 <= value <= 15
    return None


AKI_CLINICAL_REFERENCE = {
    'lactate':    {'unit': 'mmol/L', 'usual_range': '0.5–2.0',  'risk_value': None},
    'spo2':       {'unit': '%',      'usual_range': '95–100',   'risk_value': None},
    'gcs':        {'unit': 'score',  'usual_range': '15',       'risk_value': None},
    'sbp':        {'unit': 'mmHg',   'usual_range': '90–120',   'risk_value': None},
    'wbc':        {'unit': 'K/uL',   'usual_range': '4–11',     'risk_value': None},
    'hemoglobin': {'unit': 'g/dL',   'usual_range': '남 13–17 / 여 12–15', 'risk_value': None},
}

def predict_aki(vital_ts, lab_df, patient_meta):
    gru_model, xgb_model = get_models()

    # ── 전처리 ───────────────────────────────────────────────
    X_gru = preprocess_gru(vital_ts, lab_df, patient_meta)
    X_xgb = preprocess_xgb(vital_ts, lab_df, patient_meta)

    # ── 추론 ─────────────────────────────────────────────────
    prob_gru = float(gru_model.predict(X_gru, verbose=0).ravel()[0])
    prob_xgb = float(xgb_model.predict_proba(X_xgb)[0, 1])

    # ── SHAP ─────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_xgb)[0]  # (780,)

    # 피처명: RAW_COLS × 5 (raw/delta/mean/std/mask) × SEQ_LEN
    from aki_config import RAW_COLS, SEQ_LEN
    feat_names = []
    for suffix in ['raw', 'delta', 'mean', 'std', 'mask']:
        for col in RAW_COLS:
            for t in range(SEQ_LEN):
                feat_names.append(f'{col}_{suffix}_t{t}')

    shap_list = sorted(
        [{'feature': f, 'shap_value': round(float(v), 4)}
         for f, v in zip(feat_names, shap_values)],
        key=lambda d: abs(d['shap_value']),
        reverse=True
    )
    top_features = shap_list[:3]
    # ── 앙상블 (soft voting 0.5 : 0.5) ───────────────────────
    prob_final = 0.5 * prob_gru + 0.5 * prob_xgb
    gender = patient_meta.get('gender', None)
    clinical_indicators = {
        feat: {
            'value': _last_val(vital_ts if feat in ['spo2', 'sbp', 'gcs'] else lab_df, feat),
            'reference': {
                **AKI_CLINICAL_REFERENCE[feat],
                'risk_value': _calc_aki_risk_value(
                    feat,
                    _last_val(vital_ts if feat in ['spo2', 'sbp', 'gcs'] else lab_df, feat),
                    gender
                ),
            }
        }
        for feat in AKI_CLINICAL_REFERENCE
    }
    return {
        'aki': {
            'probability':    round(prob_final, 4),
            'prediction':     int(prob_final >= THRESHOLD),
            'threshold':      THRESHOLD,
            'clinical_indicators': clinical_indicators,
            'inference_time': datetime.now(timezone.utc).isoformat(),
            'data_quality': {
                'n_slots':     12,
                'is_reliable': True,
            },
            'shap':         shap_list,
            'top_features': top_features,
        }
    }