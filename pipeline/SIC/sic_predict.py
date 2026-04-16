import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import shap
from datetime import datetime, timezone

from sic_config import THRESHOLD, STATIC_COLS
from sic_loader import get_models
from sic_preprocess import preprocess_timeseries, preprocess_static

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

def _calc_sic_risk_value(feat, value):
    if value is None:
        return None
    if feat == 'platelet':
        return 150 <= value <= 450  # K/uL 단위
    if feat == 'inr':
        return 0.8 <= value <= 1.3
    return None

SIC_CLINICAL_REFERENCE = {
    'platelet': {'unit': 'K/uL',  'usual_range': '150–450', 'risk_value': None},
    'inr':      {'unit': '',      'usual_range': '0.8–1.3', 'risk_value': None},
}


def predict_sic(vital_ts, lab_df, patient_meta):
    bilstm, clf_xgb, lr = get_models()

    # ── 전처리 ───────────────────────────────────────────────
    x_ts, n_slots, ts_df = preprocess_timeseries(vital_ts, lab_df, patient_meta)
    x_static              = preprocess_static(patient_meta, ts_df=ts_df)
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

    shap_list = sorted(
        [
            {'feature': feat, 'shap_value': _safe_float(float(val))}
            for feat, val in zip(STATIC_COLS, shap_values)
        ],
        key=lambda d: abs(d['shap_value'] or 0),
        reverse=True
    )

    top_features = shap_list[:3]

    clinical_indicators = {
        feat: {
            'value': _last_val(lab_df, feat),
            'reference': {
                **SIC_CLINICAL_REFERENCE[feat],
                'risk_value': _calc_sic_risk_value(
                    feat,
                    _last_val(lab_df, feat)
                ),
            }
        }
        for feat in SIC_CLINICAL_REFERENCE
    }

    return {
        'sic': {
            'probability':    round(prob_final, 4),
            'prediction':     int(prob_final >= THRESHOLD),
            'threshold':      THRESHOLD,
            'clinical_indicators': clinical_indicators,
            'inference_time': datetime.now(timezone.utc).isoformat(),
            'data_quality': {
                'n_slots':   n_slots,
                'is_reliable': n_slots >= 6,
            },
            'shap':         shap_list,
            'top_features': top_features,
        }
    }