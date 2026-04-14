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
        if feat == 'age':
            continue
        raw_val = feats.get(feat)
        ch      = changes.get(feat, {})
        feature_values.append({
            'feature':          feat,
            'shap_value':       round(float(shap_values[shap_idx[feat]]), 4),
            'raw_value':        round(float(raw_val), 4) if raw_val is not None else None,
            'unit':             FEAT_UNITS.get(feat, ''),
            'is_imputed':       imputed.get(feat, False),
            'change':           ch.get('change'),
            'change_direction': ch.get('change_direction', 'unknown'),
        })

    result = {
        'mortality': {
            'probability':    round(prob_final, 4),
            'prediction':     int(prob_final >= THRESHOLD),
            'threshold':      THRESHOLD,
            'inference_time': datetime.now(timezone.utc).isoformat(),
            'data_quality': {
                'n_vital_slots':      n_vital_slots,
                'n_lab_measurements': n_lab_measurements,
                'is_reliable':        n_vital_slots >= 6,
            },
            'feature_values': feature_values,
        }
    }

    if patient_id:
        save_result(patient_id, result)

    return result