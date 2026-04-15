import sys, os
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


def predict_aki(vital_ts, lab_df, patient_meta):
    gru_model, xgb_model = get_models()

    # ── 전처리 ───────────────────────────────────────────────
    X_gru = preprocess_gru(vital_ts, lab_df, patient_meta)
    X_xgb = preprocess_xgb(vital_ts, lab_df, patient_meta)

    # ── 추론 ─────────────────────────────────────────────────
    prob_gru = float(gru_model.predict(X_gru, verbose=0).ravel()[0])
    prob_xgb = float(xgb_model.predict_proba(X_xgb)[0, 1])

    # ── 앙상블 (soft voting 0.5 : 0.5) ───────────────────────
    prob_final = 0.5 * prob_gru + 0.5 * prob_xgb

    return {
        'aki': {
            'probability':    round(prob_final, 4),
            'prediction':     int(prob_final >= THRESHOLD),
            'threshold':      THRESHOLD,
            'inference_time': datetime.now(timezone.utc).isoformat(),
            'data_quality': {
                'n_slots':     12,
                'is_reliable': True,
            },
        }
    }