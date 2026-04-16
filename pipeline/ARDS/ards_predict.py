# predict_ards.py
"""
ARDS 조기예측 추론 모듈
- 모델: XGBoost (calibrated via Platt Scaling)
- 대표 조합: master sepsis cohort + 24h window + 48h horizon + conservative feature
- 입력: onset 이전 24h 내 활력징후 + 혈액검사 + 환자 기본정보
- 출력: mortality predict.py와 동일 스펙
        {"ards": {"probability", "prediction", "threshold", "inference_time",
                  "data_quality", "feature_values", "top_features"}}

변경 이력
---------
v2 (2025-06):
  - 인터페이스를 mortality predict.py와 통일 (vital_ts + lab_df 2개로 통합)
    → bg 컬럼(lactate, ph, bicarbonate)은 lab_df 안에서 자동 분리
  - patient_meta 키 호환: gender / gender_bin 모두 수용
  - patient_meta 키 호환: onset_time / sepsis_onset_time 모두 수용
  - threshold 키 호환: threshold / threshold_from_val 모두 수용
  - XGBoost는 NaN을 자체 처리하므로 별도 imputation 불필요 (의도된 설계)
"""

import sys, os
ARDS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARDS_DIR)

from datetime import datetime, timezone

import joblib
import numpy as np
import shap

from ards_config import ARTIFACT_FILENAME, FEAT_COLS, WINDOW_H
from ards_loader import _get_artifact
from ards_preprocess import preprocess

ARDS_CLINICAL_REFERENCE = {
    'po2':       {'unit': 'mmHg',     'usual_range': '75–100',  'risk_value': None},
    'fio2_bg':   {'unit': 'fraction', 'usual_range': '0.21–1.0', 'risk_value': None},
    'pao2fio2ratio': {'unit': 'mmHg', 'usual_range': '>300',    'risk_value': None},
    'peep_feat': {'unit': 'cmH₂O',   'usual_range': '5–15',    'risk_value': None},
}

def _last_val(df, col):
    """DataFrame 컬럼의 마지막 non-null 값. 없으면 None. (mortality _last_val과 동일)"""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.iloc[-1]), 4) if len(s) > 0 else None


def _calc_risk_value(feat, value):
    """True = 위험(범위 밖), False = 정상(범위 안), None = 판단 불가."""
    if value is None:
        return None
    if feat == 'po2':
        return  (75 <= value <= 100)
    if feat == 'pao2fio2ratio':
        return  (value >= 300)
    return None


# ── 메인 추론 함수 ────────────────────────────────────────────
def predict_ards(vital_ts, lab_df, patient_meta):
    """
    패혈증 환자의 48시간 내 ARDS 발생 확률 예측

    인터페이스는 mortality predict.py와 동일: (vital_ts, lab_df, patient_meta)
    lab_df 안에 bg 컬럼(lactate, ph, bicarbonate)이 포함되어 있으면 자동 분리.

    Parameters
    ----------
    vital_ts : pd.DataFrame
        컬럼: charttime, spo2, resp_rate, heart_rate, mbp, sbp, temperature
    lab_df : pd.DataFrame
        컬럼: charttime, creatinine, bun, wbc, platelet
        (+ 선택적: lactate, ph, bicarbonate)
    patient_meta : dict
        필수: age, onset_time(또는 sepsis_onset_time)
        성별: gender_bin(또는 gender)

    Returns
    -------
    dict :
        mortality predict.py와 동일한 응답 스펙.
        {
            "ards": {
                "probability":    0.342,
                "prediction":     1,
                "threshold":      0.30,
                "inference_time": "2026-04-15T...",
                "data_quality": {
                    "n_vital_slots":      int,
                    "n_lab_measurements": int,
                    "is_reliable":        bool,
                },
                "feature_values": [   # 전체 피처 (43개)
                    {"feature", "raw_value", "shap_value", "unit",
                     "is_imputed", "change", "change_direction"},
                    ...
                ],
                "top_features": [ ... ],   # |shap_value| 기준 상위 3개
            }
        }
    """
    artifact = _get_artifact()
    base_model = artifact['base_model']
    calibrator = artifact['calibrator']
    # threshold 키 호환: 노트북에서 "threshold_from_val"로 저장된 경우에도 정상 동작
    threshold  = artifact.get('threshold', artifact.get('threshold_from_val', 0.30))
    features   = artifact['features']

    # 전처리
    x = preprocess(vital_ts, lab_df, patient_meta)

    # Calibrated 확률 산출
    prob = float(calibrator.predict_proba(x)[0, 1])

    # SHAP 계산 (base_model 기준 — TreeExplainer 사용)
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(x)[0]

    # feature_values: mortality 스펙과 동일한 항목 구조 (전체 피처)
    # - ARDS는 FEAT_UNITS / imputation 플래그 / 이력 변화가 없으므로
    #   unit="", is_imputed=False, change=None, change_direction="unknown"
    raw_row = x[0]
    feature_values = []
    for i, feat in enumerate(FEAT_COLS):
        rv = float(raw_row[i])
        raw_value = round(rv, 4) if np.isfinite(rv) else None
        feature_values.append({
            "feature":          feat,
            "raw_value":        raw_value,
            "shap_value":       round(float(shap_values[i]), 4),
            "unit":             "",
            "is_imputed":       False,
            "change":           None,
            "change_direction": "unknown",
        })

    # top_features: |shap_value| 기준 상위 3개
    top_features = sorted(
        feature_values,
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )[:3]
    def _last_val_ards(df, col):
        if col not in df.columns:
            return None
        s = df[col].dropna()
        return round(float(s.iloc[-1]), 4) if len(s) > 0 else None

    clinical_indicators = {
        'po2':       {
            'value': _last_val_ards(lab_df, 'po2'),
            'reference': {**ARDS_CLINICAL_REFERENCE['po2'],
                          'risk_value': _calc_risk_value('po2', _last_val_ards(lab_df, 'po2'))}
        },
        'fio2_bg':   {
            'value': _last_val_ards(lab_df, 'fio2_bg'),
            'reference': {**ARDS_CLINICAL_REFERENCE['fio2_bg'], 'risk_value': None}
        },
        'pao2fio2ratio':  {
            'value': _last_val_ards(vital_ts, 'pao2fio2ratio'),
            'reference': {**ARDS_CLINICAL_REFERENCE['pao2fio2ratio'],
                          'risk_value': _calc_risk_value('pao2fio2ratio', _last_val_ards(vital_ts, 'pao2fio2ratio'))}
        },
        'peep_feat': {
            'value': _last_val_ards(lab_df, 'peep_feat'),
            'reference': {**ARDS_CLINICAL_REFERENCE['peep_feat'], 'risk_value': None}
        },
    }
    # data_quality: mortality 구조와 동일
    n_vital_slots      = int(len(vital_ts))
    n_lab_measurements = int(len(lab_df))

    return {
    "ards": {
        "probability":         round(prob, 4),
        "prediction":          int(prob >= threshold),
        "threshold":           threshold,
        "inference_time":      datetime.now(timezone.utc).isoformat(),
        "clinical_indicators": clinical_indicators,
        "data_quality": {
            "n_vital_slots":      n_vital_slots,
            "n_lab_measurements": n_lab_measurements,
            "is_reliable":        n_vital_slots >= 6,
        },
        "feature_values": feature_values,
        "top_features":   top_features,
    }
}


# ── 모델 파일 생성 도우미 ──────────────────────────────────────
def save_artifact_for_deploy(base_model, calibrator, features, threshold=0.30, save_path=None):
    """
    학습 노트북에서 이 함수를 호출하여 배포용 .joblib 파일을 생성한다.
    """
    artifact = {
        "base_model": base_model,
        "calibrator": calibrator,
        "features": features,
        "threshold": threshold,
        "model_info": {
            "dataset": "v6_master_win24_h48_conservative",
            "model": "XGBoost",
            "track": "conservative",
            "window_h": WINDOW_H,
            "horizon_h": 48,
        }
    }
    if save_path is None:
        save_path = ARTIFACT_FILENAME
    joblib.dump(artifact, save_path)
    print(f"[ARDS] 아티팩트 저장 완료: {save_path}")
    return save_path