# predict_ards.py
"""
ARDS 조기예측 추론 모듈
- 모델: XGBoost (calibrated via Platt Scaling)
- 대표 조합: master sepsis cohort + 24h window + 48h horizon + conservative feature
- 입력: onset 이전 24h 내 활력징후 + 혈액검사 + 환자 기본정보
- 출력: {"ards": {"probability": float, "shap": [{"feature": str, "value": float}]}}

변경 이력
---------
v2 (2025-06):
  - 인터페이스를 mortality predict.py와 통일 (vital_df + lab_df 2개로 통합)
    → bg 컬럼(lactate, ph, bicarbonate)은 lab_df 안에서 자동 분리
  - patient_meta 키 호환: gender / gender_bin 모두 수용
  - patient_meta 키 호환: onset_time / sepsis_onset_time 모두 수용
  - threshold 키 호환: threshold / threshold_from_val 모두 수용
  - XGBoost는 NaN을 자체 처리하므로 별도 imputation 불필요 (의도된 설계)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import shap

from config import ARTIFACT_FILENAME, FEAT_COLS, WINDOW_H
from loader import _get_artifact
from preprocess import preprocess


# ── 메인 추론 함수 ────────────────────────────────────────────
def predict_ards(vital_df, lab_df, patient_meta):
    """
    패혈증 환자의 48시간 내 ARDS 발생 확률 예측

    인터페이스는 mortality predict.py와 동일: (vital_df, lab_df, patient_meta)
    lab_df 안에 bg 컬럼(lactate, ph, bicarbonate)이 포함되어 있으면 자동 분리.

    Parameters
    ----------
    vital_df : pd.DataFrame
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
        {
            "ards": {
                "probability": 0.342,
                "threshold": 0.30,
                "prediction": 1,
                "shap": [
                    {"feature": "lactate_missing", "value": 0.0821},
                    ...
                ]
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
    x = preprocess(vital_df, lab_df, patient_meta)

    # Calibrated 확률 산출
    prob = float(calibrator.predict_proba(x)[0, 1])

    # SHAP 계산 (base_model 기준 — TreeExplainer 사용)
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(x)[0]
    shap_list = sorted(
        [
            {"feature": feat, "value": round(float(val), 4)}
            for feat, val in zip(FEAT_COLS, shap_values)
        ],
        key=lambda d: abs(d["value"]),
        reverse=True,
    )

    return {
        "ards": {
            "probability": round(prob, 4),
            "threshold": threshold,
            "prediction": int(prob >= threshold),
            "shap": shap_list,
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
