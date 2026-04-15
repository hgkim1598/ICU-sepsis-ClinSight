import os

# ── S3 / 환경 설정 ────────────────────────────────────────────
S3_BUCKET        = os.getenv('S3_BUCKET', 'say2-1team')
MODEL_PREFIX = os.getenv('MODEL_PREFIX', 'pipeline/final_model')
HISTORY_PREFIX   = 'pipeline/patient_history'
USE_S3           = os.getenv('USE_S3', 'true').lower() == 'true'
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './models')
THRESHOLD        = 0.21
SEQ_LEN          = 48

# ── 피처 정의 ─────────────────────────────────────────────────
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

FEAT_UNITS = {
    'heart_rate_last': 'bpm', 'heart_rate_min': 'bpm', 'heart_rate_max': 'bpm',
    'mbp_last': 'mmHg', 'mbp_min': 'mmHg', 'mbp_slope': 'mmHg/h',
    'sbp_last': 'mmHg', 'sbp_min': 'mmHg', 'sbp_slope': 'mmHg/h',
    'dbp_last': 'mmHg', 'dbp_min': 'mmHg', 'dbp_slope': 'mmHg/h',
    'resp_rate_last': 'breaths/min', 'resp_rate_max': 'breaths/min', 'resp_rate_slope': 'breaths/min/h',
    'spo2_last': '%', 'spo2_min': '%',
    'temperature_min': '°C', 'temperature_max': '°C',
    'gcs_last': 'score', 'gcs_min': 'score', 'gcs_missing_flag': 'flag',
    'urine_last': 'mL', 'urine_min': 'mL', 'urine_diff': 'mL',
    'lactate_last': 'mmol/L', 'lactate_max': 'mmol/L', 'lactate_slope': 'mmol/L/h',
    'creatinine_last': 'mg/dL', 'creatinine_max': 'mg/dL', 'creatinine_slope': 'mg/dL/h',
    'bun_last': 'mg/dL', 'bun_slope': 'mg/dL/h',
    'sodium_last': 'mEq/L', 'sodium_min': 'mEq/L', 'sodium_max': 'mEq/L',
    'potassium_last': 'mEq/L', 'potassium_min': 'mEq/L', 'potassium_max': 'mEq/L',
    'glucose_last': 'mg/dL', 'glucose_min': 'mg/dL', 'glucose_max': 'mg/dL',
    'bicarbonate_last': 'mEq/L', 'bicarbonate_min': 'mEq/L',
    'pao2fio2_last': 'mmHg', 'pao2fio2_min': 'mmHg', 'pao2fio2_slope': 'mmHg/h',
    'albumin_min': 'g/dL', 'albumin_missing_flag': 'flag',
    'wbc_last': 'K/uL', 'wbc_min': 'K/uL', 'wbc_max': 'K/uL', 'wbc_slope': 'K/uL/h',
    'platelet_last': 'K/uL', 'platelet_min': 'K/uL', 'platelet_slope': 'K/uL/h',
    'hemoglobin_last': 'g/dL', 'hemoglobin_min': 'g/dL', 'hemoglobin_diff': 'g/dL',
    'bilirubin_min': 'mg/dL', 'bilirubin_max': 'mg/dL', 'bilirubin_missing_flag': 'flag',
    'age': 'years', 'gender': 'M=1/F=0',
}