import os

# ── S3 / 환경 설정 ────────────────────────────────────────────
S3_BUCKET        = os.getenv('S3_BUCKET', 'say2-1team')
MODEL_PREFIX     = os.getenv('MODEL_PREFIX', 'pipeline/final_model')
USE_S3           = os.getenv('USE_S3', 'true').lower() == 'true'
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './models/sic')
THRESHOLD        = 0.5
SEQ_LEN          = 48

# ── 정적 피처 ─────────────────────────────────────────────────
STATIC_COLS = [
    'age', 'sex_male',
    'flag_liver_failure', 'flag_ckd', 'flag_coagulopathy',
    'flag_diabetes', 'flag_immunosuppression', 'flag_chf', 'flag_septic_shock_hx'
]

# ── 시계열 원값 컬럼 + ffill 한도 ────────────────────────────
TS_COLS = ['map', 'pao2', 'creatinine', 'wbc', 'rdw', 'fio2', 'aptt', 'inr', 'lactate', 'bilirubin_total']

FFILL_LIMITS = {
    'map':            1,
    'pao2':           4,
    'creatinine':     12,
    'wbc':            12,
    'rdw':            12,
    'fio2':           4,
    'aptt':           12,
    'inr':            24,
    'lactate':        6,
    'bilirubin_total': 24,
}

# ── 시계열 모델 입력 컬럼 (원값 + mask, label leakage 제거 후) ─
TS_VALUE_COLS = [
    'hours_from_onset', 'lactate', 'creatinine', 'bilirubin_total',
    'wbc', 'rdw', 'map', 'map_mask', 'creatinine_mask', 'wbc_mask',
    'rdw_mask', 'aptt_mask', 'lactate_mask', 'bilirubin_total_mask', 'pf_ratio'
]

INPUT_DIM = 15



# ── 파생 피처 ─────────────────────────────────────────────────
TS_DERIVED_COLS = [
    'map_last', 'map_trend',
    'aptt_last', 'aptt_trend',
    'lactate_last', 'lactate_trend',
    'creatinine_last', 'creatinine_trend',
    'bilirubin_total_last', 'bilirubin_total_trend',
    'wbc_last', 'wbc_trend',
    'rdw_last', 'rdw_trend',
    'pf_ratio_last', 'pf_ratio_trend',
    'map_min', 'map_mean',
    'aptt_max', 'lactate_max', 'creatinine_max',
    'bilirubin_total_max', 'wbc_max', 'wbc_min',
    'rdw_mean', 'pf_ratio_min',
]

# BiLSTM 입력 채널 수 (value + mask + derived)
