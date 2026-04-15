"""
ARDS 추론 파이프라인 설정.
- S3 / 로컬 모델 경로
- 관측 윈도우 / 피처 정의 (FEAT_COLS, STAT_RULES 등)
"""
import os

# ── S3 / 환경 설정 ────────────────────────────────────────────
S3_BUCKET        = os.getenv('S3_BUCKET',        'say2-1team')
MODEL_PREFIX     = os.getenv('MODEL_PREFIX',     'pipeline/final_model')
USE_S3           = os.getenv('USE_S3', 'true').lower() == 'true'
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './models/ards')

ARTIFACT_FILENAME = 'ards_XGB.joblib'

# 관측 윈도우: onset 이전 24시간
WINDOW_H = 24

# ── 피처 정의 ─────────────────────────────────────────────────
# conservative track: label-adjacent 변수(po2, fio2, pf_ratio, peep) 제외
STAT_RULES = {
    "spo2":          ["last", "mean", "trend", "min"],
    "resp_rate":     ["last", "mean", "trend", "max"],
    "heart_rate":    ["last", "mean", "trend", "max"],
    "mbp":           ["last", "trend", "min"],
    "sbp":           ["last", "trend", "min"],
    "temperature":   ["last", "trend", "max"],
    "lactate":       ["last", "trend", "max", "missing"],
    "ph":            ["last", "trend", "min", "missing"],
    "bicarbonate":   ["last", "trend", "min", "missing"],
    "creatinine":    ["last", "trend"],
    "bun":           ["last", "trend"],
    "wbc":           ["last", "trend"],
    "platelet":      ["last", "trend"],
}

# 피처 → 원본 컬럼명 매핑
COL_MAP = {
    "spo2": "spo2", "resp_rate": "resp_rate", "heart_rate": "heart_rate",
    "mbp": "mbp", "sbp": "sbp", "temperature": "temperature",
    "lactate": "lactate", "ph": "ph", "bicarbonate": "bicarbonate",
    "creatinine": "creatinine", "bun": "bun", "wbc": "wbc", "platelet": "platelet",
}

# 피처 → 데이터 소스 구분 (vital / bg / lab)
SOURCE_MAP = {
    "spo2": "vital", "resp_rate": "vital", "heart_rate": "vital",
    "mbp": "vital", "sbp": "vital", "temperature": "vital",
    "lactate": "bg", "ph": "bg", "bicarbonate": "bg",
    "creatinine": "lab", "bun": "lab", "wbc": "lab", "platelet": "lab",
}

# bg(혈액가스) 컬럼 목록 — lab_df에서 자동 분리할 때 사용
BG_COLS = {"lactate", "ph", "bicarbonate"}

# 최종 피처 컬럼 순서 (age, gender_bin + 41개 summary stats = 43개)
FEAT_COLS = ["age", "gender_bin"]
for _feat_name, _stats in STAT_RULES.items():
    for _stat in _stats:
        FEAT_COLS.append(f"{_feat_name}_{_stat}")
