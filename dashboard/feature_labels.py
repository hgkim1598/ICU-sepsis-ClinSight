"""
Feature 이름 → 한국어 라벨 매핑.

백엔드 모델이 사용하는 feature 문자열을 UI에 표시할 한국어 라벨로 변환한다.
매핑에 없는 feature가 들어오면 feature 이름을 그대로 반환(fallback).
"""
from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    # 활력징후
    "heart_rate_last":        "심박수 (최근값)",
    "heart_rate_min":         "심박수 (최솟값)",
    "heart_rate_max":         "심박수 (최댓값)",
    "mbp_last":               "평균동맥압 (최근값)",
    "mbp_min":                "평균동맥압 (최솟값)",
    "mbp_slope":              "평균동맥압 (추세)",
    "sbp_last":               "수축기혈압 (최근값)",
    "sbp_min":                "수축기혈압 (최솟값)",
    "sbp_slope":              "수축기혈압 (추세)",
    "dbp_last":               "이완기혈압 (최근값)",
    "dbp_min":                "이완기혈압 (최솟값)",
    "dbp_slope":              "이완기혈압 (추세)",
    "resp_rate_last":         "호흡수 (최근값)",
    "resp_rate_max":          "호흡수 (최댓값)",
    "resp_rate_slope":        "호흡수 (추세)",
    "spo2_last":              "산소포화도 (최근값)",
    "spo2_min":               "산소포화도 (최솟값)",
    "temperature_min":        "체온 (최솟값)",
    "temperature_max":        "체온 (최댓값)",

    # 의식
    "gcs_last":               "GCS (최근값)",
    "gcs_min":                "GCS (최솟값)",
    "gcs_missing_flag":       "GCS 결측 여부",

    # 소변량
    "urine_last":             "소변량 (최근값)",
    "urine_min":              "소변량 (최솟값)",
    "urine_diff":             "소변량 (변화량)",

    # 혈액검사 - 산소/호흡
    "lactate_last":           "젖산 (최근값)",
    "lactate_max":            "젖산 (최댓값)",
    "lactate_slope":          "젖산 (추세)",
    "pao2fio2_last":          "P/F ratio (최근값)",
    "pao2fio2_min":           "P/F ratio (최솟값)",
    "pao2fio2_slope":         "P/F ratio (추세)",

    # 혈액검사 - 신장
    "creatinine_last":        "크레아티닌 (최근값)",
    "creatinine_max":         "크레아티닌 (최댓값)",
    "creatinine_slope":       "크레아티닌 (추세)",
    "bun_last":               "BUN (최근값)",
    "bun_slope":              "BUN (추세)",

    # 혈액검사 - 전해질
    "sodium_last":            "나트륨 (최근값)",
    "sodium_min":             "나트륨 (최솟값)",
    "sodium_max":             "나트륨 (최댓값)",
    "potassium_last":         "칼륨 (최근값)",
    "potassium_min":          "칼륨 (최솟값)",
    "potassium_max":          "칼륨 (최댓값)",
    "glucose_last":           "혈당 (최근값)",
    "glucose_min":            "혈당 (최솟값)",
    "glucose_max":            "혈당 (최댓값)",
    "bicarbonate_last":       "중탄산염 (최근값)",
    "bicarbonate_min":        "중탄산염 (최솟값)",

    # 혈액검사 - 간/영양
    "albumin_min":            "알부민 (최솟값)",
    "albumin_missing_flag":   "알부민 결측 여부",
    "bilirubin_min":          "빌리루빈 (최솟값)",
    "bilirubin_max":          "빌리루빈 (최댓값)",
    "bilirubin_missing_flag": "빌리루빈 결측 여부",

    # 혈액검사 - 혈구
    "wbc_last":               "백혈구 (최근값)",
    "wbc_min":                "백혈구 (최솟값)",
    "wbc_max":                "백혈구 (최댓값)",
    "wbc_slope":              "백혈구 (추세)",
    "platelet_last":          "혈소판 (최근값)",
    "platelet_min":           "혈소판 (최솟값)",
    "platelet_slope":         "혈소판 (추세)",
    "hemoglobin_last":        "헤모글로빈 (최근값)",
    "hemoglobin_min":         "헤모글로빈 (최솟값)",
    "hemoglobin_diff":        "헤모글로빈 (변화량)",

    # 환자 정보
    "age":                    "나이",
    "gender":                 "성별",

    # ── ARDS 파이프라인 추가 매핑 ──────────────────────────────
    # 활력징후 (mean/trend 계열 — mortality는 slope 사용, ARDS는 trend 사용)
    "heart_rate_mean":        "심박수 (평균값)",
    "heart_rate_trend":       "심박수 (추세)",
    "resp_rate_mean":         "호흡수 (평균값)",
    "resp_rate_trend":        "호흡수 (추세)",
    "spo2_mean":              "산소포화도 (평균값)",
    "spo2_trend":             "산소포화도 (추세)",
    "mbp_trend":              "평균동맥압 (추세)",
    "sbp_trend":              "수축기혈압 (추세)",
    "temperature_last":       "체온 (최근값)",
    "temperature_trend":      "체온 (추세)",

    # 혈액가스
    "lactate_trend":          "젖산 (추세)",
    "lactate_missing":        "젖산 (결측 여부)",
    "ph_last":                "pH (최근값)",
    "ph_min":                 "pH (최솟값)",
    "ph_trend":               "pH (추세)",
    "ph_missing":             "pH (결측 여부)",
    "bicarbonate_trend":      "중탄산염 (추세)",
    "bicarbonate_missing":    "중탄산염 (결측 여부)",

    # 혈액검사
    "creatinine_trend":       "크레아티닌 (추세)",
    "bun_trend":              "BUN (추세)",
    "wbc_trend":              "백혈구 (추세)",
    "platelet_trend":         "혈소판 (추세)",

    # 환자 정보
    "gender_bin":             "성별",
}


def get_feature_label(feature_name: str) -> str:
    """FEATURE_LABELS에 매핑이 있으면 한국어 라벨, 없으면 feature 이름 그대로."""
    return FEATURE_LABELS.get(feature_name, feature_name)
