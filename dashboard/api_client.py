from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from feature_labels import FEATURE_LABELS, get_feature_label


load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PATIENTS_ENDPOINT = "/patients"
# FastAPI 예측 endpoint (4개 모델 결과 일괄 반환)
PREDICT_ENDPOINT = "/predict/{patient_id}"
# 단일 모델 예측 endpoint (하위 호환용 — 현재 미사용)
MODEL_PREDICT_ENDPOINT = "/api/v1/predict/{patient_id}/{model_name}"
# (구) 대시보드 일괄 endpoint — 남아있으면 호환용으로 우선 사용
PREDICTION_ENDPOINT = "/predictions/latest"
REQUEST_TIMEOUT_SECONDS = 30
MODEL_ORDER = ["Mortality", "ARDS", "AKI", "SIC"]
# 표시 모델명 ↔ API 응답 키 매핑 (API는 소문자)
MODEL_API_KEY: Dict[str, str] = {
    "Mortality": "mortality",
    "ARDS": "ards",
    "AKI": "aki",
    "SIC": "sic",
}

# feature 라벨은 feature_labels.py로 분리. 구 이름은 하위 호환 alias로만 유지.
FEATURE_DISPLAY_MAP = FEATURE_LABELS

FEATURE_UNITS: Dict[str, str] = {
    "age": "세",
    "bun_last": "mg/dL",
    "creatinine_last": "mg/dL",
    "lactate_last": "mmol/L",
    "lactate_max": "mmol/L",
    "mbp_last": "mmHg",
    "mbp_slope": "mmHg/hr",
    "map_last": "mmHg",
    "pao2fio2_min": "mmHg",
    "platelet_last": "×10³/μL",
    "pt_inr_last": "",
    "resp_rate_last": "회/min",
    "bilirubin_last": "mg/dL",
    "uo_6h": "mL",
    "uo_24h": "mL",
}

# (min, max) — None on one side means one-directional bound
FEATURE_NORMAL_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "bun_last":        (7.0,    25.0),
    "creatinine_last": (0.5,     1.2),
    "lactate_last":    (0.5,     2.0),
    "lactate_max":     (0.5,     2.0),
    "mbp_last":        (70.0,  100.0),
    "map_last":        (70.0,  100.0),
    "pao2fio2_min":    (300.0,  None),   # ≥300 is normal
    "platelet_last":   (150.0, 400.0),
    "pt_inr_last":     (0.8,     1.2),
    "resp_rate_last":  (12.0,   20.0),
    "bilirubin_last":  (0.2,     1.2),
    "uo_6h":           (240.0, 600.0),
    "uo_24h":          (800.0, 2000.0),
}

FEATURE_NORMAL_RANGE_STR: Dict[str, str] = {
    "bun_last":        "7–25",
    "creatinine_last": "0.5–1.2",
    "lactate_last":    "0.5–2.0",
    "lactate_max":     "0.5–2.0",
    "mbp_last":        "70–100",
    "map_last":        "70–100",
    "pao2fio2_min":    "≥ 300",
    "platelet_last":   "150–400",
    "pt_inr_last":     "0.8–1.2",
    "resp_rate_last":  "12–20",
    "bilirubin_last":  "0.2–1.2",
    "uo_6h":           "240–600",
    "uo_24h":          "800–2000",
}

MODEL_KR_NAME = {
    "Mortality": "사망 위험도",
    "ARDS": "급성호흡곤란 (ARDS)",
    "AKI": "급성신손상 (AKI)",
    "SIC": "패혈증 유발 응고장애 (SIC)",
}

# ── 핵심 지표(clinical_indicators) 한글 매핑 ──────────────────
# API 응답 clinical_indicators의 하위 키(예: "ventilation") → UI 표시 한글.
# 매핑에 없으면 영문 키 그대로 노출(fallback).
CLINICAL_INDICATOR_LABELS: Dict[str, str] = {
    # Mortality
    "ventilation":    "기계환기",
    "norepinephrine": "노르에피네프린",
    "dopamine":       "도파민",
    "dobutamine":     "도부타민",
    "epinephrine":    "에피네프린",
    # ARDS
    "po2":            "동맥혈 산소분압",
    "fio2_bg":        "흡입 산소 농도",
    "pao2fio2ratio":  "P/F ratio",
    "peep_feat":      "호기말 양압 (PEEP)",
    # SIC
    "platelet":       "혈소판",
    "inr":            "INR (프로트롬빈)",
    # AKI
    "lactate":        "젖산",
    "spo2":           "산소포화도",
    "gcs":            "의식수준 (GCS)",
    "sbp":            "수축기혈압",
    "wbc":            "백혈구",
    "hemoglobin":     "헤모글로빈",
}


def get_clinical_indicator_label(name: str) -> str:
    """CLINICAL_INDICATOR_LABELS에 매핑이 있으면 한글 라벨, 없으면 그대로."""
    return CLINICAL_INDICATOR_LABELS.get(name, name)


def _normalize_clinical_indicators(raw: Any) -> List[Dict[str, Any]]:
    """
    API clinical_indicators dict → UI 렌더링용 list.

    입력: {"ventilation": {"value": 0, "reference": {"unit", "usual_range", "risk_value"}}, ...}
    출력 항목: {"name", "display_name", "value", "unit", "usual_range", "risk_value"}
           risk_value: True / False / None (그대로 전달 — 렌더러가 색 결정)
    비어있거나 None이면 [] 반환.
    """
    if not isinstance(raw, dict) or not raw:
        return []
    result: List[Dict[str, Any]] = []
    for key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        ref = payload.get("reference") or {}
        if not isinstance(ref, dict):
            ref = {}
        result.append({
            "name":         str(key),
            "display_name": get_clinical_indicator_label(str(key)),
            "value":        payload.get("value"),
            "unit":         str(ref.get("unit") or ""),
            "usual_range":  str(ref.get("usual_range") or ""),
            "risk_value":   ref.get("risk_value"),  # True/False/None 그대로
        })
    return result

MOCK_DASHBOARD_DATA: Dict[str, Any] = {
    "patient": {
        "name": "환자 A",
        "patient_id": "ICU-2026-0410",
        "age": 68,
        "gender": "Female",
        "admit_date": "2026-04-08",
        "diagnosis": "Septic Shock",
        "ward": "ICU-3",
        "icu_admit_time": "2026-04-08 14:30",
        "sofa_score": 12,
        "sepsis_onset": "2026-04-09 03:15",
    },
    "meta": {
        "last_updated": "2026-04-10T18:20:00",
    },
    "feature_values": {
        "bun_last":        45.2,
        "creatinine_last":  2.3,
        "lactate_last":     4.2,
        "lactate_max":      5.1,
        "mbp_last":        62.0,
        "mbp_slope":       -2.3,
        "pao2fio2_min":   180.0,
        "platelet_last":   85.0,
        "pt_inr_last":      1.8,
        "resp_rate_last":  24.0,
        "bilirubin_last":   1.9,
        "uo_6h":          180.0,
        "uo_24h":         720.0,
        "age":             68.0,
    },
    "models": {
        "Mortality": {
            "probability": 0.78,
            "shap_values": [
                {"feature": "lactate_max",  "value":  0.52},
                {"feature": "bun_last",     "value":  0.34},
                {"feature": "age",          "value":  0.28},
                {"feature": "mbp_last",     "value": -0.18},
            ],
        },
        "AKI": {
            "probability": 0.56,
            "shap_values": [
                {"feature": "creatinine_last", "value":  0.61},
                {"feature": "uo_6h",           "value": -0.43},
                {"feature": "bun_last",        "value":  0.24},
                {"feature": "mbp_last",        "value": -0.11},
            ],
        },
        "ARDS": {
            "probability": 0.33,
            "shap_values": [
                {"feature": "pao2fio2_min",   "value": -0.57},
                {"feature": "resp_rate_last", "value":  0.31},
                {"feature": "lactate_last",   "value":  0.18},
                {"feature": "mbp_last",       "value": -0.07},
            ],
        },
        "SIC": {
            "probability": 0.64,
            "shap_values": [
                {"feature": "platelet_last",  "value": -0.49},
                {"feature": "pt_inr_last",    "value":  0.38},
                {"feature": "bilirubin_last", "value":  0.29},
                {"feature": "lactate_last",   "value":  0.16},
            ],
        },
    },
}


def fetch_patients() -> List[str]:
    """
    GET /patients — 환자 ID 목록 반환.

    응답 형식: {"patients": ["환자ID1", "환자ID2", ...]}
    """
    url = f"{API_BASE_URL.rstrip('/')}{PATIENTS_ENDPOINT}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    patients = payload.get("patients", [])
    return [str(pid) for pid in patients if pid is not None]


def fetch_patient_data(patient_id: str) -> Dict[str, Any]:
    """
    GET /patients/{patient_id}/data — 환자 기본 정보 반환.

    응답 형식: {"patient_id": "...", "patient_meta": {"age": ..., "gender": ..., ...}}
    """
    url = f"{API_BASE_URL.rstrip('/')}/patients/{patient_id}/data"
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_predictions(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    POST /predict/{patient_id} — 4개 모델 예측 결과 일괄 반환.

    성공 시 {"mortality": {...}, "ards": {...}, "aki": {...}, "sic": {...}} dict,
    실패 시 None (호출 측에서 fallback 가능).
    """
    url = f"{API_BASE_URL.rstrip('/')}{PREDICT_ENDPOINT.format(patient_id=patient_id)}"
    try:
        resp = requests.post(url, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        print(f"[API] POST {url} → keys={list(data.keys())}")
        return data
    except Exception as e:
        print(f"[API ERROR] POST {url} failed: {e}")
        return None


def format_last_updated(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def get_feature_display_name(feature_name: str) -> str:
    """매핑에 없으면 feature 이름 그대로 반환(fallback)."""
    return get_feature_label(feature_name)


def normalize_shap_values(model_result: Dict[str, Any]) -> list[dict]:
    # API 스펙 상 shap 키가 표준. 구 mock은 shap_values 키를 사용하므로 둘 다 허용.
    shap_raw = model_result.get("shap")
    if shap_raw is None:
        shap_raw = model_result.get("shap_values")

    if isinstance(shap_raw, dict):
        normalized = [
            {"feature": key, "value": value}
            for key, value in shap_raw.items()
        ]
    elif isinstance(shap_raw, list):
        normalized = []
        for item in shap_raw:
            if isinstance(item, dict):
                feature_name = item.get("feature") or item.get("name") or item.get("feature_name")
                feature_value = item.get("value")
                if feature_name is not None and feature_value is not None:
                    normalized.append({"feature": feature_name, "value": feature_value})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append({"feature": item[0], "value": item[1]})
    else:
        fallback_top_features = model_result.get("top_features", [])
        normalized = [{"feature": feature, "value": 0.0} for feature in fallback_top_features]

    # API 스펙: 프론트에서 value 내림차순 정렬 후 상위 3개 slice (shap은 60+개도 가능)
    normalized.sort(key=lambda item: float(item.get("value", 0.0)), reverse=True)
    return normalized


def get_feature_value_info(
    feature_name: str,
    feature_values: Dict[str, Any],
) -> Dict[str, Any]:
    raw_value = feature_values.get(feature_name)
    unit = FEATURE_UNITS.get(feature_name, "")
    normal_range = FEATURE_NORMAL_RANGES.get(feature_name)
    normal_range_str = FEATURE_NORMAL_RANGE_STR.get(feature_name, "")
    display_name = get_feature_display_name(feature_name)

    is_abnormal = False
    direction: Optional[str] = None
    if raw_value is not None and normal_range is not None:
        lo, hi = normal_range
        if lo is not None and raw_value < lo:
            is_abnormal, direction = True, "low"
        elif hi is not None and raw_value > hi:
            is_abnormal, direction = True, "high"

    return {
        "feature": feature_name,
        "display_name": display_name,
        "value": raw_value,
        "unit": unit,
        "normal_range_str": normal_range_str,
        "is_abnormal": is_abnormal,
        "direction": direction,
    }


def build_description(model_name: str, top_features_display: list[str]) -> str:
    if not top_features_display:
        return f"{MODEL_KR_NAME.get(model_name, model_name)}에 영향을 주는 주요 피처 정보가 없습니다."
    feature_text = ", ".join(top_features_display[:3])
    return f"{feature_text}가 {MODEL_KR_NAME.get(model_name, model_name)} 예측에 크게 영향을 주었습니다."


def enrich_model_result(
    model_name: str,
    model_result: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> Dict[str, Any]:
    probability = float(model_result.get("probability", 0.0))
    sorted_shap = normalize_shap_values(model_result)
    top_shap = sorted_shap[:3]
    top_features_display = [
        get_feature_display_name(str(item.get("feature", "-")))
        for item in top_shap
    ]
    top_feature_values = [
        get_feature_value_info(str(item.get("feature", "")), feature_values)
        for item in top_shap
    ]

    # mock 경로에서도 API 스펙과 동일한 shape로 제공 (UI가 단일 경로로 렌더)
    top_features_api_shape: List[Dict[str, Any]] = []
    for item in top_shap:
        feat = str(item.get("feature", ""))
        raw = feature_values.get(feat)
        top_features_api_shape.append({
            "feature": feat,
            "shap_value": float(item.get("value", 0.0)),
            "raw_value": raw,
            "unit": FEATURE_UNITS.get(feat, ""),
            "is_imputed": False,
            "change": 0.0,
            "change_direction": "equal",
        })

    feature_values_api_shape: List[Dict[str, Any]] = []
    for feat, raw in feature_values.items():
        feature_values_api_shape.append({
            "feature": feat,
            "shap_value": 0.0,
            "raw_value": raw,
            "unit": FEATURE_UNITS.get(feat, ""),
            "is_imputed": False,
            "change": 0.0,
            "change_direction": "equal",
        })

    return {
        "probability": probability,
        "shap_values": sorted_shap,  # 구 경로 호환
        "top_features": top_features_api_shape,
        "feature_values": feature_values_api_shape,
        "top_features_display": top_features_display,
        "top_feature_values": top_feature_values,
        "clinical_indicators": [],  # mock 경로엔 API 데이터 없음 → 빈 리스트
        "description": model_result.get(
            "description",
            build_description(model_name, top_features_display),
        ),
        "has_api_data": False,
        "data_quality": {},
        "threshold": None,
        "prediction": None,
        "inference_time": None,
    }


def enrich_from_api(
    model_name: str,
    api_model: Dict[str, Any],
) -> Dict[str, Any]:
    """
    FastAPI /predict 응답의 단일 모델 블록을 UI용 dict로 변환.
    top_features는 이미 정렬된 상태이므로 별도 정렬 없이 그대로 사용.
    핵심 지표 측정값(top_feature_values)은 API에 없으므로 mock에서 가져와 회색 렌더.
    """
    probability = float(api_model.get("probability", 0.0))
    top_features_raw = api_model.get("top_features", []) or []
    feature_values_raw = api_model.get("feature_values", []) or []

    # 한글 라벨 (SHAP / 설명 텍스트)
    top_features_display = [
        get_feature_display_name(str(item.get("feature", "-")))
        for item in top_features_raw[:3]
    ]

    # 핵심 지표 테이블용 mock top_feature_values (회색 표시)
    mock_model = MOCK_DASHBOARD_DATA.get("models", {}).get(model_name, {})
    mock_feature_values = MOCK_DASHBOARD_DATA.get("feature_values", {})
    mock_sorted_shap = normalize_shap_values(mock_model)
    mock_top_feature_values = [
        get_feature_value_info(str(item.get("feature", "")), mock_feature_values)
        for item in mock_sorted_shap[:3]
    ]

    # clinical_indicators: API에 있으면 파싱, 없으면 빈 리스트
    # (빈 리스트면 렌더러가 기존 mock top_feature_values(회색)로 fallback)
    clinical_indicators = _normalize_clinical_indicators(
        api_model.get("clinical_indicators")
    )

    return {
        "probability": probability,
        "threshold": api_model.get("threshold"),
        "prediction": api_model.get("prediction"),
        "inference_time": api_model.get("inference_time"),
        "data_quality": api_model.get("data_quality", {}) or {},
        "top_features": top_features_raw,
        "feature_values": feature_values_raw,
        # 구 필드(호환): shap_values = top_features 기반 value 내림차순
        "shap_values": [
            {"feature": str(it.get("feature", "")), "value": float(it.get("shap_value", 0.0))}
            for it in top_features_raw
        ],
        "top_features_display": top_features_display,
        "top_feature_values": mock_top_feature_values,
        "clinical_indicators": clinical_indicators,
        "description": build_description(model_name, top_features_display),
        "has_api_data": True,
    }


def get_model_result(patient_id: str, model_name: str) -> Dict[str, Any]:
    """
    단일 모델의 예측 결과를 API 스펙 형태로 반환.

    반환 형식:
        {
            "probability": float,                           # 0.0 ~ 1.0
            "shap": [{"feature": str, "value": float}, ...] # 원본 순서, 정렬 없음
        }

    # TODO: API 연동 시 이 함수 내부를 실제 API 호출로 교체
    # API endpoint: /api/v1/predict/{patient_id}/{model_name}
    # 현재: mock 데이터 반환
    #
    # 교체 예시:
    #   url = f"{API_BASE_URL.rstrip('/')}" + MODEL_PREDICT_ENDPOINT.format(
    #       patient_id=patient_id, model_name=model_name.lower()
    #   )
    #   response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    #   response.raise_for_status()
    #   return response.json()  # {"probability": ..., "shap": [...]}
    """
    model_data = MOCK_DASHBOARD_DATA.get("models", {}).get(model_name, {})
    shap_raw = model_data.get("shap_values", [])

    shap_list: List[Dict[str, Any]] = []
    if isinstance(shap_raw, list):
        for item in shap_raw:
            if isinstance(item, dict):
                feature = item.get("feature") or item.get("name")
                value = item.get("value")
                if feature is not None and value is not None:
                    shap_list.append({"feature": str(feature), "value": float(value)})
    elif isinstance(shap_raw, dict):
        for key, value in shap_raw.items():
            shap_list.append({"feature": str(key), "value": float(value)})

    return {
        "probability": float(model_data.get("probability", 0.0)),
        "shap": shap_list,
    }


def _build_dashboard_envelope(
    models_enriched: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    """환자/meta는 여전히 mock에서 가져옴 (API 연동 범위 밖)."""
    meta = MOCK_DASHBOARD_DATA.get("meta", {})
    return {
        "patient": MOCK_DASHBOARD_DATA.get("patient", {}),
        "meta": {
            "source": source,
            "source_label": "API 연결" if source == "api" else "Mock data",
            "last_updated": meta.get("last_updated"),
            "last_updated_display": format_last_updated(meta.get("last_updated")),
            "api_base_url": API_BASE_URL,
        },
        "models": models_enriched,
    }


def fetch_dashboard_data(
    use_mock_override: bool = False,
    use_mock_on_error: bool = True,
    patient_id: Optional[str] = None,
    predictions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    대시보드 전체 데이터(환자 + 4개 모델 예측)를 반환.

    `predictions`가 주어지면 해당 모델은 API 데이터로, 없으면 mock으로 채움.
    개별 모델 단위로 has_api_data 플래그가 붙어 UI에서 회색 처리 여부를 결정.
    """
    pid = patient_id or str(
        MOCK_DASHBOARD_DATA.get("patient", {}).get("patient_id", "")
    )
    feature_values = MOCK_DASHBOARD_DATA.get("feature_values", {})
    predictions = predictions or {}

    models_enriched: Dict[str, Any] = {}
    any_api_hit = False
    for model_name in MODEL_ORDER:
        api_key = MODEL_API_KEY.get(model_name)
        api_model = predictions.get(api_key) if api_key else None

        if isinstance(api_model, dict) and api_model:
            models_enriched[model_name] = enrich_from_api(model_name, api_model)
            any_api_hit = True
            continue

        # API 데이터 없음 → mock fallback (has_api_data=False로 회색 렌더)
        try:
            raw = get_model_result(pid, model_name)
        except requests.RequestException:
            if not use_mock_on_error:
                raise
            raw = {
                "probability": float(
                    MOCK_DASHBOARD_DATA["models"].get(model_name, {}).get("probability", 0.0)
                ),
                "shap": [],
            }
        models_enriched[model_name] = enrich_model_result(
            model_name, raw, feature_values
        )

    if use_mock_override:
        source = "mock"
    else:
        source = "api" if any_api_hit else "mock"
    return _build_dashboard_envelope(models_enriched, source=source)
