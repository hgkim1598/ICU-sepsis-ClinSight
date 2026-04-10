from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import requests


API_BASE_URL = os.getenv("DASHBOARD_API_BASE_URL", "http://localhost:8000")
PREDICTION_ENDPOINT = "/predictions/latest"
REQUEST_TIMEOUT_SECONDS = 5
MODEL_ORDER = ["Mortality", "AKI", "ARDS", "SIC"]

FEATURE_DISPLAY_MAP = {
    "age": "나이",
    "bun_last": "BUN 최근값",
    "creatinine_last": "크레아티닌 최근값",
    "lactate_last": "Lactate 최근값",
    "lactate_max": "Lactate 최고값",
    "mbp_last": "평균동맥압 최근값",
    "mbp_slope": "평균동맥압 변화",
    "map_last": "평균동맥압 최근값",
    "pao2fio2_min": "P/F ratio 최저값",
    "platelet_last": "혈소판 최근값",
    "pt_inr_last": "PT-INR 최근값",
    "resp_rate_last": "호흡수 최근값",
    "bilirubin_last": "빌리루빈 최근값",
    "uo_6h": "6시간 소변량",
    "uo_24h": "24시간 소변량",
}

MODEL_KR_NAME = {
    "Mortality": "사망 위험도",
    "AKI": "AKI 위험도",
    "ARDS": "ARDS 위험도",
    "SIC": "SIC 위험도",
}

MOCK_DASHBOARD_DATA: Dict[str, Any] = {
    "patient": {
        "name": "Kim Minseo",
        "patient_id": "ICU-2026-0410",
        "age": 68,
        "gender": "Female",
    },
    "meta": {
        "last_updated": "2026-04-10T18:20:00",
    },
    "models": {
        "Mortality": {
            "probability": 0.78,
            "shap_values": [
                {"feature": "bun_last", "value": 0.34},
                {"feature": "lactate_max", "value": 0.52},
                {"feature": "age", "value": 0.28},
                {"feature": "mbp_last", "value": -0.18},
            ],
        },
        "AKI": {
            "probability": 0.56,
            "shap_values": [
                {"feature": "creatinine_last", "value": 0.61},
                {"feature": "uo_6h", "value": -0.43},
                {"feature": "bun_last", "value": 0.24},
                {"feature": "mbp_last", "value": -0.11},
            ],
        },
        "ARDS": {
            "probability": 0.33,
            "shap_values": [
                {"feature": "pao2fio2_min", "value": -0.57},
                {"feature": "resp_rate_last", "value": 0.31},
                {"feature": "lactate_last", "value": 0.18},
                {"feature": "mbp_last", "value": -0.07},
            ],
        },
        "SIC": {
            "probability": 0.64,
            "shap_values": [
                {"feature": "platelet_last", "value": -0.49},
                {"feature": "pt_inr_last", "value": 0.38},
                {"feature": "bilirubin_last", "value": 0.29},
                {"feature": "lactate_last", "value": 0.16},
            ],
        },
    },
}


def format_last_updated(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def get_feature_display_name(feature_name: str) -> str:
    return FEATURE_DISPLAY_MAP.get(feature_name, feature_name.replace("_", " "))


def normalize_shap_values(model_result: Dict[str, Any]) -> list[dict]:
    shap_values = model_result.get("shap_values")
    if isinstance(shap_values, dict):
        normalized = [
            {"feature": key, "value": value}
            for key, value in shap_values.items()
        ]
    elif isinstance(shap_values, list):
        normalized = []
        for item in shap_values:
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

    normalized.sort(key=lambda item: abs(float(item.get("value", 0.0))), reverse=True)
    return normalized


def build_description(model_name: str, top_features_display: list[str]) -> str:
    if not top_features_display:
        return f"{MODEL_KR_NAME.get(model_name, model_name)}에 영향을 주는 주요 피처 정보가 없습니다."

    feature_text = ", ".join(top_features_display[:3])
    return f"{feature_text}가 {MODEL_KR_NAME.get(model_name, model_name)}에 크게 영향을 주었습니다."


def enrich_model_result(model_name: str, model_result: Dict[str, Any]) -> Dict[str, Any]:
    probability = float(model_result.get("probability", 0.0))
    sorted_shap = normalize_shap_values(model_result)
    top_shap = sorted_shap[:3]
    top_features_display = [
        get_feature_display_name(str(item.get("feature", "-")))
        for item in top_shap
    ]

    return {
        "probability": probability,
        "shap_values": sorted_shap,
        "top_features_display": top_features_display,
        "description": model_result.get(
            "description",
            build_description(model_name, top_features_display),
        ),
    }


def enrich_dashboard_data(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    meta = data.get("meta", {})

    enriched = {
        "patient": data.get("patient", {}),
        "meta": {
            "source": source,
            "source_label": "API 연결" if source == "api" else "Mock data",
            "last_updated": meta.get("last_updated"),
            "last_updated_display": format_last_updated(meta.get("last_updated")),
            "api_base_url": API_BASE_URL,
        },
        "models": {},
    }

    for model_name in MODEL_ORDER:
        raw_model = data.get("models", {}).get(model_name, {})
        enriched["models"][model_name] = enrich_model_result(model_name, raw_model)

    return enriched


def get_mock_dashboard_data() -> Dict[str, Any]:
    return enrich_dashboard_data(MOCK_DASHBOARD_DATA, source="mock")


def fetch_dashboard_data(
    use_mock_override: bool = False,
    use_mock_on_error: bool = True,
) -> Dict[str, Any]:
    if use_mock_override:
        return get_mock_dashboard_data()

    url = f"{API_BASE_URL.rstrip('/')}{PREDICTION_ENDPOINT}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return enrich_dashboard_data(response.json(), source="api")
    except requests.RequestException:
        if use_mock_on_error:
            return get_mock_dashboard_data()
        raise
