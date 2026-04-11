from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from api_client import (
    MODEL_KR_NAME,
    MODEL_ORDER,
    fetch_dashboard_data,
    get_feature_display_name,
)

st.set_page_config(
    page_title="ICU ClinSight Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette ──────────────────────────────────────────────────
T_PRIMARY   = "#0f172a"
T_SECONDARY = "#475569"
T_MUTED     = "#94a3b8"
APP_BG      = "#eef2f7"
CARD_BG     = "#ffffff"
CARD_BORDER = "#dde3ed"
CARD_SHADOW = "0 1px 3px rgba(15,23,42,.07), 0 6px 24px rgba(15,23,42,.05)"

RISK_HIGH    = "#dc2626"
RISK_HIGH_BG = "#fef2f2"
RISK_MOD     = "#d97706"
RISK_MOD_BG  = "#fffbeb"
RISK_LOW     = "#16a34a"
RISK_LOW_BG  = "#f0fdf4"

DONUT_TRACK = "#e8edf4"
SHAP_POS    = "#ef4444"
SHAP_NEG    = "#22c55e"


def _risk(p: float) -> tuple[str, str, str]:
    """(label, color, bg_color)"""
    if p >= 0.70:
        return "High", RISK_HIGH, RISK_HIGH_BG
    if p >= 0.40:
        return "Moderate", RISK_MOD, RISK_MOD_BG
    return "Low", RISK_LOW, RISK_LOW_BG


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        /* ── Global ── */
        .stApp {{
            background: {APP_BG};
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            color: {T_PRIMARY};
        }}
        .block-container {{
            max-width: 1600px;
            padding-top: 0;
            padding-bottom: 1.5rem;
        }}

        /* ── Anchor helpers (invisible, used for CSS :has targeting) ── */
        .patient-panel-anchor,
        .model-card-anchor {{
            display: none;
        }}

        /* ── Patient panel card ── */
        div[data-testid="stVerticalBlock"]:has(.patient-panel-anchor) {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            box-shadow: {CARD_SHADOW};
            padding: 1.3rem 1.2rem 1rem;
        }}

        /* ── Model cards ── */
        div[data-testid="stVerticalBlock"]:has(.model-card-anchor) {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            box-shadow: {CARD_SHADOW};
            padding: 1.1rem 1.1rem 1rem;
        }}

        /* ── Page header ── */
        .page-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.1rem 0 1.1rem;
            border-bottom: 1px solid {CARD_BORDER};
            margin-bottom: 1.1rem;
        }}
        .page-title {{
            font-size: 1.35rem;
            font-weight: 800;
            color: {T_PRIMARY};
            line-height: 1.2;
        }}
        .page-subtitle {{
            font-size: 0.76rem;
            color: {T_MUTED};
            margin-top: 0.2rem;
        }}
        .page-meta {{
            text-align: right;
        }}
        .page-meta-label {{
            font-size: 0.68rem;
            color: {T_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .page-meta-value {{
            font-size: 0.88rem;
            font-weight: 600;
            color: {T_SECONDARY};
            margin-top: 0.1rem;
        }}

        /* ── Patient panel internals ── */
        .panel-section-label {{
            font-size: 0.68rem;
            font-weight: 700;
            color: {T_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.8rem;
        }}
        .patient-name {{
            font-size: 1.28rem;
            font-weight: 800;
            color: {T_PRIMARY};
            line-height: 1.2;
            margin-bottom: 0.2rem;
        }}
        .patient-id {{
            font-size: 0.75rem;
            color: {T_MUTED};
            font-family: "Consolas", monospace;
            margin-bottom: 0.6rem;
        }}
        .demo-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 0.6rem;
        }}
        .demo-item {{
            background: #f8fafc;
            border: 1px solid #e8edf3;
            border-radius: 10px;
            padding: 0.5rem 0.65rem;
        }}
        .demo-label {{
            font-size: 0.65rem;
            font-weight: 700;
            color: {T_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.15rem;
        }}
        .demo-value {{
            font-size: 0.88rem;
            font-weight: 700;
            color: {T_PRIMARY};
        }}
        .divider {{
            height: 1px;
            background: #f1f5f9;
            margin: 0.8rem 0;
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            border: 1px solid;
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
        }}
        .data-info-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 0.5rem;
        }}
        .data-freshness {{
            font-size: 0.71rem;
            color: {T_MUTED};
        }}

        /* ── Model card internals ── */
        .model-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 0.2rem;
        }}
        .model-name {{
            font-size: 0.95rem;
            font-weight: 800;
            color: {T_PRIMARY};
            line-height: 1.3;
        }}
        .risk-badge {{
            padding: 0.22rem 0.6rem;
            border-radius: 8px;
            font-size: 0.68rem;
            font-weight: 800;
            white-space: nowrap;
        }}
        .card-section-label {{
            font-size: 0.65rem;
            font-weight: 700;
            color: {T_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-top: 0.55rem;
            margin-bottom: 0.35rem;
            padding-top: 0.55rem;
            border-top: 1px solid #f1f5f9;
        }}

        /* ── SHAP bars ── */
        .shap-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.38rem;
        }}
        .shap-name {{
            font-size: 0.72rem;
            color: {T_SECONDARY};
            width: 50%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex-shrink: 0;
        }}
        .shap-track {{
            flex: 1;
            height: 7px;
            background: #f1f5f9;
            border-radius: 4px;
            overflow: hidden;
        }}
        .shap-fill {{
            height: 100%;
            border-radius: 4px;
        }}
        .shap-badge {{
            font-size: 0.69rem;
            font-weight: 700;
            min-width: 46px;
            text-align: right;
            flex-shrink: 0;
        }}

        /* ── Feature measurement table ── */
        .feat-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.72rem;
        }}
        .feat-table th {{
            font-size: 0.64rem;
            color: {T_MUTED};
            font-weight: 700;
            padding: 0.15rem 0.2rem;
            border-bottom: 1px solid #e8edf3;
            text-align: left;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .feat-table td {{
            padding: 0.28rem 0.2rem;
            border-bottom: 1px solid #f8fafc;
            vertical-align: middle;
        }}
        .fn-cell {{ color: {T_SECONDARY}; }}
        .fv-cell {{ font-weight: 700; }}
        .fr-cell {{ color: {T_MUTED}; font-size: 0.67rem; }}
        .anom-hi {{ color: #dc2626; }}
        .anom-lo {{ color: #2563eb; }}
        .val-ok  {{ color: #16a34a; }}

        /* ── Description box ── */
        .desc-box {{
            background: #f8fafc;
            border-left: 3px solid #e2e8f0;
            border-radius: 0 8px 8px 0;
            padding: 0.55rem 0.7rem;
        }}
        .desc-text {{
            font-size: 0.76rem;
            color: {T_SECONDARY};
            line-height: 1.6;
        }}

        /* ── Section title (model results area) ── */
        .section-heading {{
            font-size: 0.75rem;
            font-weight: 700;
            color: {T_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.8rem;
        }}

        /* ── Streamlit overrides ── */
        div[data-testid="stCheckbox"] {{ margin-top: 0.6rem; }}
        div[data-testid="stCheckbox"] label {{ font-size: 0.78rem; color: {T_SECONDARY}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_donut(probability: float) -> go.Figure:
    label, color, _ = _risk(probability)
    v = max(0.0, min(probability, 1.0))

    fig = go.Figure(data=[go.Pie(
        values=[v, 1 - v],
        hole=0.70,
        sort=False,
        direction="clockwise",
        marker=dict(colors=[color, DONUT_TRACK]),
        textinfo="none",
        hoverinfo="skip",
        showlegend=False,
    )])
    fig.update_layout(
        height=165,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{v * 100:.1f}%</b>",
                x=0.5, y=0.57,
                showarrow=False,
                font=dict(size=22, color=T_PRIMARY),
            ),
            dict(
                text=label,
                x=0.5, y=0.40,
                showarrow=False,
                font=dict(size=11, color=T_SECONDARY),
            ),
        ],
    )
    return fig


def _shap_bars_html(shap_values: list) -> str:
    if not shap_values:
        return f'<div class="desc-text" style="color:{T_MUTED};">기여 요인 정보가 없습니다.</div>'

    top3 = shap_values[:3]
    max_abs = max(abs(x["value"]) for x in top3) or 1.0

    rows = ""
    for item in top3:
        feat = str(item.get("feature", ""))
        val = float(item.get("value", 0.0))
        name = get_feature_display_name(feat)
        pct = int(abs(val) / max_abs * 100)
        color = SHAP_POS if val >= 0 else SHAP_NEG
        arrow = "▲" if val >= 0 else "▼"

        rows += (
            f'<div class="shap-item">'
            f'<div class="shap-name">{html.escape(name)}</div>'
            f'<div class="shap-track">'
            f'<div class="shap-fill" style="width:{pct}%;background:{color};"></div>'
            f'</div>'
            f'<div class="shap-badge" style="color:{color};">{arrow}&nbsp;{abs(val):.2f}</div>'
            f'</div>'
        )
    return rows


def _feature_table_html(top_feature_values: list) -> str:
    if not top_feature_values:
        return ""

    rows = ""
    for fv in top_feature_values:
        raw = fv.get("value")
        if raw is None:
            continue

        # Format numeric value
        if isinstance(raw, float) and raw == int(raw):
            val_str = str(int(raw))
        elif isinstance(raw, float):
            val_str = f"{raw:.1f}"
        else:
            val_str = str(raw)

        unit = fv.get("unit", "")
        display_val = f"{val_str} {unit}".strip()

        is_anom = fv.get("is_abnormal", False)
        direction = fv.get("direction")
        if is_anom and direction == "high":
            val_class, indicator = "fv-cell anom-hi", " ↑"
        elif is_anom and direction == "low":
            val_class, indicator = "fv-cell anom-lo", " ↓"
        else:
            val_class, indicator = "fv-cell val-ok", ""

        range_str = html.escape(fv.get("normal_range_str") or "–")

        rows += (
            f"<tr>"
            f'<td class="fn-cell">{html.escape(fv.get("display_name", "-"))}</td>'
            f'<td class="{val_class}">{html.escape(display_val)}{indicator}</td>'
            f'<td class="fr-cell">{range_str}</td>'
            f"</tr>"
        )

    return (
        '<table class="feat-table">'
        "<thead><tr>"
        "<th>지표</th><th>측정값</th><th>정상범위</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def render_page_header(data: dict) -> None:
    meta = data.get("meta", {})
    updated = meta.get("last_updated_display", "-")
    source_label = html.escape(meta.get("source_label", "-"))
    is_mock = meta.get("source") == "mock"
    src_color = "#94a3b8" if is_mock else "#2563eb"
    src_bg    = "#f8fafc"  if is_mock else "#eff6ff"
    src_border = "#e2e8f0" if is_mock else "#bfdbfe"

    st.markdown(
        f"""
        <div class="page-header">
          <div>
            <div class="page-title">ICU ClinSight Dashboard</div>
            <div class="page-subtitle">패혈증 환자 예후 예측 · 중환자실 임상 의사결정 지원 시스템</div>
          </div>
          <div class="page-meta">
            <div style="margin-bottom:0.3rem;">
              <span class="badge" style="color:{src_color};border-color:{src_border};background:{src_bg};">
                {source_label}
              </span>
            </div>
            <div class="page-meta-label">마지막 업데이트</div>
            <div class="page-meta-value">{html.escape(updated)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_panel(data: dict, use_mock_data: bool) -> None:
    patient = data.get("patient", {})

    age = patient.get("age", "-")
    age_str = f"{age}세" if age != "-" else "-"
    gender_map = {"Male": "남성", "Female": "여성"}
    gender = gender_map.get(str(patient.get("gender", "")), patient.get("gender", "-"))
    diagnosis = html.escape(str(patient.get("diagnosis", "-")))
    ward = html.escape(str(patient.get("ward", "-")))
    admit_date = html.escape(str(patient.get("admit_date", "-")))

    with st.container():
        st.markdown('<div class="patient-panel-anchor"></div>', unsafe_allow_html=True)

        # Section label + ICU status badge
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.9rem;">'
            '<div class="panel-section-label">환자 정보</div>'
            '<span class="badge" style="color:#2563eb;border-color:#bfdbfe;background:#eff6ff;">ICU 입원 중</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Name + ID + diagnosis badge
        st.markdown(
            f'<div class="patient-name">{html.escape(str(patient.get("name", "-")))}</div>'
            f'<div class="patient-id">{html.escape(str(patient.get("patient_id", "-")))}</div>'
            f'<div style="margin-bottom:0.75rem;">'
            f'<span class="badge" style="color:#7c3aed;border-color:#ddd6fe;background:#f5f3ff;">'
            f'{diagnosis}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Demographics 2×2 grid
        st.markdown(
            f'<div class="demo-grid">'
            f'<div class="demo-item"><div class="demo-label">나이</div>'
            f'<div class="demo-value">{html.escape(age_str)}</div></div>'
            f'<div class="demo-item"><div class="demo-label">성별</div>'
            f'<div class="demo-value">{html.escape(str(gender))}</div></div>'
            f'<div class="demo-item"><div class="demo-label">병동</div>'
            f'<div class="demo-value">{ward}</div></div>'
            f'<div class="demo-item"><div class="demo-label">입원일</div>'
            f'<div class="demo-value">{admit_date}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Data source section
        st.markdown('<div class="panel-section-label">데이터 현황</div>', unsafe_allow_html=True)
        st.checkbox("Mock 데이터 사용", value=use_mock_data, key="use_mock_data")


def render_model_card(model_name: str, model_result: dict) -> None:
    prob = float(model_result.get("probability", 0.0))
    label, color, bg_color = _risk(prob)
    kr_name = MODEL_KR_NAME.get(model_name, model_name)

    with st.container():
        st.markdown('<div class="model-card-anchor"></div>', unsafe_allow_html=True)

        # ── Header: name + risk badge ──────────────────────────
        st.markdown(
            f'<div class="model-header">'
            f'<div class="model-name">{html.escape(kr_name)}</div>'
            f'<div class="risk-badge" style="color:{color};background:{bg_color};">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Donut chart ────────────────────────────────────────
        st.plotly_chart(
            build_donut(prob),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"donut_{model_name}",
        )

        # ── SHAP: 주요 기여 요인 ──────────────────────────────
        st.markdown(
            '<div class="card-section-label">주요 기여 요인 (SHAP)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _shap_bars_html(model_result.get("shap_values", [])),
            unsafe_allow_html=True,
        )

        # ── Feature measurements: 핵심 지표 측정값 ────────────
        st.markdown(
            '<div class="card-section-label">핵심 지표 측정값</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _feature_table_html(model_result.get("top_feature_values", [])),
            unsafe_allow_html=True,
        )

        # ── Clinical interpretation ────────────────────────────
        st.markdown(
            '<div class="card-section-label">임상 해석</div>',
            unsafe_allow_html=True,
        )
        desc = html.escape(str(model_result.get("description", "-")))
        st.markdown(
            f'<div class="desc-box"><div class="desc-text">{desc}</div></div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()

    if "use_mock_data" not in st.session_state:
        st.session_state["use_mock_data"] = True

    dashboard_data = fetch_dashboard_data(
        use_mock_override=bool(st.session_state["use_mock_data"]),
        use_mock_on_error=True,
    )

    render_page_header(dashboard_data)

    left_col, right_col = st.columns([1, 3], gap="large")

    with left_col:
        render_patient_panel(
            dashboard_data,
            use_mock_data=bool(st.session_state["use_mock_data"]),
        )

    with right_col:
        st.markdown(
            '<div class="section-heading">모델 예측 결과</div>',
            unsafe_allow_html=True,
        )
        row1 = st.columns(2, gap="medium")
        row2 = st.columns(2, gap="medium")
        grid = [row1, row2]

        for idx, model_name in enumerate(MODEL_ORDER):
            with grid[idx // 2][idx % 2]:
                render_model_card(
                    model_name,
                    dashboard_data.get("models", {}).get(model_name, {}),
                )


if __name__ == "__main__":
    main()
