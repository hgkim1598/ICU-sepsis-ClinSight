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

ACCENT_BLUE = "#2563eb"


def _risk(p: float) -> tuple[str, str, str]:
    """(label, color, bg_color)"""
    if p >= 0.70:
        return "High", RISK_HIGH, RISK_HIGH_BG
    if p >= 0.40:
        return "Moderate", RISK_MOD, RISK_MOD_BG
    return "Low", RISK_LOW, RISK_LOW_BG


def _sofa_style(score) -> tuple[str, str]:
    """Return (text_color, bg_color) for SOFA score badge."""
    try:
        s = int(score)
    except (TypeError, ValueError):
        return ("#64748b", "#f1f5f9")
    if s >= 13:
        return (RISK_HIGH, RISK_HIGH_BG)
    if s >= 7:
        return (RISK_MOD, RISK_MOD_BG)
    return (RISK_LOW, RISK_LOW_BG)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        /* ── Hide Streamlit default header / toolbar (검은 바 제거) ── */
        header[data-testid="stHeader"] {{
            display: none !important;
            height: 0 !important;
            visibility: hidden !important;
        }}
        div[data-testid="stToolbar"] {{
            display: none !important;
        }}
        div[data-testid="stDecoration"] {{
            display: none !important;
        }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}

        /* ── Global ── */
        .stApp {{
            background: {APP_BG};
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            color: {T_PRIMARY};
        }}
        .block-container {{
            max-width: 1700px;
            padding-top: 1.4rem;
            padding-bottom: 1.4rem;
        }}
        div[data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}
        div[data-testid="stHorizontalBlock"] {{ gap: 0.7rem; }}

        /* ── Anchor helpers ── */
        .patient-bar-anchor,
        .summary-card-anchor,
        .summary-card-selected-anchor,
        .detail-panel-anchor {{
            display: none;
        }}

        /* ── Page header ── */
        .page-header-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.2rem 0 0.6rem;
        }}
        .page-title {{
            font-size: 1.2rem;
            font-weight: 800;
            color: {T_PRIMARY};
            line-height: 1.2;
        }}
        .page-subtitle {{
            font-size: 0.72rem;
            color: {T_MUTED};
            margin-top: 0.1rem;
        }}
        .page-meta {{
            text-align: right;
            font-size: 0.72rem;
            color: {T_MUTED};
        }}
        .page-meta-value {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {T_SECONDARY};
        }}

        /* ── Patient bar (top, single row) ── */
        .patient-bar {{
            background: linear-gradient(135deg, #eff6ff 0%, #e0ecfd 100%);
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 0.75rem 1.2rem;
            display: flex;
            align-items: center;
            gap: 1.1rem;
            flex-wrap: nowrap;
            margin-bottom: 1.6rem;
        }}
        .pb-item {{
            display: flex;
            flex-direction: column;
            gap: 0.05rem;
            min-width: 0;
        }}
        .pb-item-inline {{
            display: flex;
            align-items: baseline;
            gap: 0.4rem;
        }}
        .pb-label {{
            font-size: 0.62rem;
            color: #1e40af;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .pb-value {{
            font-size: 0.92rem;
            color: {T_PRIMARY};
            font-weight: 700;
            line-height: 1.2;
            white-space: nowrap;
        }}
        .pb-name {{
            font-size: 1.05rem;
            font-weight: 800;
        }}
        .pb-divider {{
            width: 1px;
            height: 28px;
            background: #bfdbfe;
            flex-shrink: 0;
        }}
        .pb-sofa-badge {{
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 800;
            line-height: 1.1;
        }}

        /* ── Section heading (above summary cards / detail panel) ── */
        .section-heading {{
            font-size: 0.72rem;
            font-weight: 700;
            color: {T_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin: 0.4rem 0 0.7rem;
        }}

        /* ── Summary cards (4 across) — entire card is clickable ── */
        div[data-testid="stVerticalBlock"]:has(.summary-card-anchor) {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 0.85rem 0.95rem 0.85rem;
            box-shadow: 0 1px 2px rgba(15,23,42,.04);
            opacity: 0.78;
            transition: opacity 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
            position: relative;
            cursor: pointer;
        }}
        div[data-testid="stVerticalBlock"]:has(.summary-card-anchor):hover {{
            opacity: 1;
            box-shadow: 0 2px 6px rgba(15,23,42,.08), 0 8px 20px rgba(15,23,42,.06);
            transform: translateY(-1px);
        }}
        div[data-testid="stVerticalBlock"]:has(.summary-card-selected-anchor) {{
            background: {CARD_BG};
            border: 1px solid {ACCENT_BLUE};
            border-bottom: 4px solid {ACCENT_BLUE};
            border-radius: 14px;
            padding: 0.85rem 0.95rem calc(0.85rem - 3px);
            box-shadow: 0 1px 3px rgba(15,23,42,.07), 0 6px 18px rgba(37,99,235,.13);
            opacity: 1;
            position: relative;
            cursor: pointer;
        }}

        .sc-name {{
            font-size: 0.85rem;
            font-weight: 800;
            color: {T_PRIMARY};
            text-align: center;
            margin-bottom: 0.1rem;
        }}
        .sc-meta-row {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-top: -0.2rem;
            margin-bottom: 0.3rem;
        }}
        .sc-percent {{
            font-size: 1.2rem;
            font-weight: 800;
            color: {T_PRIMARY};
        }}
        .sc-risk-badge {{
            padding: 0.22rem 0.6rem;
            border-radius: 8px;
            font-size: 0.72rem;
            font-weight: 800;
        }}

        /* Force element wrappers inside the card to not trap absolute positioning,
           so the invisible button stretches across the whole stVerticalBlock */
        div[data-testid="stVerticalBlock"]:has(.summary-card-anchor) [data-testid="stElementContainer"],
        div[data-testid="stVerticalBlock"]:has(.summary-card-selected-anchor) [data-testid="stElementContainer"] {{
            position: static;
        }}

        /* Invisible button overlay — entire card acts as clickable area */
        div[data-testid="stVerticalBlock"]:has(.summary-card-anchor) div[data-testid="stButton"],
        div[data-testid="stVerticalBlock"]:has(.summary-card-selected-anchor) div[data-testid="stButton"] {{
            position: absolute;
            inset: 0;
            margin: 0;
            max-width: none;
            width: 100%;
            height: 100%;
            z-index: 20;
        }}
        div[data-testid="stVerticalBlock"]:has(.summary-card-anchor) button[kind],
        div[data-testid="stVerticalBlock"]:has(.summary-card-selected-anchor) button[kind] {{
            width: 100%;
            height: 100%;
            opacity: 0;
            background: transparent;
            border: none;
            padding: 0;
            min-height: 0;
            cursor: pointer;
        }}
        div[data-testid="stVerticalBlock"]:has(.summary-card-anchor) button[kind]:focus,
        div[data-testid="stVerticalBlock"]:has(.summary-card-selected-anchor) button[kind]:focus {{
            outline: none;
            box-shadow: none;
        }}

        /* ── Detail panel ── */
        div[data-testid="stVerticalBlock"]:has(.detail-panel-anchor) {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            box-shadow: {CARD_SHADOW};
            padding: 1.4rem 1.6rem 1.3rem;
            margin-top: 1.6rem;
        }}
        .detail-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid #f1f5f9;
        }}
        .detail-title-text {{
            font-size: 0.95rem;
            font-weight: 800;
            color: {T_PRIMARY};
        }}
        .detail-title-meta {{
            font-size: 0.72rem;
            color: {T_MUTED};
            font-weight: 600;
        }}

        .card-section-label {{
            font-size: 0.7rem;
            font-weight: 700;
            color: {T_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.7rem;
            margin-top: 0.3rem;
        }}

        /* ── SHAP bars ── */
        .shap-item {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.75rem;
        }}
        .shap-name {{
            font-size: 0.78rem;
            color: {T_SECONDARY};
            width: 38%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex-shrink: 0;
        }}
        .shap-track {{
            flex: 1;
            height: 9px;
            background: #f1f5f9;
            border-radius: 4px;
            overflow: hidden;
        }}
        .shap-fill {{
            height: 100%;
            border-radius: 4px;
        }}
        .shap-badge {{
            font-size: 0.74rem;
            font-weight: 700;
            min-width: 54px;
            text-align: right;
            flex-shrink: 0;
        }}

        /* ── Feature table ── */
        .feat-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
        }}
        .feat-table th {{
            font-size: 0.66rem;
            color: {T_MUTED};
            font-weight: 700;
            padding: 0.2rem 0.3rem;
            border-bottom: 1px solid #e8edf3;
            text-align: left;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .feat-table td {{
            padding: 0.5rem 0.3rem;
            border-bottom: 1px solid #f8fafc;
            vertical-align: middle;
        }}
        .fn-cell {{ color: {T_SECONDARY}; }}
        .fv-cell {{ font-weight: 700; }}
        .fr-cell {{ color: {T_MUTED}; font-size: 0.7rem; }}
        .anom-hi {{ color: #dc2626; }}
        .anom-lo {{ color: #2563eb; }}
        .val-ok  {{ color: #16a34a; }}

        /* ── Description box ── */
        .desc-box {{
            background: #f8fafc;
            border-left: 3px solid {ACCENT_BLUE};
            border-radius: 0 8px 8px 0;
            padding: 0.85rem 1rem;
            margin-top: 0.3rem;
        }}
        .desc-text {{
            font-size: 0.78rem;
            color: {T_SECONDARY};
            line-height: 1.6;
        }}

        /* ── Streamlit overrides ── */
        div[data-testid="stCheckbox"] label {{
            font-size: 0.75rem;
            color: {T_SECONDARY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Chart / HTML builders
# ─────────────────────────────────────────────────────────────
def build_summary_donut(probability: float) -> go.Figure:
    """Compact donut for summary cards."""
    _, color, _ = _risk(probability)
    v = max(0.0, min(probability, 1.0))

    fig = go.Figure(data=[go.Pie(
        values=[v, 1 - v],
        hole=0.74,
        sort=False,
        direction="clockwise",
        marker=dict(colors=[color, DONUT_TRACK]),
        textinfo="none",
        hoverinfo="skip",
        showlegend=False,
    )])
    fig.update_layout(
        height=140,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{v * 100:.1f}%</b>",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color=T_PRIMARY),
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


# ─────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────
def render_page_header(data: dict) -> None:
    meta = data.get("meta", {})
    updated = html.escape(meta.get("last_updated_display", "-"))
    source_label = html.escape(meta.get("source_label", "-"))
    is_mock = meta.get("source") == "mock"
    src_color = "#94a3b8" if is_mock else ACCENT_BLUE

    h_left, h_right = st.columns([3, 2], gap="small")
    with h_left:
        st.markdown(
            '<div class="page-header-row">'
            '<div>'
            '<div class="page-title">ICU ClinSight Dashboard</div>'
            '<div class="page-subtitle">패혈증 환자 예후 예측 · 중환자실 임상 의사결정 지원 시스템</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with h_right:
        st.markdown(
            f'<div class="page-meta" style="padding-top:0.3rem;">'
            f'<span style="color:{src_color};font-weight:700;">● {source_label}</span>'
            f' &nbsp;·&nbsp; '
            f'마지막 업데이트 <span class="page-meta-value">{updated}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_patient_bar(data: dict) -> None:
    p = data.get("patient", {})

    name = html.escape(str(p.get("name", "-")))
    age = p.get("age", "-")
    age_str = f"{age}세" if age != "-" else "-"
    gender_map = {"Male": "남성", "Female": "여성"}
    gender = html.escape(gender_map.get(str(p.get("gender", "")), str(p.get("gender", "-"))))
    icu_admit = html.escape(str(p.get("icu_admit_time", "-")))
    sepsis_onset = html.escape(str(p.get("sepsis_onset", "-")))
    sofa = p.get("sofa_score", "-")
    sofa_color, sofa_bg = _sofa_style(sofa)
    sofa_str = html.escape(str(sofa))

    st.markdown(
        f"""
        <div class="patient-bar">
          <div class="pb-item">
            <span class="pb-label">환자</span>
            <span class="pb-value pb-name">{name}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">나이</span>
            <span class="pb-value">{html.escape(age_str)}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">성별</span>
            <span class="pb-value">{gender}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">ICU 입실</span>
            <span class="pb-value">{icu_admit}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">SOFA 점수</span>
            <span class="pb-sofa-badge" style="color:{sofa_color};background:{sofa_bg};">{sofa_str}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">패혈증 onset</span>
            <span class="pb-value">{sepsis_onset}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _select_model(model_name: str) -> None:
    st.session_state["selected_model"] = model_name


def render_summary_cards(data: dict) -> None:
    selected = st.session_state.get("selected_model", MODEL_ORDER[0])
    cols = st.columns(4, gap="small")

    for i, model_name in enumerate(MODEL_ORDER):
        model_result = data.get("models", {}).get(model_name, {})
        prob = float(model_result.get("probability", 0.0))
        label, color, bg_color = _risk(prob)
        kr_name = MODEL_KR_NAME.get(model_name, model_name)
        is_selected = (selected == model_name)

        with cols[i]:
            anchor = "summary-card-selected-anchor" if is_selected else "summary-card-anchor"
            st.markdown(f'<div class="{anchor}"></div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="sc-name">{html.escape(kr_name)}</div>',
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                build_summary_donut(prob),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"summary_donut_{model_name}",
            )

            st.markdown(
                f'<div class="sc-meta-row">'
                f'<span class="sc-risk-badge" style="color:{color};background:{bg_color};">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Invisible button overlay covering the entire card via CSS
            st.button(
                kr_name,
                key=f"sel_{model_name}",
                on_click=_select_model,
                args=(model_name,),
                use_container_width=True,
            )


def render_detail_panel(data: dict) -> None:
    selected = st.session_state.get("selected_model", MODEL_ORDER[0])
    model_result = data.get("models", {}).get(selected, {})
    kr_name = MODEL_KR_NAME.get(selected, selected)
    prob = float(model_result.get("probability", 0.0))
    label, color, bg_color = _risk(prob)

    with st.container():
        st.markdown('<div class="detail-panel-anchor"></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="detail-title">'
            f'<div class="detail-title-text">{html.escape(kr_name)} · 상세 분석</div>'
            f'<div>'
            f'<span class="sc-risk-badge" style="color:{color};background:{bg_color};">{label} · {prob*100:.1f}%</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                '<div class="card-section-label">주요 기여 요인 (SHAP)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _shap_bars_html(model_result.get("shap_values", [])),
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                '<div class="card-section-label">핵심 지표 측정값</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _feature_table_html(model_result.get("top_feature_values", [])),
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="card-section-label" style="margin-top:1.3rem;">임상 해석</div>',
                unsafe_allow_html=True,
            )
            desc = html.escape(str(model_result.get("description", "-")))
            st.markdown(
                f'<div class="desc-box"><div class="desc-text">{desc}</div></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    inject_styles()

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = MODEL_ORDER[0]
    if "use_mock_data" not in st.session_state:
        st.session_state["use_mock_data"] = True

    dashboard_data = fetch_dashboard_data(
        use_mock_override=bool(st.session_state["use_mock_data"]),
        use_mock_on_error=True,
    )

    render_page_header(dashboard_data)
    render_patient_bar(dashboard_data)

    st.markdown(
        '<div class="section-heading">모델 위험도 요약</div>',
        unsafe_allow_html=True,
    )
    render_summary_cards(dashboard_data)
    render_detail_panel(dashboard_data)


if __name__ == "__main__":
    main()
