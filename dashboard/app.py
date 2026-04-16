from __future__ import annotations

import html
import json
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from api_client import (
    MODEL_KR_NAME,
    MODEL_ORDER,
    fetch_dashboard_data,
    fetch_patient_data,
    fetch_patients,
    fetch_predictions,
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

        /* ── CSS custom properties (light / dark) ── */
        :root {{
            --c-primary:   {T_PRIMARY};
            --c-secondary: {T_SECONDARY};
            --c-muted:     {T_MUTED};
        }}
        body.cs-dark {{
            --c-primary:   #f1f5f9;
            --c-secondary: #cbd5e1;
            --c-muted:     #94a3b8;
        }}

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

        /* ═══════════════════════════════════════════════════════
           신규 추가: 사이드바 & 새로고침 버튼
           (기존 컴포넌트/클래스와 독립된 cs- 네임스페이스)
           ═══════════════════════════════════════════════════════ */

        /* Streamlit의 기본 사이드바는 hover peek 등을 유발하므로 완전히 숨김 */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapseButton"],
        div[data-testid="collapsedControl"] {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            min-width: 0 !important;
            pointer-events: none !important;
        }}

        /* ── Streamlit main block container 높이 고정 ──
           hover 시 stMainBlockContainer가 height: auto로 줄어드는 현상이 있어
           사이드바 트리거/인식 영역이 왔다갔다 하는 것처럼 보임.
           항상 전체 화면 높이를 유지하도록 강제. */
        [data-testid="stMainBlockContainer"] {{
            min-height: 100vh !important;
            height: 100vh !important;
        }}

        /* ── 햄버거 버튼 (사이드바 열림 시 사이드바 우측 상단으로 이동) ── */
        .cs-hamburger {{
            position: fixed;
            top: 14px;
            left: 14px;
            z-index: 1050;
            width: 34px;
            height: 34px;
            border: 1px solid {CARD_BORDER};
            background: {CARD_BG};
            border-radius: 8px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            box-shadow: 0 1px 3px rgba(15,23,42,.08);
            color: {T_PRIMARY};
            font-size: 16px;
            /* hover에 반응하는 속성 없음 — 위치는 오직 body.cs-open 상태로만 변함 */
            transition: left 0.28s cubic-bezier(.4,0,.2,1);
        }}
        .cs-hamburger-icon {{
            display: inline-flex;
            flex-direction: column;
            gap: 4px;
        }}
        .cs-hamburger-icon span {{
            width: 16px;
            height: 2px;
            background: {T_PRIMARY};
            border-radius: 1px;
        }}
        body.cs-open .cs-hamburger {{
            /* 사이드바(320px)의 우측 끝 상단으로 이동 */
            left: calc(320px - 34px - 14px);
        }}

        /* ── 슬라이드 사이드바 (라이트 테마, 화면 전체 높이 고정) ── */
        .cs-sidebar {{
            position: fixed;
            top: 0;
            left: 0;
            /* 마우스 위치/부모 크기와 무관하게 항상 100vh — min/max 모두 고정 */
            height: 100vh !important;
            min-height: 100vh !important;
            max-height: 100vh !important;
            width: 320px;
            min-width: 320px;
            max-width: 320px;
            background: #f8fafc;
            color: #1e293b;
            z-index: 1045;
            transform: translateX(-100%);
            /* 닫힘 상태: 마우스 이벤트 완전 차단 */
            pointer-events: none;
            transition: transform 0.28s cubic-bezier(.4,0,.2,1);
            display: flex;
            flex-direction: column;
            box-shadow: 2px 0 12px rgba(15,23,42,.1);
            overflow: hidden;
        }}
        /* 열림 상태 — transform/pointer-events만 변경, hover 없음 */
        body.cs-open .cs-sidebar {{
            transform: translateX(0);
            pointer-events: auto;
        }}

        .cs-sidebar-header {{
            padding: 1.1rem 1.1rem 0.9rem;
            border-bottom: 1px solid #e2e8f0;
            flex-shrink: 0;
            background: #ffffff;
        }}
        .cs-sidebar-title {{
            font-size: 0.92rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: 0.01em;
        }}
        .cs-sidebar-subtitle {{
            font-size: 0.68rem;
            color: #64748b;
            margin-top: 0.15rem;
        }}
        .cs-search-wrap {{
            position: relative;
            margin-top: 0.8rem;
        }}
        .cs-search {{
            width: 100%;
            padding: 0.5rem 0.7rem 0.5rem 2rem;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            color: #1e293b;
            font-size: 0.76rem;
            outline: none;
            box-sizing: border-box;
            font-family: inherit;
        }}
        .cs-search::placeholder {{ color: #94a3b8; }}
        .cs-search:focus {{
            border-color: {ACCENT_BLUE};
            background: #ffffff;
        }}
        .cs-search-icon {{
            position: absolute;
            left: 0.65rem;
            top: 50%;
            transform: translateY(-50%);
            color: #94a3b8;
            font-size: 0.8rem;
            pointer-events: none;
        }}

        /* ── 환자 리스트 (스크롤 없음, 페이지당 4명) ── */
        .cs-patient-list {{
            flex: 1;
            overflow: hidden;
            padding: 0.7rem 0.8rem 0.4rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}
        .cs-patient-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0.75rem;
            border-radius: 8px;
            cursor: pointer;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            transition: background 0.12s, border-color 0.12s;
        }}
        .cs-patient-item:hover {{
            background: #dbeafe;
            border-color: #bfdbfe;
        }}
        .cs-patient-item.is-selected {{
            background: #dbeafe;
            border-color: {ACCENT_BLUE};
        }}
        .cs-patient-item.is-hidden {{ display: none; }}

        .cs-patient-info {{
            display: flex;
            flex-direction: column;
            min-width: 0;
        }}
        .cs-patient-name {{
            font-size: 0.84rem;
            font-weight: 700;
            color: #0f172a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .cs-patient-id {{
            font-size: 0.66rem;
            color: #64748b;
            margin-top: 0.1rem;
        }}
        .cs-patient-meta {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.22rem;
            flex-shrink: 0;
            margin-left: 0.6rem;
        }}
        .cs-sofa {{
            font-size: 0.68rem;
            color: #64748b;
        }}
        .cs-sofa-val {{ font-weight: 800; color: #0f172a; }}
        .cs-risk-badge {{
            padding: 0.14rem 0.45rem;
            border-radius: 6px;
            font-size: 0.6rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        /* 위험도 뱃지 색상은 유지 (기존 대시보드 색상 참조) */
        .cs-risk-High     {{ background: {RISK_HIGH_BG}; color: {RISK_HIGH}; }}
        .cs-risk-Moderate {{ background: {RISK_MOD_BG};  color: {RISK_MOD};  }}
        .cs-risk-Low      {{ background: {RISK_LOW_BG};  color: {RISK_LOW};  }}

        .cs-empty {{
            text-align: center;
            color: #94a3b8;
            font-size: 0.76rem;
            padding: 1.5rem 0.5rem;
        }}

        /* ── 페이지네이션 ── */
        .cs-pagination {{
            flex-shrink: 0;
            padding: 0.7rem 0.9rem 1rem;
            border-top: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.3rem;
            background: #ffffff;
        }}
        .cs-page-btn {{
            min-width: 28px;
            height: 28px;
            padding: 0 0.5rem;
            border-radius: 6px;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #475569;
            font-size: 0.74rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-family: inherit;
        }}
        .cs-page-btn:hover:not(:disabled) {{
            background: #dbeafe;
            border-color: #bfdbfe;
            color: #1e40af;
        }}
        .cs-page-btn:disabled {{
            opacity: 0.35;
            cursor: not-allowed;
        }}
        .cs-page-btn.is-active {{
            background: {ACCENT_BLUE};
            border-color: {ACCENT_BLUE};
            color: #ffffff;
        }}

        /* ── 새로고침 버튼 (헤더 우측) ── */
        .cs-refresh-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 0.55rem;
            width: 22px;
            height: 22px;
            border-radius: 6px;
            background: transparent;
            border: 1px solid {CARD_BORDER};
            color: {T_SECONDARY};
            cursor: pointer;
            vertical-align: middle;
            transition: background 0.15s, color 0.15s, border-color 0.15s;
            padding: 0;
            line-height: 1;
        }}
        .cs-refresh-btn svg {{
            width: 12px;
            height: 12px;
            transition: transform 0.4s ease;
        }}
        .cs-refresh-btn:hover {{
            background: #eff6ff;
            color: {ACCENT_BLUE};
            border-color: #93c5fd;
        }}
        .cs-refresh-btn:hover svg,
        .cs-refresh-btn.is-spinning svg {{
            animation: cs-spin 0.8s linear infinite;
        }}
        @keyframes cs-spin {{
            from {{ transform: rotate(0deg); }}
            to   {{ transform: rotate(360deg); }}
        }}

        /* ── 테마 토글 버튼 (새로고침 버튼 옆) ── */
        .cs-theme-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 0.4rem;
            width: 22px;
            height: 22px;
            border-radius: 6px;
            background: transparent;
            border: 1px solid {CARD_BORDER};
            color: {T_SECONDARY};
            cursor: pointer;
            vertical-align: middle;
            transition: background 0.15s, border-color 0.15s;
            padding: 0;
            line-height: 1;
            font-size: 11px;
        }}
        .cs-theme-btn:hover {{
            background: #eff6ff;
            border-color: #93c5fd;
        }}
        .cs-theme-icon {{ font-size: 11px; line-height: 1; }}

        /* 숨겨진 Streamlit trigger 버튼 / height=0 iframe은 JS에서 직접 스타일 처리 */

        /* ═══════════════════════════════════════════════════════
           다크모드 오버라이드 (body.cs-dark)
           기존 라이트 스타일은 수정하지 않고, 같은 셀렉터를 더 강한
           명시도로 덮어쓰는 방식
           ═══════════════════════════════════════════════════════ */
        body.cs-dark .stApp {{
            background: #0f172a;
            color: #e2e8f0;
        }}
        body.cs-dark .page-title {{ color: #f1f5f9; }}
        body.cs-dark .page-subtitle,
        body.cs-dark .page-meta {{ color: #94a3b8; }}
        body.cs-dark .page-meta-value {{ color: #cbd5e1; }}

        /* Patient bar */
        body.cs-dark .patient-bar {{
            background: linear-gradient(135deg, #1e293b 0%, #1e3a5f 100%);
            border-color: #334155;
        }}
        body.cs-dark .pb-label {{ color: #93c5fd; }}
        body.cs-dark .pb-value {{ color: #f1f5f9; }}
        body.cs-dark .pb-divider {{ background: #334155; }}

        /* Section heading */
        body.cs-dark .section-heading {{ color: #94a3b8; }}

        /* Summary cards */
        body.cs-dark div[data-testid="stVerticalBlock"]:has(.summary-card-anchor) {{
            background: #1e293b;
            border-color: #334155;
        }}
        body.cs-dark div[data-testid="stVerticalBlock"]:has(.summary-card-selected-anchor) {{
            background: #1e293b;
            border-color: #60a5fa;
            border-bottom-color: #60a5fa;
        }}
        body.cs-dark .sc-name,
        body.cs-dark .sc-percent {{ color: #f1f5f9; }}

        /* Plotly 도넛 중앙 퍼센트 텍스트는 SVG로 렌더링되어
           Python 측에서 지정한 색(T_PRIMARY)이 SVG fill로 박혀있음.
           다크모드에서 보이도록 fill을 덮어씀. */
        body.cs-dark div.js-plotly-plot text,
        body.cs-dark .stPlotlyChart text,
        body.cs-dark .stPlotlyChart .annotation-text {{
            fill: #f1f5f9 !important;
        }}

        /* Detail panel */
        body.cs-dark div[data-testid="stVerticalBlock"]:has(.detail-panel-anchor) {{
            background: #1e293b;
            border-color: #334155;
        }}
        body.cs-dark .detail-title {{ border-bottom-color: #334155; }}
        body.cs-dark .detail-title-text {{ color: #f1f5f9; }}
        body.cs-dark .detail-title-meta,
        body.cs-dark .card-section-label {{ color: #94a3b8; }}

        /* SHAP */
        body.cs-dark .shap-name {{ color: #cbd5e1; }}
        body.cs-dark .shap-track {{ background: #334155; }}

        /* Feature table */
        body.cs-dark .feat-table th {{
            color: #94a3b8;
            border-bottom-color: #334155;
        }}
        body.cs-dark .feat-table td {{ border-bottom-color: #253145; }}
        body.cs-dark .fn-cell {{ color: #cbd5e1; }}
        body.cs-dark .fr-cell {{ color: #94a3b8; }}

        /* Description box */
        body.cs-dark .desc-box {{
            background: #253145;
            border-left-color: #60a5fa;
        }}
        body.cs-dark .desc-text {{ color: #cbd5e1; }}

        /* Sidebar — dark variant */
        body.cs-dark .cs-sidebar {{
            background: #0f172a;
            color: #e2e8f0;
            box-shadow: 2px 0 16px rgba(0,0,0,.45);
        }}
        body.cs-dark .cs-sidebar-header {{
            background: #1e293b;
            border-bottom-color: #334155;
        }}
        body.cs-dark .cs-sidebar-title {{ color: #f1f5f9; }}
        body.cs-dark .cs-sidebar-subtitle,
        body.cs-dark .cs-empty {{ color: #94a3b8; }}
        body.cs-dark .cs-search {{
            background: #1e293b;
            border-color: #334155;
            color: #e2e8f0;
        }}
        body.cs-dark .cs-search::placeholder {{ color: #64748b; }}
        body.cs-dark .cs-search:focus {{
            border-color: #60a5fa;
            background: #253145;
        }}
        body.cs-dark .cs-search-icon {{ color: #64748b; }}
        body.cs-dark .cs-patient-item {{
            background: #1e293b;
            border-color: #334155;
        }}
        body.cs-dark .cs-patient-item:hover {{
            background: #253145;
            border-color: #475569;
        }}
        body.cs-dark .cs-patient-item.is-selected {{
            background: rgba(37,99,235,.22);
            border-color: #60a5fa;
        }}
        body.cs-dark .cs-patient-name,
        body.cs-dark .cs-sofa-val {{ color: #f1f5f9; }}
        body.cs-dark .cs-patient-id {{ color: #94a3b8; }}
        body.cs-dark .cs-sofa {{ color: #cbd5e1; }}
        body.cs-dark .cs-pagination {{
            background: #1e293b;
            border-top-color: #334155;
        }}
        body.cs-dark .cs-page-btn {{
            background: #1e293b;
            border-color: #334155;
            color: #cbd5e1;
        }}
        body.cs-dark .cs-page-btn:hover:not(:disabled) {{
            background: #253145;
            border-color: #475569;
            color: #f1f5f9;
        }}

        /* Hamburger / header buttons in dark */
        body.cs-dark .cs-hamburger {{
            background: #1e293b;
            border-color: #334155;
            color: #f1f5f9;
        }}
        body.cs-dark .cs-hamburger-icon span {{ background: #f1f5f9; }}
        body.cs-dark .cs-refresh-btn,
        body.cs-dark .cs-theme-btn {{
            border-color: #334155;
            color: #cbd5e1;
        }}
        body.cs-dark .cs-refresh-btn:hover,
        body.cs-dark .cs-theme-btn:hover {{
            background: #253145;
            border-color: #475569;
            color: #f1f5f9;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Chart / HTML builders
# ─────────────────────────────────────────────────────────────
def build_summary_donut(probability: float, is_api: bool = True) -> go.Figure:
    """Compact donut for summary cards. is_api=False면 회색 톤."""
    if is_api:
        _, color, _ = _risk(probability)
        text_color = T_PRIMARY
    else:
        color = "#888888"
        text_color = "#888888"
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
                font=dict(size=20, color=text_color),
            ),
        ],
    )
    return fig


def _shap_bars_html(top_features: list, is_api: bool = True) -> str:
    """
    SHAP 기여도만 표시 (피처명 + 바). raw_value/unit/change/is_imputed는 여기서 제외 —
    SHAP은 '이 피처가 예측에 얼마나 기여했는가'만 보여주는 영역.
    is_api=False면 전체 회색 톤.
    """
    if not top_features:
        return '<div class="desc-text" style="color:var(--c-muted);">기여 요인 정보가 없습니다.</div>'

    top3 = top_features[:3]
    vals = [float(x.get("shap_value", x.get("value", 0.0))) for x in top3]
    max_abs = max(abs(v) for v in vals) or 1.0

    gray = "#888888"
    gray_bar = "#b0b0b0"

    rows = ""
    for item, val in zip(top3, vals):
        feat = str(item.get("feature", ""))
        name = get_feature_display_name(feat)
        pct = int(abs(val) / max_abs * 100)

        if is_api:
            bar_color = SHAP_POS if val >= 0 else SHAP_NEG
            name_color = "var(--c-primary)"
        else:
            bar_color = gray_bar
            name_color = gray

        rows += (
            f'<div class="shap-item">'
            f'<div class="shap-name" style="color:{name_color};">'
            f'{html.escape(name)}</div>'
            f'<div class="shap-track">'
            f'<div class="shap-fill" style="width:{pct}%;background:{bar_color};"></div>'
            f'</div>'
            f'</div>'
        )
    return rows


def _clinical_indicators_table_html(indicators: list) -> str:
    """
    API clinical_indicators 렌더링 (지표 | 측정값+단위 | 정상범위).
    측정값 색상은 reference.risk_value로 결정:
      - True  → 빨강 (범위 밖 = 위험)
      - False → 초록 (범위 안 = 정상)
      - None  → 기본색 (판단 불가)
    레이아웃/클래스는 기존 _feature_table_html과 동일.
    """
    if not indicators:
        return ""

    color_risk = "#ef4444"
    color_safe = "#22c55e"
    color_default = "var(--c-primary)"
    name_color = "var(--c-primary)"
    range_color = "var(--c-secondary)"

    rows = ""
    for ind in indicators:
        raw = ind.get("value")
        if raw is None:
            val_str = "—"
        elif isinstance(raw, bool):
            val_str = "1" if raw else "0"
        elif isinstance(raw, float):
            if raw == int(raw):
                val_str = str(int(raw))
            else:
                val_str = f"{raw:.4f}".rstrip("0").rstrip(".") or "0"
        else:
            val_str = str(raw)

        unit = str(ind.get("unit") or "")
        # "binary" 단위는 usual_range에서 의미가 이미 설명됨 → 숫자만 표시
        if val_str != "—" and unit and unit.lower() != "binary":
            display_val = f"{val_str} {unit}".strip()
        else:
            display_val = val_str

        risk = ind.get("risk_value")
        if risk is True:
            val_color = color_safe
        elif risk is False:
            val_color = color_risk
        else:
            val_color = color_default

        usual_range = str(ind.get("usual_range") or "").strip()
        range_str = html.escape(usual_range) if usual_range else "—"

        rows += (
            f"<tr>"
            f'<td class="fn-cell" style="color:{name_color};">'
            f'{html.escape(str(ind.get("display_name", "-")))}</td>'
            f'<td class="fv-cell" style="color:{val_color};">'
            f'{html.escape(display_val)}</td>'
            f'<td class="fr-cell" style="color:{range_color};">{range_str}</td>'
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


def _feature_table_html(top_feature_values: list) -> str:
    """
    핵심 지표 측정값 — 모델 학습에서 의도적으로 제외한 임상 지표.
    현재 API에 이 값들이 내려오지 않으므로 mock 데이터를 전체 회색으로 표시.
    별도 API 필드 추가 시 이 함수에 연결 예정.
    """
    if not top_feature_values:
        return ""

    gray = "#888888"
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
            indicator = " ↑"
        elif is_anom and direction == "low":
            indicator = " ↓"
        else:
            indicator = ""

        range_str = html.escape(fv.get("normal_range_str") or "–")

        rows += (
            f"<tr>"
            f'<td class="fn-cell" style="color:{gray};">'
            f'{html.escape(fv.get("display_name", "-"))}</td>'
            f'<td class="fv-cell" style="color:{gray};">'
            f'{html.escape(display_val)}{indicator}</td>'
            f'<td class="fr-cell" style="color:{gray};">{range_str}</td>'
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
            f'<button class="cs-refresh-btn" id="cs-refresh-btn" title="새로고침" aria-label="새로고침">'
            f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round">'
            f'<polyline points="23 4 23 10 17 10"></polyline>'
            f'<polyline points="1 20 1 14 7 14"></polyline>'
            f'<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"></path>'
            f'<path d="M20.49 15a9 9 0 0 1-14.85 3.36L1 14"></path>'
            f'</svg>'
            f'</button>'
            f'<button class="cs-theme-btn" id="cs-theme-btn" '
            f'title="테마 전환" aria-label="테마 전환">'
            f'<span class="cs-theme-icon">🌙</span>'
            f'</button>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_patient_bar(data: dict) -> None:
    p = data.get("patient", {})
    meta = p.get("patient_meta", {}) or {}

    # 하드코딩(회색): 환자명 — API에 없어서 mock 값 유지
    name = html.escape(str(p.get("name", "-")))
    # SOFA 점수: patient_meta에 있으면 API 값(검은색), 없으면 mock(회색)
    sofa_api = meta.get("sofa_score")
    if sofa_api not in (None, "", "None"):
        sofa_str = html.escape(str(sofa_api))
        sofa_is_api = True
    else:
        sofa_str = html.escape(str(p.get("sofa_score", "-")))
        sofa_is_api = False

    # API(검은색): 나이, 성별, ICU 입실, 패혈증 onset — patient_meta에서 매핑
    age_raw = meta.get("age")
    age_str = f"{age_raw}세" if age_raw not in (None, "") else "-"

    gender_raw = str(meta.get("gender", ""))
    gender_map = {"1": "남성", "0": "여성"}
    gender_str = gender_map.get(gender_raw, "-")

    icu_admit_str = str(meta.get("intime") or "-")
    sepsis_onset_str = str(meta.get("sepsis_onset_time") or "-")

    api_style = "color:var(--c-primary);"
    hc_style = "color:var(--c-muted);"

    st.markdown(
        f"""
        <div class="patient-bar">
          <div class="pb-item">
            <span class="pb-label">환자</span>
            <span class="pb-value pb-name" style="{hc_style}">{name}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">나이</span>
            <span class="pb-value" style="{api_style}">{html.escape(age_str)}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">성별</span>
            <span class="pb-value" style="{api_style}">{html.escape(gender_str)}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">ICU 입실</span>
            <span class="pb-value" style="{api_style}">{html.escape(icu_admit_str)}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">SOFA 점수</span>
            <span class="pb-value" style="{api_style if sofa_is_api else hc_style}">{sofa_str}</span>
          </div>
          <div class="pb-divider"></div>
          <div class="pb-item">
            <span class="pb-label">패혈증 onset</span>
            <span class="pb-value" style="{api_style}">{html.escape(sepsis_onset_str)}</span>
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
        is_api = bool(model_result.get("has_api_data"))
        dq = model_result.get("data_quality", {}) or {}
        is_reliable = dq.get("is_reliable", True)

        if is_api:
            label, color, bg_color = _risk(prob)
        else:
            label = "Mock"
            color = "#666666"
            bg_color = "#f1f5f9"

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
                build_summary_donut(prob, is_api=is_api),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"summary_donut_{model_name}",
            )

            badges = (
                f'<span class="sc-risk-badge" '
                f'style="color:{color};background:{bg_color};">{label}</span>'
            )
            if is_api and not is_reliable:
                badges += (
                    ' <span class="sc-risk-badge" '
                    'style="color:#b45309;background:#fef3c7;'
                    'margin-left:4px;">⚠ 데이터 부족</span>'
                )
            st.markdown(
                f'<div class="sc-meta-row">{badges}</div>',
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
    is_api = bool(model_result.get("has_api_data"))
    dq = model_result.get("data_quality", {}) or {}
    is_reliable = dq.get("is_reliable", True)

    if is_api:
        label, color, bg_color = _risk(prob)
    else:
        label = "Mock"
        color = "#666666"
        bg_color = "#f1f5f9"

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

        if is_api and not is_reliable:
            st.markdown(
                '<div class="desc-box" style="border:1px solid #fcd34d;'
                'background:#fffbeb;margin-bottom:0.8rem;">'
                '<div class="desc-text" style="color:#b45309;">'
                '⚠ 측정 데이터가 부족하여 예측 신뢰도가 낮을 수 있습니다.'
                '</div></div>',
                unsafe_allow_html=True,
            )

        left, right = st.columns(2, gap="large")

        top_features = model_result.get("top_features", [])
        # 핵심 지표 테이블은 SHAP과 분리된 영역 — 항상 mock top_feature_values 사용
        top_feature_values = model_result.get("top_feature_values", [])

        with left:
            st.markdown(
                '<div class="card-section-label">주요 기여 요인 (SHAP)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _shap_bars_html(top_features, is_api=is_api),
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                '<div class="card-section-label">핵심 지표 측정값</div>',
                unsafe_allow_html=True,
            )
            clinical_indicators = model_result.get("clinical_indicators") or []
            if clinical_indicators:
                # API clinical_indicators 활성 — risk_value 기반 색상
                st.markdown(
                    _clinical_indicators_table_html(clinical_indicators),
                    unsafe_allow_html=True,
                )
            else:
                # API 데이터 없음 → 기존 mock 테이블(회색) 유지
                st.markdown(
                    _feature_table_html(top_feature_values),
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="card-section-label" style="margin-top:1.3rem;">임상 해석</div>',
                unsafe_allow_html=True,
            )
            desc = html.escape(str(model_result.get("description", "-")))
            desc_color = "var(--c-muted)" if not is_api else "var(--c-secondary)"
            st.markdown(
                f'<div class="desc-box"><div class="desc-text" '
                f'style="color:{desc_color};">{desc}</div></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────
# Sidebar overlay + refresh wiring (신규 추가)
# ─────────────────────────────────────────────────────────────
def _pick_patient(pid: str) -> None:
    """사이드바 숨김 버튼의 on_click 콜백. session_state를 갱신하고 Streamlit이 auto-rerun."""
    st.session_state["patient_id"] = pid
    print(f"[DEBUG] _pick_patient → patient_id={pid}")


def _render_patient_items_html(patient_ids: list[str], current_id: str) -> str:
    items = []
    for idx, pid in enumerate(patient_ids):
        is_selected = pid == current_id
        safe_pid = html.escape(pid)
        items.append(
            f'<div class="cs-patient-item{" is-selected" if is_selected else ""}" '
            f'data-id="{safe_pid.lower()}" '
            f'data-pid="{safe_pid}" '
            f'data-idx="{idx}">'
            f'<div class="cs-patient-info">'
            f'<div class="cs-patient-name">{safe_pid}</div>'
            f'</div>'
            f'</div>'
        )
    return "".join(items)


def render_sidebar_and_controls(
    data: dict,
    patient_ids: list[str],
    selected_pid: str | None,
) -> None:
    current_id = selected_pid or str(data.get("patient", {}).get("patient_id", ""))
    patient_items_html = _render_patient_items_html(patient_ids, current_id)

    # 햄버거 + 배경 + 사이드바 마크업
    st.markdown(
        f"""
        <button class="cs-hamburger" id="cs-hamburger" aria-label="환자 목록 열기">
          <span class="cs-hamburger-icon"><span></span><span></span><span></span></span>
        </button>
        <aside class="cs-sidebar" id="cs-sidebar" aria-hidden="true">
          <div class="cs-sidebar-header">
            <div class="cs-sidebar-title">환자 목록</div>
            <div class="cs-sidebar-subtitle">이름 또는 ID로 검색</div>
            <div class="cs-search-wrap">
              <span class="cs-search-icon">🔍</span>
              <input type="text" class="cs-search" id="cs-search"
                     placeholder="환자 검색..." autocomplete="off" />
            </div>
          </div>
          <div class="cs-patient-list" id="cs-patient-list">
            {patient_items_html}
            <div class="cs-empty" id="cs-empty" style="display:none;">
              검색 결과가 없습니다.
            </div>
          </div>
          <div class="cs-pagination" id="cs-pagination"></div>
        </aside>
        """,
        unsafe_allow_html=True,
    )

    # 새로고침 trigger (숨겨진 Streamlit 버튼 — JS에서 click()으로 호출)
    with st.container():
        st.markdown(
            '<div class="cs-refresh-trigger-anchor"></div>',
            unsafe_allow_html=True,
        )
        if st.button("refresh", key="_cs_refresh_trigger"):
            # 버튼 클릭 자체가 rerun을 유발하며 fetch_dashboard_data도 재호출됨
            pass

    # 환자 선택 trigger (per-patient 숨김 버튼)
    # JS 클릭 시 해당 pid의 버튼을 click() → on_click 콜백이 session_state를 갱신
    # → Streamlit이 자동 rerun → 새 patient_id로 API 호출
    with st.container():
        st.markdown(
            '<div class="cs-pick-trigger-anchor"></div>',
            unsafe_allow_html=True,
        )
        for pid in patient_ids:
            st.button(
                f"__pick__{pid}",
                key=f"_cs_pick_{pid}",
                on_click=_pick_patient,
                args=(pid,),
            )

    # 상호작용 JS 주입 (height=0 iframe)
    components.html(
        """
        <script>
        (function() {
            const pageWin = window.parent;
            const pageDoc = pageWin.document;
            const PER_PAGE = 10;

            function waitFor(selector, cb, tries) {
                tries = tries == null ? 50 : tries;
                const el = pageDoc.querySelector(selector);
                if (el) return cb(el);
                if (tries <= 0) return;
                setTimeout(function() { waitFor(selector, cb, tries - 1); }, 80);
            }

            // ── stMainBlockContainer height 고정 (Streamlit이 inline style을
            //    동적으로 덮어쓰는 것을 MutationObserver로 즉시 복원) ──
            function lockMainContainerHeight(target) {
                if (!target || target.dataset.csHeightLocked === '1') return;
                target.dataset.csHeightLocked = '1';

                function forceHeight() {
                    if (target.style.height !== '100vh') {
                        target.style.setProperty('height', '100vh', 'important');
                    }
                    if (target.style.minHeight !== '100vh') {
                        target.style.setProperty('min-height', '100vh', 'important');
                    }
                }
                forceHeight();

                const observer = new pageWin.MutationObserver(function() {
                    forceHeight();
                });
                observer.observe(target, {
                    attributes: true,
                    attributeFilter: ['style']
                });
            }
            waitFor('[data-testid="stMainBlockContainer"]', lockMainContainerHeight);

            // 숨겨진 trigger 컨테이너를 off-screen 처리 (refresh + 환자 선택 picker)
            function hideAnchorContainer(anchorSelector) {
                const anchor = pageDoc.querySelector(anchorSelector);
                if (!anchor) return;
                const container = anchor.closest('[data-testid="stVerticalBlock"]');
                if (container && !container.dataset.csHidden) {
                    container.dataset.csHidden = '1';
                    container.style.position = 'absolute';
                    container.style.left = '-9999px';
                    container.style.top = '-9999px';
                    container.style.width = '1px';
                    container.style.height = '1px';
                    container.style.overflow = 'hidden';
                }
            }
            function hideRefreshTriggerContainer() {
                hideAnchorContainer('.cs-refresh-trigger-anchor');
            }
            function hidePickTriggerContainer() {
                hideAnchorContainer('.cs-pick-trigger-anchor');
            }
            hideRefreshTriggerContainer();
            hidePickTriggerContainer();
            waitFor('.cs-refresh-trigger-anchor', hideRefreshTriggerContainer);
            waitFor('.cs-pick-trigger-anchor', hidePickTriggerContainer);

            waitFor('#cs-hamburger', function() {
                const body = pageDoc.body;

                // ── 리런 시 body에 누적된 stale hamburger/sidebar 제거 ──
                // Streamlit이 markdown 컨테이너를 다시 렌더하면 새 DOM이 생성되는데,
                // 이전 실행에서 body로 이식한 것들은 그대로 남아있어 ID가 중복됨.
                // getElementById가 먼저 만나는 stale 요소를 집어버리면 클릭 시
                // 이전 render의 data-pid가 발화되어 환자 전환이 안 되는 버그가 생김.
                pageDoc.querySelectorAll('body > #cs-hamburger').forEach(function(el) {
                    el.remove();
                });
                pageDoc.querySelectorAll('body > #cs-sidebar').forEach(function(el) {
                    el.remove();
                });

                const hamburger = pageDoc.getElementById('cs-hamburger');
                const sidebar = pageDoc.getElementById('cs-sidebar');
                if (!hamburger || !sidebar) return;

                // ── 이번 render에서 생성된 fresh 요소를 body 직속으로 이식 ──
                if (hamburger.parentElement !== body) {
                    const originalContainer = hamburger.closest(
                        '[data-testid="stElementContainer"]'
                    );
                    if (originalContainer) originalContainer.style.display = 'none';
                    body.appendChild(hamburger);
                }
                if (sidebar.parentElement !== body) {
                    body.appendChild(sidebar);
                }

                const searchInput = pageDoc.getElementById('cs-search');
                const listEl = pageDoc.getElementById('cs-patient-list');
                const emptyEl = pageDoc.getElementById('cs-empty');
                const pagerEl = pageDoc.getElementById('cs-pagination');

                // fresh 요소이므로 csBound는 항상 false → 매 리런마다 새로 바인딩
                hamburger.dataset.csBound = '1';

                const items = Array.prototype.slice.call(
                    listEl.querySelectorAll('.cs-patient-item')
                );

                let currentPage = 1;
                let filteredItems = items.slice();

                function openSidebar() {
                    body.classList.add('cs-open');
                    sidebar.setAttribute('aria-hidden', 'false');
                }
                function closeSidebar() {
                    body.classList.remove('cs-open');
                    sidebar.setAttribute('aria-hidden', 'true');
                }
                hamburger.addEventListener('click', function(e) {
                    e.stopPropagation();
                    if (body.classList.contains('cs-open')) closeSidebar();
                    else openSidebar();
                });
                // 사이드바 내부 클릭이 document로 버블링되어 즉시 닫히지 않도록 차단
                sidebar.addEventListener('click', function(e) { e.stopPropagation(); });

                // pageDoc 레벨 리스너는 매 리런마다 중복 등록되지 않도록 1회만 바인딩.
                // 핸들러는 항상 최신 sidebar/hamburger를 ID로 다시 조회.
                if (!pageWin.__csGlobalDocListeners) {
                    pageWin.__csGlobalDocListeners = true;
                    pageDoc.addEventListener('click', function(e) {
                        if (!body.classList.contains('cs-open')) return;
                        const sb = pageDoc.getElementById('cs-sidebar');
                        const hb = pageDoc.getElementById('cs-hamburger');
                        if (sb && sb.contains(e.target)) return;
                        if (hb && hb.contains(e.target)) return;
                        body.classList.remove('cs-open');
                        if (sb) sb.setAttribute('aria-hidden', 'true');
                    });
                    pageDoc.addEventListener('keydown', function(e) {
                        if (e.key !== 'Escape') return;
                        body.classList.remove('cs-open');
                        const sb = pageDoc.getElementById('cs-sidebar');
                        if (sb) sb.setAttribute('aria-hidden', 'true');
                    });
                }

                function renderPage() {
                    const total = filteredItems.length;
                    const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));
                    if (currentPage > totalPages) currentPage = totalPages;
                    if (currentPage < 1) currentPage = 1;

                    if (total === 0) {
                        emptyEl.style.display = 'block';
                    } else {
                        emptyEl.style.display = 'none';
                        const start = (currentPage - 1) * PER_PAGE;
                        const end = start + PER_PAGE;
                        const visible = filteredItems.slice(start, end);
                        items.forEach(function(el) {
                            el.classList.add('is-hidden');
                        });
                        visible.forEach(function(el) {
                            el.classList.remove('is-hidden');
                        });
                    }
                    renderPager(totalPages);
                }

                function renderPager(totalPages) {
                    pagerEl.innerHTML = '';
                    // 1페이지뿐이어도 페이지네이션은 항상 표시 (사용자 요청)

                    const prev = pageDoc.createElement('button');
                    prev.className = 'cs-page-btn';
                    prev.textContent = '<';
                    prev.disabled = currentPage === 1;
                    prev.addEventListener('click', function() {
                        if (currentPage > 1) { currentPage--; renderPage(); }
                    });
                    pagerEl.appendChild(prev);

                    for (let i = 1; i <= totalPages; i++) {
                        const b = pageDoc.createElement('button');
                        b.className = 'cs-page-btn' + (i === currentPage ? ' is-active' : '');
                        b.textContent = String(i);
                        b.addEventListener('click', (function(page) {
                            return function() { currentPage = page; renderPage(); };
                        })(i));
                        pagerEl.appendChild(b);
                    }

                    const next = pageDoc.createElement('button');
                    next.className = 'cs-page-btn';
                    next.textContent = '>';
                    next.disabled = currentPage === totalPages;
                    next.addEventListener('click', function() {
                        if (currentPage < totalPages) { currentPage++; renderPage(); }
                    });
                    pagerEl.appendChild(next);
                }

                function applyFilter() {
                    const q = (searchInput.value || '').trim().toLowerCase();
                    if (!q) {
                        filteredItems = items.slice();
                    } else {
                        filteredItems = items.filter(function(el) {
                            const n = el.dataset.name || '';
                            const id = el.dataset.id || '';
                            return n.indexOf(q) !== -1 || id.indexOf(q) !== -1;
                        });
                    }
                    // 모든 리스트 항목은 기본적으로 is-hidden로 만들고,
                    // 현재 페이지만 보여주는 방식으로 일원화
                    items.forEach(function(el) { el.classList.add('is-hidden'); });
                    currentPage = 1;
                    renderPage();
                }
                searchInput.addEventListener('input', applyFilter);

                // 환자 항목 클릭 — pid별 숨김 Streamlit 버튼을 click()
                // → Streamlit 네이티브 on_click 콜백이 session_state['patient_id']를 갱신
                // → Streamlit이 자동 rerun하여 새 환자 데이터로 화면 갱신
                function findPickButton(pid) {
                    const anchor = pageDoc.querySelector('.cs-pick-trigger-anchor');
                    if (!anchor) return null;
                    const container = anchor.closest('[data-testid="stVerticalBlock"]');
                    if (!container) return null;
                    const btns = container.querySelectorAll('button');
                    const targetLabel = '__pick__' + pid;
                    for (let i = 0; i < btns.length; i++) {
                        if ((btns[i].textContent || '').trim() === targetLabel) {
                            return btns[i];
                        }
                    }
                    return null;
                }

                items.forEach(function(el) {
                    el.addEventListener('click', function() {
                        items.forEach(function(x) { x.classList.remove('is-selected'); });
                        el.classList.add('is-selected');

                        const pid = el.getAttribute('data-pid') || '';
                        if (!pid) return;

                        const pickBtn = findPickButton(pid);
                        if (pickBtn) {
                            pickBtn.click();
                        } else {
                            console.warn('[ClinSight] pick button not found for pid=' + pid);
                        }
                    });
                });

                applyFilter();

                // ── 새로고침 버튼: 숨겨진 Streamlit 버튼을 클릭 ──
                const refreshBtn = pageDoc.getElementById('cs-refresh-btn');
                if (refreshBtn && refreshBtn.dataset.csBound !== '1') {
                    refreshBtn.dataset.csBound = '1';
                    refreshBtn.addEventListener('click', function() {
                        refreshBtn.classList.add('is-spinning');
                        const anchor = pageDoc.querySelector('.cs-refresh-trigger-anchor');
                        if (!anchor) return;
                        const container = anchor.closest('[data-testid="stVerticalBlock"]');
                        if (!container) return;
                        const hiddenBtn = container.querySelector('button');
                        if (hiddenBtn) hiddenBtn.click();
                    });
                }

                // ── 테마 토글 (라이트 ↔ 다크), localStorage 저장 ──
                const themeBtn = pageDoc.getElementById('cs-theme-btn');
                function applyTheme(mode) {
                    if (mode === 'dark') {
                        body.classList.add('cs-dark');
                    } else {
                        body.classList.remove('cs-dark');
                    }
                    if (themeBtn) {
                        const iconEl = themeBtn.querySelector('.cs-theme-icon');
                        if (iconEl) iconEl.textContent = (mode === 'dark') ? '☀️' : '🌙';
                        themeBtn.setAttribute(
                            'title',
                            mode === 'dark' ? '라이트모드로 전환' : '다크모드로 전환'
                        );
                    }
                }
                let savedTheme = 'light';
                try {
                    savedTheme = pageWin.localStorage.getItem('cs-theme') || 'light';
                } catch (e) { /* localStorage 접근 실패 시 light */ }
                applyTheme(savedTheme);

                if (themeBtn && themeBtn.dataset.csBound !== '1') {
                    themeBtn.dataset.csBound = '1';
                    themeBtn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        const next = body.classList.contains('cs-dark') ? 'light' : 'dark';
                        applyTheme(next);
                        try { pageWin.localStorage.setItem('cs-theme', next); }
                        catch (err) { /* 저장 실패 무시 */ }
                    });
                }
            });
        })();
        </script>
        """,
        height=0,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    inject_styles()

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = MODEL_ORDER[0]
    if "use_mock_data" not in st.session_state:
        st.session_state["use_mock_data"] = False
    if "prediction_cache" not in st.session_state:
        st.session_state["prediction_cache"] = {}

    # 환자 목록 조회
    try:
        patient_ids = fetch_patients()
    except Exception:
        patient_ids = []

    # patient_id source of truth = st.session_state["patient_id"]
    # 최초 진입 시 URL query_params 또는 목록 첫 환자로 초기화.
    if "patient_id" not in st.session_state:
        qp_pid = st.query_params.get("patient_id")
        st.session_state["patient_id"] = qp_pid or (
            patient_ids[0] if patient_ids else None
        )

    selected_pid = st.session_state["patient_id"]

    # URL도 동기화 (공유용, rerun 유발하지 않음)
    if selected_pid and st.query_params.get("patient_id") != selected_pid:
        st.query_params["patient_id"] = selected_pid

    print(f"[DEBUG] selected_pid={selected_pid}")

    # 예측 API 호출 (환자별 캐시). use_mock_data=True면 호출 생략.
    predictions = None
    if selected_pid and not st.session_state["use_mock_data"]:
        cache = st.session_state["prediction_cache"]
        if selected_pid not in cache:
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                f"""
                <div style="position:fixed;top:0;left:0;width:100vw;height:100vh;
                            display:flex;flex-direction:column;align-items:center;
                            justify-content:center;background:rgba(255,255,255,0.85);
                            z-index:99999;">
                  <div style="width:56px;height:56px;border:5px solid #e5e7eb;
                              border-top-color:#2563eb;border-radius:50%;
                              animation:cs-loading-spin 0.9s linear infinite;"></div>
                  <div style="margin-top:18px;color:#334155;font-size:14px;
                              font-weight:500;letter-spacing:0.01em;">
                    예측 결과를 불러오는 중... ({html.escape(str(selected_pid))})
                  </div>
                </div>
                <style>
                @keyframes cs-loading-spin {{ to {{ transform: rotate(360deg); }} }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            try:
                cache[selected_pid] = fetch_predictions(selected_pid)
            finally:
                loading_placeholder.empty()
        predictions = cache.get(selected_pid)
        print(f"[DEBUG] predictions fetched for {selected_pid}: "
              f"keys={list(predictions.keys()) if predictions else None}")
        print("[RESPONSE]", json.dumps(
            predictions, indent=2, ensure_ascii=False, default=str
        ))

    dashboard_data = fetch_dashboard_data(
        use_mock_override=bool(st.session_state["use_mock_data"]),
        use_mock_on_error=True,
        patient_id=selected_pid,
        predictions=predictions,
    )

    # 환자 기본 정보(API)를 dashboard_data.patient에 overlay
    if selected_pid:
        try:
            patient_api = fetch_patient_data(selected_pid)
            dashboard_data.setdefault("patient", {})
            dashboard_data["patient"]["patient_id"] = patient_api.get(
                "patient_id", selected_pid
            )
            dashboard_data["patient"]["patient_meta"] = patient_api.get(
                "patient_meta", {}
            )
        except Exception:
            dashboard_data.setdefault("patient", {}).setdefault("patient_meta", {})

    # 새로고침 시 last-updated가 실제로 갱신되도록 현재 시각을 stamp
    dashboard_data.setdefault("meta", {})["last_updated_display"] = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    render_page_header(dashboard_data)
    render_patient_bar(dashboard_data)

    st.markdown(
        '<div class="section-heading">모델 위험도 요약</div>',
        unsafe_allow_html=True,
    )
    render_summary_cards(dashboard_data)
    render_detail_panel(dashboard_data)

    # 신규 추가: 좌측 슬라이드 사이드바 (overlay) + 새로고침 버튼 wiring
    render_sidebar_and_controls(dashboard_data, patient_ids, selected_pid)

    # ── Debug: API 응답 전체 보기 ──
    # 표시하려면 SHOW_DEBUG = True 로 변경
    SHOW_DEBUG = False
    if SHOW_DEBUG:
        with st.expander("🔍 Debug: API Response", expanded=False):
            st.write(f"**selected_pid:** `{selected_pid}`")
            st.markdown("**predictions:**")
            st.json(predictions if predictions else {"_": "no predictions"})
            st.markdown("**patient_meta:**")
            st.json(
                dashboard_data.get("patient", {}).get("patient_meta", {})
                or {"_": "no patient_meta"}
            )


if __name__ == "__main__":
    main()
