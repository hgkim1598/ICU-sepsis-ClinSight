from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from api_client import MODEL_ORDER, fetch_dashboard_data


st.set_page_config(page_title="ICU Clinical Dashboard", layout="wide")

TEXT_PRIMARY = "#1a1a2e"
TEXT_SECONDARY = "#5c6575"
APP_BACKGROUND = "#f6f8fb"
CARD_BACKGROUND = "#ffffff"
CARD_BORDER = "#dfe5ec"
CARD_SHADOW = "0 4px 12px rgba(20, 32, 56, 0.05)"
RISK_HIGH = "#d84c4c"
RISK_MODERATE = "#f0a53a"
RISK_LOW = "#2ea66a"
DONUT_REMAINDER = "#e8edf3"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {APP_BACKGROUND};
                color: {TEXT_PRIMARY};
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }}

            .block-container {{
                max-width: 1480px;
                padding-top: 1.4rem;
                padding-bottom: 1.6rem;
            }}

            .patient-panel-anchor,
            .model-card-anchor {{
                display: none;
            }}

            div[data-testid="stVerticalBlock"]:has(.patient-panel-anchor) {{
                background: {CARD_BACKGROUND};
                border: 1px solid {CARD_BORDER};
                border-radius: 12px;
                box-shadow: {CARD_SHADOW};
                padding: 1rem;
            }}

            div[data-testid="stVerticalBlock"]:has(.model-card-anchor) {{
                background: {CARD_BACKGROUND};
                border: 1px solid {CARD_BORDER};
                border-radius: 12px;
                box-shadow: {CARD_SHADOW};
                padding: 1rem 1rem 0.85rem;
                min-height: 390px;
            }}

            .panel-title {{
                font-size: 1.05rem;
                font-weight: 700;
                color: {TEXT_PRIMARY};
                margin-bottom: 0.9rem;
            }}

            .info-card {{
                background: #fbfcfe;
                border: 1px solid {CARD_BORDER};
                border-radius: 10px;
                padding: 0.85rem 0.9rem;
                margin-bottom: 0.7rem;
            }}

            .info-label {{
                font-size: 0.78rem;
                color: {TEXT_SECONDARY};
                font-weight: 600;
                margin-bottom: 0.15rem;
                letter-spacing: 0.01em;
            }}

            .info-value {{
                font-size: 1rem;
                color: {TEXT_PRIMARY};
                font-weight: 700;
                line-height: 1.35;
            }}

            .status-badge {{
                display: inline-block;
                padding: 0.28rem 0.65rem;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 700;
                border: 1px solid;
                margin-right: 0.45rem;
                margin-bottom: 0.45rem;
            }}

            .section-title {{
                font-size: 1.08rem;
                font-weight: 700;
                color: {TEXT_PRIMARY};
                margin: 0.15rem 0 0.9rem;
            }}

            .model-title {{
                font-size: 1rem;
                font-weight: 700;
                color: {TEXT_PRIMARY};
                margin-bottom: 0.3rem;
            }}

            .chart-caption {{
                font-size: 0.8rem;
                color: {TEXT_SECONDARY};
                font-weight: 600;
                margin-bottom: 0.4rem;
            }}

            .chip {{
                display: inline-block;
                padding: 0.28rem 0.62rem;
                margin: 0 0.42rem 0.42rem 0;
                border-radius: 999px;
                border: 1px solid #d7dfe8;
                background: #f3f6fa;
                color: {TEXT_PRIMARY};
                font-size: 0.76rem;
                font-weight: 600;
            }}

            .description-text {{
                font-size: 0.82rem;
                line-height: 1.45;
                color: {TEXT_SECONDARY};
                margin-top: 0.35rem;
            }}

            div[data-testid="stCheckbox"] {{
                margin-top: 0.2rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_risk_style(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "High", RISK_HIGH
    if probability >= 0.4:
        return "Moderate", RISK_MODERATE
    return "Low", RISK_LOW


def build_donut_chart(probability: float) -> go.Figure:
    risk_label, risk_color = get_risk_style(probability)
    value = max(0.0, min(probability, 1.0))

    fig = go.Figure(
        data=[
            go.Pie(
                values=[value, 1 - value],
                hole=0.72,
                sort=False,
                direction="clockwise",
                marker=dict(colors=[risk_color, DONUT_REMAINDER]),
                textinfo="none",
                hoverinfo="skip",
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        height=205,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{value * 100:.1f}%</b>",
                x=0.5,
                y=0.54,
                showarrow=False,
                font=dict(size=22, color=TEXT_PRIMARY),
            ),
            dict(
                text=risk_label,
                x=0.5,
                y=0.39,
                showarrow=False,
                font=dict(size=12, color=TEXT_SECONDARY),
            ),
        ],
    )
    return fig


def render_patient_info_panel(data: dict, use_mock_data: bool) -> None:
    patient = data.get("patient", {})
    meta = data.get("meta", {})
    source = meta.get("source_label", "Unknown")
    updated = meta.get("last_updated_display", "-")

    badge_color = "#64748b" if meta.get("source") == "mock" else "#2563eb"
    mock_badge_color = "#64748b" if use_mock_data else "#2ea66a"
    age_value = patient.get("age", "-")
    age_display = f"{age_value}세" if age_value != "-" else "-"

    with st.container():
        st.markdown('<div class="patient-panel-anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">환자 기본 정보</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <span class="status-badge" style="color:{badge_color}; border-color:{badge_color}33; background:{badge_color}12;">
                {html.escape(str(source))}
            </span>
            <span class="status-badge" style="color:{mock_badge_color}; border-color:{mock_badge_color}33; background:{mock_badge_color}12;">
                Mock 데이터 {'ON' if use_mock_data else 'AUTO'}
            </span>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Last updated: {updated}")

        fields = [
            ("이름", patient.get("name", "-")),
            ("환자 번호", patient.get("patient_id", "-")),
            ("나이", age_display),
            ("성별", patient.get("gender", "-")),
        ]

        for label, value in fields:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">{html.escape(str(label))}</div>
                    <div class="info-value">{html.escape(str(value))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.checkbox("Mock 데이터 사용", value=use_mock_data, key="use_mock_data")


def render_feature_chips(features: list[str]) -> None:
    if not features:
        st.caption("주요 피처 정보가 없습니다.")
        return

    chip_html = "".join(
        f'<span class="chip">{html.escape(str(feature))}</span>'
        for feature in features[:3]
    )
    st.markdown(chip_html, unsafe_allow_html=True)


def render_model_card(model_name: str, model_result: dict) -> None:
    with st.container():
        st.markdown('<div class="model-card-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="model-title">{html.escape(model_name)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-caption">예측 확률</div>', unsafe_allow_html=True)

        fig = build_donut_chart(float(model_result.get("probability", 0.0)))
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False},
            key=f"donut_{model_name}",
        )

        st.markdown('<div class="chart-caption">주요 피처 Top 3</div>', unsafe_allow_html=True)
        render_feature_chips(model_result.get("top_features_display", []))

        st.markdown(
            f'<div class="description-text">{html.escape(str(model_result.get("description", "-")))}</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()
    st.title("ICU Clinical Dashboard")

    if "use_mock_data" not in st.session_state:
        st.session_state["use_mock_data"] = False

    dashboard_data = fetch_dashboard_data(
        use_mock_override=bool(st.session_state["use_mock_data"]),
        use_mock_on_error=True,
    )

    left_col, right_col = st.columns([1, 3], gap="large")

    with left_col:
        render_patient_info_panel(
            dashboard_data,
            use_mock_data=bool(st.session_state["use_mock_data"]),
        )

    with right_col:
        st.markdown('<div class="section-title">모델 결과</div>', unsafe_allow_html=True)
        row1 = st.columns(2, gap="large")
        row2 = st.columns(2, gap="large")
        rows = [row1, row2]

        for index, model_name in enumerate(MODEL_ORDER):
            with rows[index // 2][index % 2]:
                render_model_card(
                    model_name,
                    dashboard_data.get("models", {}).get(model_name, {}),
                )


if __name__ == "__main__":
    main()
