"""
Page 4: Train / Validation / Test Split
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Chronological Data Splitting, Leak-Free Protocol, and Walk-Forward Validation Setup.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import config
from utils import setup_page, render_prototype_banner, render_chart_caption, apply_plotly_theme, is_dark_mode

# Initialize page
setup_page("Train/Val/Test Split", "✂️")

# Prototype demonstration banner
render_prototype_banner()

dark = is_dark_mode()
box_bg = "#062A20" if dark else "#ECFDF5"
box_border = "#059669" if dark else "#A7F3D0"
box_title_color = "#6EE7B7" if dark else "#065F46"
box_text_color = "#A7F3D0" if dark else "#064E3B"
card_bg = "#1E293B" if dark else "#FFFFFF"
card_border = "#334155" if dark else "#D1FAE5"
card_title_color = "#34D399" if dark else "#047857"
card_text_color = "#E2E8F0" if dark else "#374151"

st.markdown("## ✂️ Chronological Partitioning & Leak-Free Splitting")
st.markdown(
    "In time-series modeling, standard random shuffling introduces catastrophic **data leakage** "
    "(training on future data to predict the past). CarbonLens enforces strict **chronological ordering** across all training, validation, and testing partitions."
)

# ==============================================================================
# NO-LEAKAGE CONFIRMATION BOX
# ==============================================================================
st.markdown(f"""
<div style="background-color: {box_bg}; border: 1px solid {box_border}; border-left: 6px solid #10B981; padding: 20px 24px; border-radius: 8px; margin-bottom: 24px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <span style="font-size: 26px;">🛡️</span>
        <h3 style="margin: 0; color: {box_title_color}; font-size: 18px;">
            Guaranteed No-Leakage Verification Protocol
        </h3>
    </div>
    <p style="margin: 0 0 12px 0; font-size: 13.5px; color: {box_text_color}; line-height: 1.55;">
        To ensure total scientific integrity for our forecasting framework, the pipeline enforces four strict barriers:
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px;">
        <div style="background: {card_bg}; border: 1px solid {card_border}; padding: 12px 14px; border-radius: 6px;">
            <strong style="color: {card_title_color}; font-size: 13px;">1. Zero Temporal Shuffling:</strong>
            <p style="font-size: 12px; color: {card_text_color}; margin: 4px 0 0 0;">Observations are partitioned strictly by calendar dates. Future observations never enter the training set.</p>
        </div>
        <div style="background: {card_bg}; border: 1px solid {card_border}; padding: 12px 14px; border-radius: 6px;">
            <strong style="color: {card_title_color}; font-size: 13px;">2. Scaler Isolation:</strong>
            <p style="font-size: 12px; color: {card_text_color}; margin: 4px 0 0 0;">Normalizers (StandardScaler / MinMaxScaler) are fitted <em>strictly on the Train partition</em> and transform Val/Test without recalculating statistics.</p>
        </div>
        <div style="background: {card_bg}; border: 1px solid {card_border}; padding: 12px 14px; border-radius: 6px;">
            <strong style="color: {card_title_color}; font-size: 13px;">3. Causal Lag Formulation:</strong>
            <p style="font-size: 12px; color: {card_text_color}; margin: 4px 0 0 0;">All exogenous feature vectors x<sub>t</sub> strictly reference historical lags (t-1, t-2, t-7) with zero contemporaneous future lookahead.</p>
        </div>
        <div style="background: {card_bg}; border: 1px solid {card_border}; padding: 12px 14px; border-radius: 6px;">
            <strong style="color: {card_title_color}; font-size: 13px;">4. Walk-Forward Evaluation:</strong>
            <p style="font-size: 12px; color: {card_text_color}; margin: 4px 0 0 0;">Out-of-sample testing uses expanding rolling windows simulating real-world production deployment.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TIMELINE VISUALIZATION (CHRONOLOGICAL PERIODS)
# ==============================================================================
st.markdown('<div class="section-header">📅 Chronological Timeline Chart</div>', unsafe_allow_html=True)

# Define partition specifications
split_specs = [
    # Daily scale
    {"Scale": "Daily Emissions (MtCO₂/day)", "Split": "Train Set", "Start": "2019-01-01", "End": "2022-12-31", "Days": 1461, "Pct": "57.1%", "Color": config.COLORS["train"]},
    {"Scale": "Daily Emissions (MtCO₂/day)", "Split": "Validation Set", "Start": "2023-01-01", "End": "2023-12-31", "Days": 365, "Pct": "14.3%", "Color": config.COLORS["val"]},
    {"Scale": "Daily Emissions (MtCO₂/day)", "Split": "Test Set (Final Eval)", "Start": "2024-01-01", "End": "2025-12-31", "Days": 731, "Pct": "28.6%", "Color": config.COLORS["test"]},

    # Annual scale
    {"Scale": "Annual Indicators (MtCO₂/yr)", "Split": "Train Set", "Start": "1990-01-01", "End": "2014-12-31", "Days": 25 * 365, "Pct": "73.5%", "Color": config.COLORS["train"]},
    {"Scale": "Annual Indicators (MtCO₂/yr)", "Split": "Validation Set", "Start": "2015-01-01", "End": "2019-12-31", "Days": 5 * 365, "Pct": "14.7%", "Color": config.COLORS["val"]},
    {"Scale": "Annual Indicators (MtCO₂/yr)", "Split": "Test Set (Final Eval)", "Start": "2020-01-01", "End": "2023-12-31", "Days": 4 * 365, "Pct": "11.8%", "Color": config.COLORS["test"]}
]

fig_timeline = go.Figure()

for item in split_specs:
    fig_timeline.add_trace(go.Bar(
        y=[item["Scale"]],
        x=[pd.to_datetime(item["End"]) - pd.to_datetime(item["Start"])],
        base=[pd.to_datetime(item["Start"])],
        name=item["Split"],
        orientation="h",
        marker=dict(color=item["Color"]),
        text=f"{item['Split']}<br>{item['Start']} to {item['End']} ({item['Pct']})",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11, family="Arial, sans-serif"),
        hovertemplate=(
            f"<b>{item['Scale']}</b><br>"
            f"<b>Partition:</b> {item['Split']}<br>"
            f"<b>Start:</b> {item['Start']}<br>"
            f"<b>End:</b> {item['End']}<br>"
            f"<b>Proportion:</b> {item['Pct']}<extra></extra>"
        ),
        showlegend=True if item["Scale"].startswith("Daily") else False
    ))

fig_timeline.update_layout(
    title={
        "text": "Chronological Partitioning Across Daily & Annual Temporal Horizons",
        "y": 0.95, "x": 0.02, "xanchor": "left",
        "font": {"color": "#F8FAFC" if dark else "#0F172A", "size": 16}
    },
    barmode="stack",
    template="plotly_dark" if dark else "plotly_white",
    paper_bgcolor="#1E293B" if dark else "#FFFFFF",
    plot_bgcolor="#0B1120" if dark else "#FAFAFA",
    height=320,
    margin=dict(l=60, r=40, t=55, b=45),
    xaxis=dict(
        type="date",
        title="Calendar Timeline (Years)",
        showgrid=True,
        gridcolor="#334155" if dark else "#E2E8F0",
        tickfont=dict(color="#CBD5E1" if dark else "#475569"),
        titlefont=dict(color="#F8FAFC" if dark else "#0F172A")
    ),
    yaxis=dict(
        title="",
        autorange="reversed",
        tickfont=dict(color="#CBD5E1" if dark else "#475569")
    ),
    legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", font=dict(color="#CBD5E1" if dark else "#334155"))
)

st.plotly_chart(fig_timeline, use_container_width=True)
render_chart_caption(
    "Visual breakdown of chronological splits. Blue denotes the Training period (model fitting), Amber denotes the Validation period "
    "(hyperparameter tuning, model checkpointing), and Green denotes the unseen Test period reserved exclusively for evaluating final baseline results."
)

# ==============================================================================
# DETAILED DATE RANGES AND SAMPLE SIZES TABLE
# ==============================================================================
st.markdown('<div class="section-header">📋 Date Ranges & Sample Partition Summary</div>', unsafe_allow_html=True)

col_tbl1, col_tbl2 = st.columns(2)

with col_tbl1:
    st.markdown("#### 📅 Daily Scale Splits (Per Region: 2,557 days total)")
    daily_summary_df = pd.DataFrame([
        {"Partition": "Train Set", "Date Range": "2019-01-01 to 2022-12-31", "Duration": "4 Years (1,461 days)", "Share": "57.1%", "Role": "Model fitting & parameter learning"},
        {"Partition": "Validation Set", "Date Range": "2023-01-01 to 2023-12-31", "Duration": "1 Year (365 days)", "Share": "14.3%", "Role": "Hyperparameter optimization & early stopping"},
        {"Partition": "Test Set", "Date Range": "2024-01-01 to 2025-12-31", "Duration": "2 Years (731 days)", "Share": "28.6%", "Role": "Final out-of-sample benchmark evaluation"}
    ])
    st.dataframe(daily_summary_df, use_container_width=True)

with col_tbl2:
    st.markdown("#### 📅 Annual Scale Splits (Per Country: 34 years total)")
    annual_summary_df = pd.DataFrame([
        {"Partition": "Train Set", "Date Range": "1990 to 2014", "Duration": "25 Years", "Share": "73.5%", "Role": "Historical baseline pattern learning"},
        {"Partition": "Validation Set", "Date Range": "2015 to 2019", "Duration": "5 Years", "Share": "14.7%", "Role": "Tuning regularization & kernel penalties"},
        {"Partition": "Test Set", "Date Range": "2020 to 2023", "Duration": "4 Years", "Share": "11.8%", "Role": "Recent post-COVID climate trajectory testing"}
    ])
    st.dataframe(annual_summary_df, use_container_width=True)

# ==============================================================================
# TIME-SERIES CROSS VALIDATION METHODOLOGY
# ==============================================================================
st.markdown('<div class="section-header">🔄 Expanding-Window Walk-Forward Validation Setup</div>', unsafe_allow_html=True)

st.markdown("""
Unlike static train/test splits, CarbonLens models utilize **Expanding Window (Walk-Forward) Cross-Validation**:
- **Fold 1:** Train [2019–2021] &rarr; Validate [2022]
- **Fold 2:** Train [2019–2022] &rarr; Validate [2023]
- **Final Evaluation:** Train [2019–2023] &rarr; Evaluate Test [2024–2025]

This accurately mimics operational deployment where energy forecasters retrain models incrementally as new monthly or daily observations arrive from power grids and satellite monitoring.
""")
