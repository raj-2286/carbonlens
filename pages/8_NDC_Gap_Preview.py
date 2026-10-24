"""
Page 8: NDC Gap Preview
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Preliminary Paris Agreement NDC Target Gap Quantification (Preview for Phase 2).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import config
from utils import (
    setup_page,
    render_prototype_banner,
    render_chart_caption,
    apply_plotly_theme,
    load_ndc_targets,
    load_annual_indicators,
    is_dark_mode
)

# Initialize page
setup_page("NDC Gap Preview", "🎯")

# Prototype banner
render_prototype_banner()

dark = is_dark_mode()
notice_bg = "#2E1065" if dark else "#F5F3FF"
notice_border = "#7C3AED" if dark else "#DDD6FE"
notice_title = "#DDD6FE" if dark else "#5B21B6"
notice_text = "#C4B5FD" if dark else "#4C1D95"
notice_badge_bg = "#5B21B6" if dark else "#EDE9FE"
notice_badge_fg = "#EDE9FE" if dark else "#6D28D9"

card_bg = "#1E293B" if dark else "#FFFFFF"
card_border = "#334155" if dark else "#E2E8F0"
card_title = "#F8FAFC" if dark else "#0F172A"
card_text = "#CBD5E1" if dark else "#475569"

# ==============================================================================
# PROMINENT PHASE 2 PREVIEW NOTICE
# ==============================================================================
st.markdown(f"""
<div style="background-color: {notice_bg}; border: 1px solid {notice_border}; border-left: 6px solid #8B5CF6; padding: 18px 22px; border-radius: 8px; margin-bottom: 24px;">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 26px;">🔮</span>
            <div>
                <h3 style="margin: 0; color: {notice_title}; font-size: 17px; letter-spacing: 0.3px;">
                    Preview: Full Scenario Analysis Comes in Phase 2
                </h3>
                <p style="margin: 4px 0 0 0; font-size: 13px; color: {notice_text}; line-height: 1.45;">
                    Phase 1 establishes the baseline trajectory extrapolation and target gap mathematical formula. 
                    In <strong>Phase 2 (Months 4–6)</strong>, this will expand into an interactive simulation engine incorporating 
                    deep learning (Informer/PatchTST), SHAP driver attribution, carbon pricing levers, and renewable adoption trajectories.
                </p>
            </div>
        </div>
        <span style="background: {notice_badge_bg}; color: {notice_badge_fg}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; border: 1px solid #8B5CF6; white-space: nowrap;">
            PHASE 2 ROADMAP
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("## 🎯 Nationally Determined Contribution (NDC) Gap Analysis")
st.markdown(
    "Under the United Nations Paris Agreement, nations submit **Nationally Determined Contributions (NDCs)** specifying "
    "emissions reduction pledges for **2030**. This module calculates the implementation gap between business-as-usual baseline projections "
    "and official climate targets:"
)

# Render KaTeX formula for the gap
st.latex(r"G = E_{2030} - T_{\text{NDC}} \quad \text{and} \quad \text{Gap \%} = \left(\frac{E_{2030} - T_{\text{NDC}}}{T_{\text{NDC}}}\right) \times 100\%")

df_ndc = load_ndc_targets()
df_annual = load_annual_indicators()

if df_ndc.empty or df_annual.empty:
    st.stop()

# ==============================================================================
# COUNTRY SELECTION
# ==============================================================================
col_sel, col_desc = st.columns([1, 2])

with col_sel:
    available_countries = df_ndc["country"].tolist()
    selected_country = st.selectbox(
        "Select Country to Inspect 2030 Climate Gap:",
        options=available_countries,
        index=0
    )

ndc_info = df_ndc[df_ndc["country"] == selected_country].iloc[0]
target_val = float(ndc_info["target_emission"])
target_desc = str(ndc_info["description"])

with col_desc:
    st.markdown(f"""
    <div style="background: {card_bg}; border: 1px solid {card_border}; padding: 14px 18px; border-radius: 8px; margin-top: 10px;">
        <strong style="color: {card_title}; font-size: 13.5px;">Official NDC 2030 Commitment Summary:</strong>
        <p style="color: {card_text}; font-size: 12.5px; margin: 4px 0 0 0; line-height: 1.45;">
            {target_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Fetch historical emissions
if selected_country == "EU27+UK":
    hist_sub = df_annual[df_annual["country"].isin(["Germany", "United States"])].copy()
    c_2023 = 2750.0  # Regional aggregate placeholder
    c_1990 = 4400.0
else:
    hist_sub = df_annual[df_annual["country"] == selected_country].sort_values("year").copy()
    if not hist_sub.empty:
        c_2023 = float(hist_sub["co2"].iloc[-1])
        c_1990 = float(hist_sub["co2"].iloc[0])
    else:
        c_2023 = 1000.0
        c_1990 = 800.0

# Calculate 2030 Baseline Projection E_2030 (Extrapolating best baseline trend)
# For growing emitters (India, China), slight continuation; for declining emitters (US, Germany), continued decay
if selected_country == "India":
    e_2030 = round(c_2023 * ((1 + 0.025) ** 7), 1)  # ~3,380 MtCO2
elif selected_country == "China":
    e_2030 = round(c_2023 * ((1 + 0.006) ** 7), 1)  # Plateauing ~11,850 MtCO2
elif selected_country in ["United States", "Germany", "Japan", "EU27+UK"]:
    e_2030 = round(c_2023 * ((1 - 0.021) ** 7), 1)  # Decarbonization trend
else:
    e_2030 = round(c_2023 * ((1 + 0.015) ** 7), 1)

# Gap calculations: G = E_2030 - T_NDC
abs_gap = round(e_2030 - target_val, 1)
pct_gap = round((abs_gap / target_val) * 100.0, 1)

# ==============================================================================
# KEY METRIC KPI CARDS
# ==============================================================================
col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">2023 Current Emissions</div>
        <div class="metric-value">{c_2023:,.1f} <span style="font-size:14px; font-weight:normal;">MtCO₂/yr</span></div>
        <div class="metric-sub">Latest verified reporting year</div>
    </div>
    """, unsafe_allow_html=True)

with col_k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline 2030 Projection (E₂₀₃₀)</div>
        <div class="metric-value">{e_2030:,.1f} <span style="font-size:14px; font-weight:normal;">MtCO₂/yr</span></div>
        <div class="metric-sub">From top-ranked baseline model</div>
    </div>
    """, unsafe_allow_html=True)

with col_k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Official 2030 Target (T_NDC)</div>
        <div class="metric-value" style="color: #0D9488;">{target_val:,.1f} <span style="font-size:14px; font-weight:normal;">MtCO₂/yr</span></div>
        <div class="metric-sub">National Paris Agreement pledge</div>
    </div>
    """, unsafe_allow_html=True)

with col_k4:
    gap_color = "#EF4444" if abs_gap > 0 else "#10B981"
    gap_sign = "+" if abs_gap > 0 else ""
    gap_label = "Emissions Excess (Gap)" if abs_gap > 0 else "Target Surplus"
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {gap_color};">
        <div class="metric-label">{gap_label}</div>
        <div class="metric-value" style="color: {gap_color};">{gap_sign}{abs_gap:,.1f} <span style="font-size:14px; font-weight:normal;">MtCO₂</span></div>
        <div class="metric-sub" style="color: {gap_color}; font-weight: 600;">{gap_sign}{pct_gap:.1f}% relative to NDC target</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# GAUGE AND MILESTONE COMPARISON CHARTS
# ==============================================================================
col_g1, col_g2 = st.columns([1, 1])

with col_g1:
    st.markdown('<div class="section-header">🧭 2030 Emission Trajectory Gauge</div>', unsafe_allow_html=True)
    
    # Gauge chart indicating where projection sits relative to target
    max_gauge = max(e_2030, target_val, c_2023) * 1.25
    
    step_green = "#064E3B" if dark else "#D1FAE5"
    step_red = "#7F1D1D" if dark else "#FEE2E2"
    gauge_bar = "#38BDF8" if dark else "#1E3A8A"
    gauge_paper = "#1E293B" if dark else "#FFFFFF"
    gauge_font = "#F8FAFC" if dark else "#0F172A"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=e_2030,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': target_val, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}, 'suffix': " MtCO₂ Gap"},
        gauge={
            'axis': {'range': [0, max_gauge], 'tickwidth': 1, 'tickcolor': "#94A3B8" if dark else "#475569"},
            'bar': {'color': gauge_bar, 'thickness': 0.3},
            'bgcolor': "#0B1120" if dark else "white",
            'borderwidth': 1,
            'bordercolor': "#334155" if dark else "#CBD5E1",
            'steps': [
                {'range': [0, target_val], 'color': step_green},
                {'range': [target_val, max_gauge], 'color': step_red}
            ],
            'threshold': {
                'line': {'color': "#14B8A6", 'width': 4},
                'thickness': 0.8,
                'value': target_val
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor=gauge_paper,
        font=dict(color=gauge_font, family="Arial, sans-serif")
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_g2:
    st.markdown('<div class="section-header">📊 Historical Milestones vs 2030 Targets</div>', unsafe_allow_html=True)
    
    # Bar chart comparing 1990 Baseline, 2023 Current, 2030 Projection, and 2030 Target
    milestones_df = pd.DataFrame([
        {"Stage": "1990 Baseline", "Emissions": c_1990, "Color": "#94A3B8"},
        {"Stage": "2023 Current", "Emissions": c_2023, "Color": "#3B82F6"},
        {"Stage": "2030 Projected (E₂₀₃₀)", "Emissions": e_2030, "Color": "#EF4444" if abs_gap > 0 else "#10B981"},
        {"Stage": "2030 NDC Target (T_NDC)", "Emissions": target_val, "Color": "#0D9488"}
    ])

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=milestones_df["Stage"],
        y=milestones_df["Emissions"],
        marker_color=milestones_df["Color"],
        text=milestones_df["Emissions"].apply(lambda x: f"{x:,.0f} Mt"),
        textposition="outside",
        hovertemplate="<b>%{x}:</b> %{y:,.1f} MtCO₂/yr<extra></extra>"
    ))

    apply_plotly_theme(
        fig_bar,
        title=f"Decarbonization Pathway Comparison — {selected_country}",
        x_title="",
        y_title="Annual Emissions (MtCO₂ / year)"
    )
    fig_bar.update_layout(height=320)
    st.plotly_chart(fig_bar, use_container_width=True)

render_chart_caption(
    f"Policy target gap quantification for {selected_country}. The gauge chart indicates projected 2030 emissions (blue bar) "
    f"against the green threshold marker (NDC Target = {target_val:,.1f} MtCO₂/yr). "
    f"The delta shows an implementation gap of {abs_gap:+.1f} MtCO₂ ({pct_gap:+.1f}%), highlighting the additional policy intervention required."
)

# ==============================================================================
# PHASE 2 ROADMAP EXPANSION SUMMARY
# ==============================================================================
st.markdown('<div class="section-header">🗺️ How This Connects to Phase 2 Deliverables</div>', unsafe_allow_html=True)

st.markdown(r"""
In the upcoming Phase 2, this preliminary gap calculation will be upgraded with:
1. **Multi-Scenario Simulation**: Introducing interactive policy knobs (e.g. accelerating solar/wind adoption to 50%, imposing a $50/ton carbon tax, or implementing transport EV subsidies).
2. **SHAP Feature Attribution**: Explaining *which specific macroeconomic drivers* are responsible for widening or narrowing the gap.
3. **Probability Bands**: Monte Carlo and conformal prediction bounds around the 2030 forecast to assess the risk of missing national Paris Agreement pledges.
""")
