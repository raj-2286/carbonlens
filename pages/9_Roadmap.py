"""
Page 9: Roadmap & Team Allocation
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Phase 1-3 Synopsis Roadmap, Current Phase Status, and Team Responsibility Matrix.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import config
from utils import setup_page, render_prototype_banner, render_chart_caption, apply_plotly_theme, is_dark_mode

# Initialize page
setup_page("Roadmap & Team", "🗺️")

# Prototype banner
render_prototype_banner()

dark = is_dark_mode()
card_bg = "#1E293B" if dark else "#FFFFFF"
card_border = "#334155" if dark else "#CBD5E1"
active_border = "#38BDF8" if dark else "#2563EB"
title_color = "#F8FAFC" if dark else "#0F172A"
text_color = "#E2E8F0" if dark else "#334155"
text_muted = "#94A3B8" if dark else "#64748B"
bar_bg = "#334155" if dark else "#F1F5F9"
guide_bg = "#172554" if dark else "#F8FAFC"
guide_border = "#1E40AF" if dark else "#CBD5E1"

st.markdown("## 🗺️ Project Roadmap & Team Allocation Matrix")
st.markdown(
    "Comprehensive execution plan for the **B.Tech Major Project** across its 9-month lifecycle, "
    "tracking milestone completion from preliminary data baselines to deep learning architectures and policy deployment."
)

# ==============================================================================
# PHASE 1-3 TIMELINE & MILESTONES
# ==============================================================================
st.markdown('<div class="section-header">📅 9-Month Project Lifecycle (Phases 1–3)</div>', unsafe_allow_html=True)

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    # Phase 1: Current Active Phase (Highlighted)
    st.markdown(f"""
    <div style="background: {card_bg}; border: 2px solid {active_border}; border-radius: 10px; padding: 20px; box-shadow: 0 4px 12px rgba(37,99,235,0.25); position: relative;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="background: {active_border}; color: white; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 12px; letter-spacing: 0.5px;">
                CURRENT PHASE (ACTIVE)
            </span>
            <span style="font-size: 13px; font-weight: 700; color: #38BDF8;">Months 1–3</span>
        </div>
        <h3 style="margin: 0 0 8px 0; color: {title_color}; font-size: 18px;">Phase 1: Data & Baselines</h3>
        <p style="font-size: 12.5px; color: {text_muted}; margin-bottom: 12px; line-height: 1.45;">
            Multi-scale schema ingestion, statistical preprocessing, unit-root stationarity tests, Granger lag selection, and 8 ML/econometric baselines.
        </p>
        <div style="border-top: 1px solid {card_border}; padding-top: 10px; font-size: 12px; color: {text_color};">
            <strong>Key Deliverables:</strong><br>
            ✅ Daily & Annual Data Schemas<br>
            ✅ ADF Stationarity & Differencing<br>
            ✅ Granger Causality Lag Selection<br>
            ✅ 8 Baseline Benchmarks<br>
            ✅ Streamlit Prototype App
        </div>
        <div style="margin-top: 14px; background: {bar_bg}; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: #2563EB; width: 100%; height: 100%;"></div>
        </div>
        <div style="font-size: 11px; color: #38BDF8; font-weight: 700; margin-top: 4px; text-align: right;">100% Phase 1 Complete</div>
    </div>
    """, unsafe_allow_html=True)

with col_p2:
    # Phase 2: Upcoming
    st.markdown(f"""
    <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="background: {bar_bg}; color: {text_muted}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; border: 1px solid {card_border};">
                UPCOMING NEXT
            </span>
            <span style="font-size: 13px; font-weight: 700; color: {text_muted};">Months 4–6</span>
        </div>
        <h3 style="margin: 0 0 8px 0; color: {title_color}; font-size: 18px;">Phase 2: Deep Learning & XAI</h3>
        <p style="font-size: 12.5px; color: {text_muted}; margin-bottom: 12px; line-height: 1.45;">
            Temporal deep learning architectures (LSTM, BiLSTM, Informer/PatchTST), SHAP value explainability, and multi-scenario policy simulation.
        </p>
        <div style="border-top: 1px solid {card_border}; padding-top: 10px; font-size: 12px; color: {text_color};">
            <strong>Planned Deliverables:</strong><br>
            ⏳ BiLSTM & Informer Models<br>
            ⏳ Attention-based Multi-Horizon Forecasts<br>
            ⏳ SHAP Feature Attribution Engine<br>
            ⏳ Renewable Policy Simulator<br>
            ⏳ Guide Mid-Term Evaluation
        </div>
        <div style="margin-top: 14px; background: {bar_bg}; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: #94A3B8; width: 0%; height: 100%;"></div>
        </div>
        <div style="font-size: 11px; color: {text_muted}; font-weight: 600; margin-top: 4px; text-align: right;">Scheduled Start: Month 4</div>
    </div>
    """, unsafe_allow_html=True)

with col_p3:
    # Phase 3: Final
    st.markdown(f"""
    <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="background: {bar_bg}; color: {text_muted}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; border: 1px solid {card_border};">
                FUTURE HORIZON
            </span>
            <span style="font-size: 13px; font-weight: 700; color: {text_muted};">Months 7–9</span>
        </div>
        <h3 style="margin: 0 0 8px 0; color: {title_color}; font-size: 18px;">Phase 3: Integration & Defense</h3>
        <p style="font-size: 12.5px; color: {text_muted}; margin-bottom: 12px; line-height: 1.45;">
            Production cloud deployment, automated NDC gap quantification engine, research paper drafting, and final B.Tech major thesis defense.
        </p>
        <div style="border-top: 1px solid {card_border}; padding-top: 10px; font-size: 12px; color: {text_color};">
            <strong>Planned Deliverables:</strong><br>
            ⏳ Multi-Regional Cloud API<br>
            ⏳ Automated Policy Gap Reporting<br>
            ⏳ Peer-reviewed Paper Submission<br>
            ⏳ Comprehensive Project Thesis<br>
            ⏳ Final University Viva Defense
        </div>
        <div style="margin-top: 14px; background: {bar_bg}; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: #94A3B8; width: 0%; height: 100%;"></div>
        </div>
        <div style="font-size: 11px; color: {text_muted}; font-weight: 600; margin-top: 4px; text-align: right;">Scheduled Start: Month 7</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# VISUAL GANTT / TIMELINE CHART
# ==============================================================================
st.markdown('<div class="section-header">📊 Synopsis Milestone Timeline Chart</div>', unsafe_allow_html=True)

gantt_tasks = [
    {"Task": "Multi-scale Data Ingestion & Quality Audit", "Start": 1, "Duration": 1.2, "Phase": "Phase 1: Data & Baselines", "Color": "#3B82F6"},
    {"Task": "ADF Stationarity Testing & Differencing", "Start": 1.5, "Duration": 0.8, "Phase": "Phase 1: Data & Baselines", "Color": "#3B82F6"},
    {"Task": "Granger Lag Selection & Feature Ranking", "Start": 2, "Duration": 0.8, "Phase": "Phase 1: Data & Baselines", "Color": "#3B82F6"},
    {"Task": "8 Baseline Models Implementation & Bench", "Start": 2.2, "Duration": 0.8, "Phase": "Phase 1: Data & Baselines", "Color": "#3B82F6"},
    {"Task": "Phase 1 Guide Demo & Streamlit App", "Start": 2.8, "Duration": 0.3, "Phase": "Phase 1: Data & Baselines", "Color": "#10B981"},

    {"Task": "LSTM / BiLSTM Recurrent Neural Networks", "Start": 4, "Duration": 1.5, "Phase": "Phase 2: Deep Learning & XAI", "Color": "#F59E0B"},
    {"Task": "PatchTST & Informer Transformer Models", "Start": 4.8, "Duration": 1.4, "Phase": "Phase 2: Deep Learning & XAI", "Color": "#F59E0B"},
    {"Task": "SHAP Explainability & Driver Attribution", "Start": 5.2, "Duration": 0.9, "Phase": "Phase 2: Deep Learning & XAI", "Color": "#F59E0B"},

    {"Task": "Automated NDC 2030 Policy Gap Engine", "Start": 7, "Duration": 1.2, "Phase": "Phase 3: Integration & Defense", "Color": "#8B5CF6"},
    {"Task": "Cloud API Deployment & Real-Data Ingestion", "Start": 7.8, "Duration": 0.8, "Phase": "Phase 3: Integration & Defense", "Color": "#8B5CF6"},
    {"Task": "Research Paper Drafting & Final Thesis Viva", "Start": 8.2, "Duration": 0.9, "Phase": "Phase 3: Integration & Defense", "Color": "#8B5CF6"}
]

fig_gantt = go.Figure()

for task in reversed(gantt_tasks):
    fig_gantt.add_trace(go.Bar(
        y=[task["Task"]],
        x=[task["Duration"]],
        base=[task["Start"]],
        name=task["Phase"],
        orientation="h",
        marker=dict(color=task["Color"]),
        text=f"{task['Duration']} mo",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11),
        hovertemplate=(
            f"<b>{task['Task']}</b><br>"
            f"<b>Phase:</b> {task['Phase']}<br>"
            f"<b>Timeline:</b> Month {task['Start']:.1f} to {task['Start'] + task['Duration']:.1f}<extra></extra>"
        ),
        showlegend=False
    ))

apply_plotly_theme(
    fig_gantt,
    title="B.Tech Project 9-Month Roadmap Gantt Chart",
    x_title="Project Timeline (Months 1 to 9)",
    y_title=""
)
fig_gantt.update_layout(
    height=480,
    margin=dict(l=260, r=40, t=55, b=45),
    xaxis=dict(range=[0.5, 9.5], dtick=1, gridcolor="#E2E8F0")
)

st.plotly_chart(fig_gantt, use_container_width=True)

render_chart_caption(
    "Project timeline across all 9 months. Blue bars represent completed Phase 1 tasks (culminating in the green milestone for today's guide demo). "
    "Amber represents upcoming Phase 2 deep learning and explainability work, and purple represents final cloud integration and thesis defense."
)

# ==============================================================================
# TEAM RESPONSIBILITY MATRIX: "WHO DOES WHAT"
# ==============================================================================
st.markdown('<div class="section-header">👥 Team Responsibilities Matrix ("Who Does What")</div>', unsafe_allow_html=True)

col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #3B82F6; background-color: {card_bg}; border-color: {card_border};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 28px;">🤖</span>
            <div>
                <h4 style="margin: 0; color: {title_color}; font-size: 18px;">Rajat Singh</h4>
                <div style="font-size: 12px; color: #38BDF8; font-weight: 700;">Models & Evaluation Lead</div>
            </div>
        </div>
        <ul style="font-size: 12.5px; color: {text_color}; margin: 0; padding-left: 18px; line-height: 1.6;">
            <li><strong>Time-Series Baselines:</strong> Implementation of ARIMA and SARIMA models with seasonal grid search.</li>
            <li><strong>Machine Learning Ensembles:</strong> Training SVR, Random Forest, Gradient Boosting, XGBoost, and Voting Regressors.</li>
            <li><strong>Cross-Validation:</strong> Design of expanding walk-forward chronological validation folds.</li>
            <li><strong>Evaluation Bench:</strong> Calculation of MAE, RMSE, MAPE, and R² scores across daily and annual scales.</li>
            <li><strong>Phase 2 Role:</strong> Deep learning model training (BiLSTM & Transformer).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_r2:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #10B981; background-color: {card_bg}; border-color: {card_border};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 28px;">💻</span>
            <div>
                <h4 style="margin: 0; color: {title_color}; font-size: 18px;">Pranav Kumar Singh</h4>
                <div style="font-size: 12px; color: #34D399; font-weight: 700;">Data Pipeline, SHAP & App Lead</div>
            </div>
        </div>
        <ul style="font-size: 12.5px; color: {text_color}; margin: 0; padding-left: 18px; line-height: 1.6;">
            <li><strong>Data Pipeline:</strong> Dual-scale data schema architecture and reproducible generator script.</li>
            <li><strong>Streamlit Dashboard:</strong> Full 9-page multipage UI development, Plotly interactive theming, and error handling.</li>
            <li><strong>NDC Gap Calculator:</strong> Formulation of mathematical equations and gauge visualizations for 2030 targets.</li>
            <li><strong>Config Architecture:</strong> One-line toggle mechanism (<code>DATA_MODE="sample"|"real"</code>).</li>
            <li><strong>Phase 2 Role:</strong> SHAP feature attribution pipeline and scenario simulation sliders.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_r3:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #F59E0B; background-color: {card_bg}; border-color: {card_border};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 28px;">📋</span>
            <div>
                <h4 style="margin: 0; color: {title_color}; font-size: 18px;">Pradhyumna Sharma</h4>
                <div style="font-size: 12px; color: #FBBF24; font-weight: 700;">Data Collection & Testing Lead</div>
            </div>
        </div>
        <ul style="font-size: 12.5px; color: {text_color}; margin: 0; padding-left: 18px; line-height: 1.6;">
            <li><strong>Data Collection:</strong> Curation of historical macroeconomic and energy indicators for 11 largest emitters.</li>
            <li><strong>Stationarity Diagnostics:</strong> Augmented Dickey-Fuller (ADF) unit-root testing before and after differencing.</li>
            <li><strong>Causality Testing:</strong> Granger causality F-statistic computation for multi-lag driver identification.</li>
            <li><strong>Quality Assurance:</strong> Missing value auditing, unit consistency checks, and testing.</li>
            <li><strong>Documentation:</strong> Project report writing, technical documentation, and thesis drafting.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Responsibilities summary table
st.markdown("#### 📋 Detailed Responsibility Matrix")

resp_df = pd.DataFrame([
    {"Module / Subsystem": "Daily & Annual Data Schemas", "Primary Lead": "Pranav Kumar Singh", "Support": "Pradhyumna Sharma", "Status": "✅ Completed"},
    {"Module / Subsystem": "ADF Stationarity & Differencing", "Primary Lead": "Pradhyumna Sharma", "Support": "Rajat Singh", "Status": "✅ Completed"},
    {"Module / Subsystem": "Granger Causality Lag Discovery", "Primary Lead": "Pradhyumna Sharma", "Support": "Rajat Singh", "Status": "✅ Completed"},
    {"Module / Subsystem": "ARIMA / SARIMA Statistical Baselines", "Primary Lead": "Rajat Singh", "Support": "Pradhyumna Sharma", "Status": "✅ Completed"},
    {"Module / Subsystem": "Tree & Ensemble ML Models (RF, XGB, Voting)", "Primary Lead": "Rajat Singh", "Support": "Pranav Kumar Singh", "Status": "✅ Completed"},
    {"Module / Subsystem": "Evaluation Bench & Metric Analysis", "Primary Lead": "Rajat Singh", "Support": "Pradhyumna Sharma", "Status": "✅ Completed"},
    {"Module / Subsystem": "Streamlit Multipage Dashboard", "Primary Lead": "Pranav Kumar Singh", "Support": "Rajat Singh", "Status": "✅ Completed"},
    {"Module / Subsystem": "NDC Target Gap Quantification", "Primary Lead": "Pranav Kumar Singh", "Support": "Pradhyumna Sharma", "Status": "✅ Completed"},
    {"Module / Subsystem": "Phase 1 Synopsis & Guide Documentation", "Primary Lead": "Pradhyumna Sharma", "Support": "Team", "Status": "✅ Completed"}
])

st.dataframe(resp_df, use_container_width=True)

# ==============================================================================
# PROJECT GUIDE ENDORSEMENT & VIVA NOTES
# ==============================================================================
st.markdown('<div class="section-header">👩‍🏫 Project Guide Review & Sign-Off</div>', unsafe_allow_html=True)

badge_style = "background: #064E3B; color: #6EE7B7; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 12px; border: 1px solid #059669;" if dark else "background: #D1FAE5; color: #065F46; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 12px; border: 1px solid #A7F3D0;"

st.markdown(f"""
<div style="background: {guide_bg}; border: 1px solid {guide_border}; border-radius: 8px; padding: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <strong style="color: {title_color}; font-size: 15px;">Supervisor: {config.SUPERVISOR}</strong><br>
            <span style="color: {text_muted}; font-size: 13px;">Department of Information Technology &bull; {config.INSTITUTION}</span>
        </div>
        <span style="{badge_style}">
            PHASE 1 SUBMISSION READY
        </span>
    </div>
    <div style="margin-top: 14px; font-size: 13px; color: {text_color}; line-height: 1.55;">
        <strong>Phase 1 Evaluation Criteria Satisfied:</strong><br>
        1. Clean problem formulation coupling high-frequency daily emissions with multi-decade macroeconomic drivers.<br>
        2. Sound statistical preprocessing verifying stationarity via Augmented Dickey-Fuller unit-root tests.<br>
        3. Principled lag selection using Granger Causality to eliminate arbitrary window heuristics.<br>
        4. Benchmark comparison establishing baseline performance across 8 classical econometric and modern ensemble algorithms.<br>
        5. Zero-leakage chronological validation protocol ready for Phase 2 deep learning scaling.
    </div>
</div>
""", unsafe_allow_html=True)
