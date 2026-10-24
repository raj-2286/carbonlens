"""
Page 1: Home
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Phase 1 Overview, Team Information, Progress Checklist, and Mini Pipeline Diagram.
"""

import streamlit as st
import config
from utils import setup_page, render_prototype_banner, is_dark_mode

# Configure page settings
setup_page("Home", "🏠")

# Mandatory prototype demonstration banner
render_prototype_banner()

dark = is_dark_mode()
title_color = "#F8FAFC" if dark else "#0F172A"
card_bg = "#1E293B" if dark else "#FFFFFF"
border_color = "#334155" if dark else "#E2E8F0"
sub_bg = "#0B1120" if dark else "#F8FAFC"
text_muted = "#94A3B8" if dark else "#64748B"
text_body = "#E2E8F0" if dark else "#475569"

# ==============================================================================
# HERO BANNER / HEADER
# ==============================================================================
st.markdown(f"""
<div style="background: linear-gradient(135deg, #1E3A8A 0%, #0D9488 100%); padding: 32px 36px; border-radius: 12px; color: white; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
    <div style="font-size: 13px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #A7F3D0; margin-bottom: 6px;">
        B.Tech Major Project &bull; Department of Information Technology / CSE
    </div>
    <h1 style="color: white; margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.5px;">
        {config.PROJECT_FULL_TITLE}
    </h1>
    <p style="font-size: 16px; margin-top: 10px; margin-bottom: 18px; color: #E0E7FF; max-width: 900px; line-height: 1.5;">
        A multi-scale artificial intelligence framework coupling high-frequency daily emissions with macroeconomic drivers 
        for robust baseline benchmarking and Paris Agreement NDC target gap quantification.
    </p>
    <div style="display: flex; flex-wrap: wrap; gap: 24px; font-size: 13.5px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 14px;">
        <div>🏛️ <strong>Institution:</strong> {config.INSTITUTION}</div>
        <div>👩‍🏫 <strong>Project Guide:</strong> {config.SUPERVISOR}</div>
        <div>📅 <strong>Phase:</strong> Phase 1 (Data, Stationarity & Baselines)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TEAM MEMBERS DISPLAY CARDS
# ==============================================================================
st.markdown('<div class="section-header">👥 Project Team & Primary Roles</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #3B82F6; background-color: {card_bg}; border-color: {border_color};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 26px;">🤖</span>
            <div>
                <h4 style="margin: 0; color: {title_color}; font-size: 17px;">Rajat Singh</h4>
                <span style="font-size: 12px; color: #38BDF8; font-weight: 700;">Models & Evaluation</span>
            </div>
        </div>
        <p style="font-size: 13px; color: {text_body}; margin: 0; line-height: 1.45;">
            Time-series baselines (ARIMA/SARIMA), ML algorithms (SVR, RF, XGBoost), train/test split protocols, and statistical evaluation bench.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #10B981; background-color: {card_bg}; border-color: {border_color};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 26px;">💻</span>
            <div>
                <h4 style="margin: 0; color: {title_color}; font-size: 17px;">Pranav Kumar Singh</h4>
                <span style="font-size: 12px; color: #34D399; font-weight: 700;">Data Pipeline & Dashboard</span>
            </div>
        </div>
        <p style="font-size: 13px; color: {text_body}; margin: 0; line-height: 1.45;">
            Multi-source ingestion pipeline, Streamlit dashboard architecture, NDC target gap calculator, and Phase 2 SHAP explainability setup.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #F59E0B; background-color: {card_bg}; border-color: {border_color};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 26px;">📋</span>
            <div>
                <h4 style="margin: 0; color: {title_color}; font-size: 17px;">Pradhyumna Sharma</h4>
                <span style="font-size: 12px; color: #FBBF24; font-weight: 700;">Data Collection & Testing</span>
            </div>
        </div>
        <p style="font-size: 13px; color: {text_body}; margin: 0; line-height: 1.45;">
            Historical indicator curation, stationarity (ADF) & Granger causality testing, data validation, and project report documentation.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SHORT PROJECT SUMMARY
# ==============================================================================
st.markdown('<div class="section-header">📌 Project Summary & Objectives</div>', unsafe_allow_html=True)

st.markdown("""
**CarbonLens** is an end-to-end AI-powered carbon emission forecasting system developed to address the critical gaps in 
conventional greenhouse gas accounting. Traditional emission reports suffer from severe publishing delays (1–2 years), 
obscuring real-time progress toward climate targets.

Our solution tackles this challenge across three core pillars:
1. **Multi-Scale Temporal Modeling**: Synergizes high-frequency daily emissions (capturing immediate industrial and power variations) with multi-decade annual macroeconomic indicators (GDP, population, energy intensity, and renewable penetration).
2. **Rigorous Baseline Benchmarking**: Establishes a transparent baseline comparing traditional econometric methods (*ARIMA, SARIMA*) against modern machine learning ensembles (*SVR, Random Forest, Gradient Boosting, XGBoost, and Voting*).
3. **Actionable Climate Policy Alignment**: Projects national 2030 trajectory pathways against Paris Agreement Nationally Determined Contributions (*NDCs*), quantifying policy implementation gaps in absolute ($MtCO_2$) and relative terms.
""")

# ==============================================================================
# MINI PIPELINE DIAGRAM
# (Data -> Preprocessing -> Lag/Feature selection -> Baseline models -> Evaluation)
# ==============================================================================
st.markdown('<div class="section-header">🔄 Phase 1 Architectural Pipeline</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="background-color: {card_bg}; border: 1px solid {border_color}; border-radius: 10px; padding: 22px; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: stretch; gap: 12px;">
        <!-- Step 1 -->
        <div style="flex: 1; min-width: 170px; background: {sub_bg}; border: 1px solid {border_color}; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="background: #3B82F6; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto; font-weight: bold; font-size: 13px;">1</div>
            <div style="font-weight: 700; color: {title_color}; font-size: 14px; margin-bottom: 4px;">Data Ingestion</div>
            <div style="font-size: 12px; color: {text_muted};">Daily Sectoral CO2 (4 regions) & Annual Drivers (11 emitters)</div>
        </div>
        <div style="display: flex; align-items: center; font-size: 20px; color: {text_muted}; font-weight: bold;">&rarr;</div>
        <!-- Step 2 -->
        <div style="flex: 1; min-width: 170px; background: {sub_bg}; border: 1px solid {border_color}; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="background: #0D9488; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto; font-weight: bold; font-size: 13px;">2</div>
            <div style="font-weight: 700; color: {title_color}; font-size: 14px; margin-bottom: 4px;">Preprocessing</div>
            <div style="font-size: 12px; color: {text_muted};">ADF unit-root tests, differencing (d=1) for stationarity</div>
        </div>
        <div style="display: flex; align-items: center; font-size: 20px; color: {text_muted}; font-weight: bold;">&rarr;</div>
        <!-- Step 3 -->
        <div style="flex: 1; min-width: 170px; background: {sub_bg}; border: 1px solid {border_color}; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="background: #8B5CF6; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto; font-weight: bold; font-size: 13px;">3</div>
            <div style="font-weight: 700; color: {title_color}; font-size: 14px; margin-bottom: 4px;">Lag & Features</div>
            <div style="font-size: 12px; color: {text_muted};">Granger causality lag selection & RF/LASSO feature ranking</div>
        </div>
        <div style="display: flex; align-items: center; font-size: 20px; color: {text_muted}; font-weight: bold;">&rarr;</div>
        <!-- Step 4 -->
        <div style="flex: 1; min-width: 170px; background: {sub_bg}; border: 1px solid {border_color}; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="background: #F59E0B; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto; font-weight: bold; font-size: 13px;">4</div>
            <div style="font-weight: 700; color: {title_color}; font-size: 14px; margin-bottom: 4px;">Baseline Models</div>
            <div style="font-size: 12px; color: {text_muted};">8 baselines (ARIMA, SARIMA, SVR, RF, GB, XGBoost, Ensembles)</div>
        </div>
        <div style="display: flex; align-items: center; font-size: 20px; color: {text_muted}; font-weight: bold;">&rarr;</div>
        <!-- Step 5 -->
        <div style="flex: 1; min-width: 170px; background: {sub_bg}; border: 1px solid {border_color}; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="background: #EF4444; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto; font-weight: bold; font-size: 13px;">5</div>
            <div style="font-weight: 700; color: {title_color}; font-size: 14px; margin-bottom: 4px;">Evaluation</div>
            <div style="font-size: 12px; color: {text_muted};">MAE, RMSE, MAPE, R2 & 2030 NDC Gap preliminary projection</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# PHASE 1 PROGRESS CHECKLIST WITH TICK MARKS
# ==============================================================================
st.markdown('<div class="section-header">✅ Phase 1 Deliverables & Progress Checklist</div>', unsafe_allow_html=True)

col_check1, col_check2 = st.columns(2)

checklist_items_1 = [
    ("Multi-scale dataset schema formulation (Daily + Annual)", True, "10,228 daily rows + 374 annual rows structured"),
    ("Augmented Dickey-Fuller (ADF) stationarity testing", True, "Confirmed stationarity after d=1 first-order differencing"),
    ("Granger Causality lag selection", True, "Identified optimal lags for GDP, electricity & industry drivers"),
    ("Leak-free chronological Train/Val/Test splitting", True, "Strict time ordering without lookahead or shuffle bias")
]

checklist_items_2 = [
    ("Implementation of 8 baseline forecasting models", True, "ARIMA, SARIMA, SVR, RF, Gradient Boosting, XGBoost, Voting, Bagging"),
    ("Model evaluation bench with 4 standardized metrics", True, "Evaluated on MAE, RMSE, MAPE, and R2 across scales"),
    ("Interactive multi-model forecast & residual viewer", True, "Dynamic model comparison against actual test period emissions"),
    ("Preliminary NDC 2030 policy gap quantification", True, "Quantifies absolute & percentage gap to national climate pledges")
]

with col_check1:
    for title, completed, details in checklist_items_1:
        icon = "✅" if completed else "⏳"
        st.markdown(f"""
        <div style="background: {sub_bg}; border: 1px solid {border_color}; border-left: 4px solid #10B981; padding: 12px 14px; border-radius: 4px 8px 8px 4px; margin-bottom: 10px;">
            <div style="font-weight: 700; font-size: 14px; color: {title_color};">{icon} {title}</div>
            <div style="font-size: 12px; color: {text_muted}; margin-top: 2px;">{details}</div>
        </div>
        """, unsafe_allow_html=True)

with col_check2:
    for title, completed, details in checklist_items_2:
        icon = "✅" if completed else "⏳"
        st.markdown(f"""
        <div style="background: {sub_bg}; border: 1px solid {border_color}; border-left: 4px solid #10B981; padding: 12px 14px; border-radius: 4px 8px 8px 4px; margin-bottom: 10px;">
            <div style="font-weight: 700; font-size: 14px; color: {title_color};">{icon} {title}</div>
            <div style="font-size: 12px; color: {text_muted}; margin-top: 2px;">{details}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer guide note
st.markdown("---")
st.info(
    "💡 **Guide Review Instructions**: Use the left sidebar navigation (Pages 2 through 9) to inspect each stage of the Phase 1 pipeline, "
    "from exploratory analysis and stationarity diagnostics to baseline model comparisons and forecast residuals."
)
