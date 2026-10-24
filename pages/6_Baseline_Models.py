"""
Page 6: Baseline Models
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Comprehensive Baseline Benchmarking across Statistical and Machine Learning Models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import config
from utils import (
    setup_page,
    render_prototype_banner,
    render_chart_caption,
    apply_plotly_theme,
    load_baseline_results,
    is_dark_mode
)

# Initialize page
setup_page("Baseline Models", "🤖")

# Prototype banner
render_prototype_banner()

st.markdown("## 🤖 Baseline Model Performance Benchmark")
st.markdown(
    "Phase 1 establishes a rigorous benchmark across **8 baseline models** spanning classical econometrics "
    "(*ARIMA, SARIMA*), kernel methods (*SVR*), tree ensembles (*Random Forest, Gradient Boosting, XGBoost*), "
    "and meta-ensembles (*Voting, Bagging*)."
)

df_base = load_baseline_results()

if df_base.empty:
    st.stop()

# ==============================================================================
# INTERACTIVE FILTERS
# ==============================================================================
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    scale_filter = st.selectbox(
        "Select Temporal Scale:",
        options=["daily", "annual"],
        format_func=lambda x: "Daily Emissions (MtCO₂/day)" if x == "daily" else "Annual Indicators (MtCO₂/yr)",
        index=0
    )

df_filtered_scale = df_base[df_base["scale"] == scale_filter].copy()

with col_f2:
    available_regions = sorted(df_filtered_scale["region_or_country"].unique())
    region_filter = st.selectbox(
        "Select Region / Country:",
        options=available_regions,
        index=0
    )

df_region_base = df_filtered_scale[df_filtered_scale["region_or_country"] == region_filter].copy()

with col_f3:
    metric_choice = st.selectbox(
        "Select Evaluation Metric:",
        options=["MAPE", "RMSE", "MAE", "R2"],
        format_func=lambda x: f"{x} ({'Lower is better' if x in ['MAPE', 'RMSE', 'MAE'] else 'Higher is better'})",
        index=0
    )

# ==============================================================================
# HIGHLIGHT BEST MODEL PER REGION
# ==============================================================================
# Determine best model based on metric
if metric_choice == "R2":
    best_row = df_region_base.loc[df_region_base["R2"].idxmax()]
else:
    best_row = df_region_base.loc[df_region_base[metric_choice].idxmin()]

col_best1, col_best2 = st.columns([1, 1])

with col_best1:
    unit_str = "%" if metric_choice == "MAPE" else ("" if metric_choice == "R2" else ("MtCO₂/day" if scale_filter == "daily" else "MtCO₂/yr"))
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); color: white; padding: 18px 22px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #93C5FD;">
                🏆 Best Baseline Model for {region_filter}
            </span>
            <span style="background: #10B981; color: white; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 10px;">
                Top Performer
            </span>
        </div>
        <h3 style="margin: 6px 0 2px 0; color: white; font-size: 24px;">{best_row['model']}</h3>
        <div style="font-size: 14px; color: #DBEAFE;">
            Achieved <strong>{metric_choice} = {best_row[metric_choice]:.2f} {unit_str}</strong> (R² = {best_row['R2']:.3f}, MAPE = {best_row['MAPE']:.1f}%)
        </div>
        <div style="font-size: 12px; color: #BFDBFE; margin-top: 6px;">
            Training Computation Time: {best_row['training_time_sec']:.2f} seconds
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_best2:
    # Summary of runner up or general ensemble advantage
    voting_row = df_region_base[df_region_base["model"] == "Voting"].iloc[0]
    xgb_row = df_region_base[df_region_base["model"] == "XGBoost"].iloc[0]
    arima_row = df_region_base[df_region_base["model"] == "ARIMA"].iloc[0]
    
    dark = is_dark_mode()
    sub_card_bg = "#1E293B" if dark else "#FFFFFF"
    sub_card_border = "#334155" if dark else "#E2E8F0"
    sub_title_color = "#94A3B8" if dark else "#64748B"
    sub_text_color = "#E2E8F0" if dark else "#334155"

    st.markdown(f"""
    <div style="background: {sub_card_bg}; border: 1px solid {sub_card_border}; padding: 18px 22px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: {sub_title_color};">
            📊 Statistical vs Machine Learning Divergence
        </div>
        <p style="margin: 6px 0 0 0; font-size: 13px; color: {sub_text_color}; line-height: 1.5;">
            • <strong>Ensemble Boost:</strong> <code>{best_row['model']}</code> outperforms classical <code>ARIMA</code> by 
            <strong>{abs(arima_row['MAPE'] - best_row['MAPE']):.1f}% MAPE reduction</strong> in {region_filter}.<br>
            • <strong>Speed vs Precision:</strong> <code>XGBoost</code> trains in <strong>{xgb_row['training_time_sec']:.1f}s</strong> while delivering competitive <strong>R² = {xgb_row['R2']:.3f}</strong>.<br>
            • <strong>Phase 1 Conclusion:</strong> Non-linear ML models reliably capture holiday and multi-lag dynamics where linear autoregression degrades.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# BAR CHART: MODEL COMPARISON BY CHOSEN METRIC
# ==============================================================================
st.markdown(f'<div class="section-header">📊 Model Benchmark Comparison ({metric_choice})</div>', unsafe_allow_html=True)

# Sort models: if R2 sort descending, else sort ascending
sort_asc = False if metric_choice == "R2" else True
df_sorted = df_region_base.sort_values(metric_choice, ascending=sort_asc).copy()

# Highlight best bar
bar_colors = ["#10B981" if m == best_row["model"] else "#3B82F6" if "Boost" in m or "Forest" in m or "Voting" in m else "#94A3B8" for m in df_sorted["model"]]

fig_metric = go.Figure()

fig_metric.add_trace(go.Bar(
    x=df_sorted["model"],
    y=df_sorted[metric_choice],
    marker_color=bar_colors,
    text=df_sorted[metric_choice].apply(lambda x: f"{x:.2f}" if metric_choice in ["R2", "MAPE"] else f"{x:.1f}"),
    textposition="outside",
    hovertemplate=f"<b>Model:</b> %{{x}}<br><b>{metric_choice}:</b> %{{y:.3f}}<extra></extra>"
))

apply_plotly_theme(
    fig_metric,
    title=f"Baseline Model Comparison by {metric_choice} — {region_filter} ({scale_filter.capitalize()} Scale)",
    x_title="Model Architecture",
    y_title=f"{metric_choice} ({unit_str})"
)
fig_metric.update_layout(height=400)

st.plotly_chart(fig_metric, use_container_width=True)

render_chart_caption(
    f"Comparison of 8 baseline models on {metric_choice} for {region_filter}. "
    f"The top-performing model is highlighted in green ({best_row['model']}). "
    "Tree-based and ensemble architectures (blue) consistently outperform classical autoregressive models (grey) due to non-linear driver interactions."
)

# ==============================================================================
# TRAINING TIME COMPARISON CHART
# ==============================================================================
st.markdown('<div class="section-header">⚡ Computational Complexity: Training Time (Seconds)</div>', unsafe_allow_html=True)

df_time_sorted = df_region_base.sort_values("training_time_sec", ascending=True).copy()

fig_time = go.Figure()

fig_time.add_trace(go.Bar(
    y=df_time_sorted["model"],
    x=df_time_sorted["training_time_sec"],
    orientation="h",
    marker_color="#0D9488",
    text=df_time_sorted["training_time_sec"].apply(lambda x: f"{x:.2f}s"),
    textposition="outside",
    hovertemplate="<b>Model:</b> %{y}<br><b>Fit Time:</b> %{x:.2f} seconds<extra></extra>"
))

apply_plotly_theme(
    fig_time,
    title=f"Training Execution Time — {region_filter} (Single Run on Standard CPU)",
    x_title="Wall-Clock Training Time (Seconds — Lower is Faster)",
    y_title=""
)
fig_time.update_layout(height=360, margin=dict(l=140, r=60, t=55, b=45))

st.plotly_chart(fig_time, use_container_width=True)

render_chart_caption(
    "Wall-clock training duration across algorithms. Notice that SARIMA exhibits significant computational overhead "
    "due to iterative seasonal maximum likelihood estimation (searching multi-period polynomials), whereas tree-based algorithms (XGBoost, Random Forest) "
    "and SVR execute in a fraction of the time with superior accuracy."
)

# ==============================================================================
# FILTERABLE COMPREHENSIVE DATA TABLE
# ==============================================================================
st.markdown('<div class="section-header">📋 Complete Benchmark Results Table</div>', unsafe_allow_html=True)

# Format table for display
display_cols = ["model", "scale", "region_or_country", "MAE", "RMSE", "MAPE", "R2", "training_time_sec"]
styled_table = df_region_base[display_cols].copy()
styled_table.columns = ["Model", "Scale", "Region / Country", "MAE", "RMSE", "MAPE (%)", "R² Score", "Fit Time (s)"]

st.dataframe(
    styled_table.style.format({
        "MAE": "{:.3f}",
        "RMSE": "{:.3f}",
        "MAPE (%)": "{:.2f}%",
        "R² Score": "{:.3f}",
        "Fit Time (s)": "{:.2f}s"
    }),
    use_container_width=True
)

# Download Table
csv_base = styled_table.to_csv(index=False).encode('utf-8')
st.download_button(
    label=f"📥 Download Baseline Benchmark Table for {region_filter} (CSV)",
    data=csv_base,
    file_name=f"carbonlens_baseline_benchmark_{region_filter.lower().replace('+', '_')}.csv",
    mime="text/csv"
)
