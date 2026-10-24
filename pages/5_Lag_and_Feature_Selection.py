"""
Page 5: Lag and Feature Selection
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Granger Causality Lag Discovery, Feature Importance Ranking, and Selection Benchmarking.
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
    load_selection_results,
    is_dark_mode
)

# Initialize page
setup_page("Lag & Feature Selection", "🎯")

# Prototype demonstration banner
render_prototype_banner()

st.markdown("## 🎯 Temporal Lag Selection & Feature Importance")
st.markdown(
    "High-dimensional economic and energy indicators often suffer from severe **multicollinearity** and spurious correlations. "
    "We employ **Granger Causality tests** to determine optimal predictive lag orders, coupled with **L1 LASSO** and **Random Forest Gini Importance** to isolate primary drivers."
)

df_selection = load_selection_results()

if df_selection.empty:
    st.stop()

# ==============================================================================
# 1. BAR CHART OF SELECTED GRANGER LAGS PER DRIVER
# ==============================================================================
st.markdown('<div class="section-header">⏱️ Granger Causality: Optimal Temporal Lags per Driver</div>', unsafe_allow_html=True)

df_granger = df_selection[df_selection["category"] == "granger_lag"].copy()

# Sort by F-statistic
df_granger = df_granger.sort_values("stat_or_score", ascending=True)

fig_granger = go.Figure()

fig_granger.add_trace(go.Bar(
    y=df_granger["driver_or_feature"],
    x=df_granger["stat_or_score"],
    orientation="h",
    marker=dict(
        color=df_granger["stat_or_score"],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="F-Statistic", thickness=15)
    ),
    text=[f"Lag: {lag} (p={p:.4f})" for lag, p in zip(df_granger["optimal_lag"], df_granger["p_value"])],
    textposition="outside",
    hovertemplate=(
        "<b>Driver:</b> %{y}<br>"
        "<b>Granger F-Statistic:</b> %{x:.2f}<br>"
        "<b>Lag Order:</b> %{text}<extra></extra>"
    )
))

apply_plotly_theme(
    fig_granger,
    title="Optimal Granger Causality Lags (Ranked by Predictive F-Statistic)",
    x_title="Granger F-Statistic (Higher Indicates Stronger Predictive Power)",
    y_title=""
)
fig_granger.update_layout(height=420, margin=dict(l=150, r=90, t=55, b=45))

st.plotly_chart(fig_granger, use_container_width=True)

render_chart_caption(
    "This bar chart displays Granger causality F-statistics for macroeconomic and energy indicators. "
    "Higher F-statistics confirm that historical values of these drivers contain statistically significant information for predicting emissions. "
    "Labels indicate the optimal lag order selected by minimizing the Akaike Information Criterion (AIC)."
)

# ==============================================================================
# 2. TABLE OF SELECTED FEATURES PER COUNTRY
# ==============================================================================
st.markdown('<div class="section-header">🌍 Selected Driver Features & Relative Importance per Region</div>', unsafe_allow_html=True)

df_feat = df_selection[df_selection["category"] == "selected_features"].copy()

col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    selected_entity = st.selectbox(
        "Select Region / Country:",
        options=sorted(df_feat["entity"].unique()),
        index=0
    )

df_feat_filtered = df_feat[df_feat["entity"] == selected_entity].sort_values("stat_or_score", ascending=False)

# Display feature importance breakdown
col_chart, col_table = st.columns([1, 1])

with col_chart:
    fig_feat = px.bar(
        df_feat_filtered,
        x="stat_or_score",
        y="driver_or_feature",
        orientation="h",
        color="stat_or_score",
        color_continuous_scale="Blues",
        labels={"stat_or_score": "Importance Score (0.0 to 1.0)", "driver_or_feature": "Selected Feature"},
        title=f"Feature Weights for {selected_entity}"
    )
    dark = is_dark_mode()
    fig_feat.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="#1E293B" if dark else "#FFFFFF",
        plot_bgcolor="#0B1120" if dark else "#FAFAFA",
        title_font=dict(color="#F8FAFC" if dark else "#0F172A"),
        xaxis=dict(tickfont=dict(color="#CBD5E1" if dark else "#475569"), titlefont=dict(color="#F8FAFC" if dark else "#0F172A")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#CBD5E1" if dark else "#475569"), titlefont=dict(color="#F8FAFC" if dark else "#0F172A")),
        coloraxis_showscale=False,
        height=320,
        margin=dict(l=120, r=20, t=45, b=40)
    )
    st.plotly_chart(fig_feat, use_container_width=True)

with col_table:
    display_tbl = df_feat_filtered[["driver_or_feature", "stat_or_score", "notes"]].copy()
    display_tbl.columns = ["Selected Feature (Lagged)", "Importance Weight", "Domain Rationale"]
    st.dataframe(
        display_tbl.style.format({"Importance Weight": "{:.2f}"}),
        use_container_width=True,
        height=320
    )

# ==============================================================================
# 3. WITH VS WITHOUT FEATURE SELECTION COMPARISON
# ==============================================================================
st.markdown('<div class="section-header">⚖️ Benchmark Comparison: With vs Without Feature Selection</div>', unsafe_allow_html=True)

st.markdown(
    "Feeding all raw collinear variables into ML algorithms causes overfitting and noisy parameter weights. "
    "Here we compare model Root Mean Squared Error (RMSE) on the test set **Before Selection** (all raw indicators) vs **After Selection** (optimal Granger/LASSO features)."
)

df_comp = df_selection[df_selection["category"] == "selection_comparison"].copy()

fig_comp = go.Figure()

# Grouping by entity + model
entities = df_comp["entity"].unique()

for model_name in ["XGBoost", "Random Forest"]:
    sub = df_comp[df_comp["driver_or_feature"] == model_name]
    
    # Trace 1: Without Selection
    fig_comp.add_trace(go.Bar(
        name=f"{model_name} (Without Selection)",
        x=sub["entity"],
        y=sub["metric_all"],
        marker_color="#94A3B8" if model_name == "Random Forest" else "#64748B",
        text=sub["metric_all"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
        hovertemplate=f"<b>{model_name} (All Features)</b><br>RMSE: %{{y:.1f}} MtCO₂<extra></extra>"
    ))

    # Trace 2: With Selection
    fig_comp.add_trace(go.Bar(
        name=f"{model_name} (With Selection)",
        x=sub["entity"],
        y=sub["metric_selected"],
        marker_color="#10B981" if model_name == "XGBoost" else "#3B82F6",
        text=sub["metric_selected"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
        hovertemplate=f"<b>{model_name} (Selected Features)</b><br>RMSE: %{{y:.1f}} MtCO₂<extra></extra>"
    ))

apply_plotly_theme(
    fig_comp,
    title="Test RMSE Comparison: All Features vs Selected Features",
    x_title="Region / Country",
    y_title="Root Mean Squared Error (RMSE in MtCO₂/yr)"
)
fig_comp.update_layout(barmode="group", height=430)

st.plotly_chart(fig_comp, use_container_width=True)

# Highlight average error reduction
avg_improvement = df_comp["improvement_pct"].mean()
st.success(
    f"🚀 **Empirical Finding**: Pruning non-causal drivers via Granger lag selection and LASSO regularization achieves an "
    f"average **{avg_improvement:.1f}% reduction in test RMSE** across all regions, confirming that parsimonious models generalize significantly better."
)

render_chart_caption(
    "Comparison of test error (lower is better). The grey bars represent model error when trained on all raw features indiscriminately. "
    "The colored bars (blue and green) represent performance when trained only on Granger-verified causal features, visibly demonstrating the error reduction."
)
