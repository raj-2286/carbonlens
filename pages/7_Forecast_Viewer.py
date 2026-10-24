"""
Page 7: Forecast Viewer
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Interactive Multi-Model Out-of-Sample Predictions and Residual Diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import config
from utils import (
    setup_page,
    render_prototype_banner,
    render_chart_caption,
    apply_plotly_theme,
    load_predictions,
    is_dark_mode
)

# Initialize page
setup_page("Forecast Viewer", "📈")

# Prototype banner
render_prototype_banner()

st.markdown("## 📈 Out-of-Sample Forecast Viewer & Residuals")
st.markdown(
    "Evaluate out-of-sample model predictions on the **unseen test period (2024–2025)**. "
    "Compare individual model trajectories against observed actual emissions and diagnose systematic errors via residual analysis."
)

df_preds = load_predictions()

if df_preds.empty:
    st.stop()

# Available model columns (excluding date, region, actual)
available_models = [c for c in df_preds.columns if c not in ["date", "region", "actual"]]

# ==============================================================================
# INTERACTIVE CONTROLS
# ==============================================================================
col_c1, col_c2 = st.columns([1, 2])

with col_c1:
    selected_region = st.selectbox(
        "Select Region:",
        options=config.DAILY_REGIONS,
        index=0
    )

with col_c2:
    selected_models = st.multiselect(
        "Select Baseline Models to Compare Against Actuals:",
        options=available_models,
        default=["Voting", "XGBoost", "SARIMA"]
    )

if not selected_models:
    st.warning("⚠️ Please select at least one model from the multiselect box above to view predictions.")
    st.stop()

# Filter dataframe for selected region
df_reg_preds = df_preds[df_preds["region"] == selected_region].sort_values("date").reset_index(drop=True)

# Date range selection
col_d1, col_d2 = st.columns(2)
min_date = df_reg_preds["date"].min().date()
max_date = df_reg_preds["date"].max().date()

with col_d1:
    start_date = st.date_input("Start Date (Test Period):", value=min_date, min_value=min_date, max_value=max_date)
with col_d2:
    end_date = st.date_input("End Date (Test Period):", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.error("Error: Start Date must be earlier than End Date.")
    st.stop()

# Filter by date range
mask = (df_reg_preds["date"].dt.date >= start_date) & (df_reg_preds["date"].dt.date <= end_date)
df_active = df_reg_preds[mask].copy()

if df_active.empty:
    st.warning("No data points found for the selected date window.")
    st.stop()

# ==============================================================================
# MAIN FORECAST PLOT: ACTUAL VS PREDICTED
# ==============================================================================
st.markdown(f'<div class="section-header">📊 Actual vs Predicted Emissions — {selected_region}</div>', unsafe_allow_html=True)

fig_forecast = go.Figure()

# Actual emissions trace
dark = is_dark_mode()
actual_line_color = "#FFFFFF" if dark else "#0F172A"
fig_forecast.add_trace(go.Scatter(
    x=df_active["date"],
    y=df_active["actual"],
    mode="lines",
    name="Ground Truth Actuals",
    line=dict(color=actual_line_color, width=2.8),
    hovertemplate="<b>Actual:</b> %{y:.2f} MtCO₂/day<extra></extra>"
))

# Model color palette
model_colors = {
    "ARIMA": "#94A3B8",
    "SARIMA": "#64748B",
    "SVR": "#8B5CF6",
    "Random_Forest": "#F59E0B",
    "Gradient_Boosting": "#F97316",
    "XGBoost": "#10B981",
    "Voting": "#3B82F6",
    "Bagging": "#EC4899"
}

for m in selected_models:
    color = model_colors.get(m, "#6B7280")
    dash_style = "dash" if m in ["ARIMA", "SARIMA"] else "solid"
    
    fig_forecast.add_trace(go.Scatter(
        x=df_active["date"],
        y=df_active[m],
        mode="lines",
        name=f"Pred: {m}",
        line=dict(color=color, width=1.8, dash=dash_style),
        opacity=0.88,
        hovertemplate=f"<b>{m}:</b> %{{y:.2f}} MtCO₂/day<extra></extra>"
    ))

apply_plotly_theme(
    fig_forecast,
    title=f"Out-of-Sample Forecast Horizon ({start_date} to {end_date})",
    x_title="Date",
    y_title="Daily CO₂ Emissions (MtCO₂ / day)"
)
fig_forecast.update_layout(height=480)

st.plotly_chart(fig_forecast, use_container_width=True)

render_chart_caption(
    f"Out-of-sample forecast for {selected_region}. The bold black curve represents the ground-truth observed emissions. "
    "Notice how XGBoost (green) and the Voting ensemble (blue) closely match weekend troughs and seasonal peaks, "
    "whereas classical statistical models (dashed grey) suffer from minor cyclical time lag and underpredict peak volatility."
)

# ==============================================================================
# DYNAMIC ERROR METRICS FOR SELECTED MODELS
# ==============================================================================
st.markdown('<div class="section-header">📋 Out-of-Sample Performance on Active Test Window</div>', unsafe_allow_html=True)

metrics_list = []
y_true = df_active["actual"].values

for m in selected_models:
    y_pred = df_active[m].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    r2 = r2_score(y_true, y_pred)
    
    metrics_list.append({
        "Model": m,
        "MAE (MtCO₂/day)": mae,
        "RMSE (MtCO₂/day)": rmse,
        "MAPE (%)": mape,
        "R² Score": r2
    })

df_metrics = pd.DataFrame(metrics_list).sort_values("MAPE (%)")

st.dataframe(
    df_metrics.style.format({
        "MAE (MtCO₂/day)": "{:.3f}",
        "RMSE (MtCO₂/day)": "{:.3f}",
        "MAPE (%)": "{:.2f}%",
        "R² Score": "{:.3f}"
    }),
    use_container_width=True
)

# ==============================================================================
# RESIDUAL ANALYSIS PLOT
# (Residuals over time + Residual Distribution)
# ==============================================================================
st.markdown('<div class="section-header">📉 Residual Diagnostic Analysis (Error = Actual - Predicted)</div>', unsafe_allow_html=True)

col_res1, col_res2 = st.columns([3, 2])

with col_res1:
    fig_res_time = go.Figure()
    
    # Zero line
    zero_line_color = "#CBD5E1" if dark else "#000000"
    fig_res_time.add_trace(go.Scatter(
        x=df_active["date"],
        y=[0] * len(df_active),
        mode="lines",
        name="Zero Error Baseline",
        line=dict(color=zero_line_color, width=1.5, dash="dot"),
        hoverinfo="skip"
    ))

    for m in selected_models:
        residuals = df_active["actual"] - df_active[m]
        color = model_colors.get(m, "#6B7280")
        fig_res_time.add_trace(go.Scatter(
            x=df_active["date"],
            y=residuals,
            mode="lines",
            name=f"Residual ({m})",
            line=dict(color=color, width=1.2),
            opacity=0.75,
            hovertemplate=f"<b>{m} Error:</b> %{{y:.2f}} MtCO₂<extra></extra>"
        ))

    apply_plotly_theme(
        fig_res_time,
        title="Residual Error Over Time (eₜ = Actual - Pred)",
        x_title="Date",
        y_title="Residual Error (MtCO₂ / day)"
    )
    fig_res_time.update_layout(height=380)
    st.plotly_chart(fig_res_time, use_container_width=True)

with col_res2:
    fig_hist = go.Figure()

    for m in selected_models:
        residuals = df_active["actual"] - df_active[m]
        color = model_colors.get(m, "#6B7280")
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            name=m,
            marker_color=color,
            opacity=0.55,
            nbinsx=35,
            hovertemplate=f"<b>{m} Bin:</b> %{{x:.2f}}<br><b>Count:</b> %{{y}}<extra></extra>"
        ))

    apply_plotly_theme(
        fig_hist,
        title="Residual Distribution (Zero-Mean Normality Check)",
        x_title="Residual Error (MtCO₂ / day)",
        y_title="Frequency Count"
    )
    fig_hist.update_layout(barmode="overlay", height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

render_chart_caption(
    "Residual diagnostic check. An ideal unbiased forecasting model produces residuals that oscillate randomly around zero "
    "(left time series) with a symmetric bell curve centered at zero (right histogram). "
    "A distribution skewed to the positive side indicates persistent underforecasting, whereas wide tails reveal vulnerability to sudden shocks."
)
