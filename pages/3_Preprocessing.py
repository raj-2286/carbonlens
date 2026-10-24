"""
Page 3: Preprocessing
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Augmented Dickey-Fuller (ADF) Stationarity Analysis and Differencing Diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from utils import (
    setup_page,
    render_prototype_banner,
    render_chart_caption,
    apply_plotly_theme,
    load_adf_results,
    load_daily_emissions,
    load_annual_indicators,
    is_dark_mode
)

# Initialize page
setup_page("Preprocessing", "🔬")

# Mandatory prototype banner
render_prototype_banner()

dark = is_dark_mode()
box_bg = "#172554" if dark else "#EFF6FF"
box_border = "#1E40AF" if dark else "#BFDBFE"
box_title_color = "#93C5FD" if dark else "#1E3A8A"
box_text_color = "#DBEAFE" if dark else "#1E293B"
box_li_color = "#BFDBFE" if dark else "#334155"

st.markdown("## 🔬 Statistical Preprocessing & Stationarity Analysis")
st.markdown(
    "Time-series models such as **ARIMA** and **SARIMA** require the underlying stochastic process to be "
    "**weakly stationary**. This page presents unit-root diagnostic tests and visualizes the first-order differencing transformation."
)

# ==============================================================================
# PLAIN-LANGUAGE EXPLANATION OF STATIONARITY
# ==============================================================================
st.markdown(f"""
<div style="background-color: {box_bg}; border: 1px solid {box_border}; border-left: 5px solid #2563EB; padding: 18px 22px; border-radius: 8px; margin-bottom: 24px;">
    <h4 style="margin: 0 0 8px 0; color: {box_title_color}; font-size: 16px;">
        💡 Plain-Language Guide: What is Stationarity and Why is it Essential?
    </h4>
    <p style="margin: 0 0 10px 0; font-size: 13.5px; color: {box_text_color}; line-height: 1.55;">
        A time series is <strong>stationary</strong> if its statistical properties—specifically its <strong>mean</strong>, 
        <strong>variance</strong>, and <strong>autocorrelation</strong>—do not change over time. 
        Raw carbon emissions naturally exhibit non-stationary behavior because economic expansion introduces long-term upward trends, 
        and seasonal weather introduces cyclical shifts.
    </p>
    <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: {box_li_color}; line-height: 1.5;">
        <li><strong>The Risk of Non-Stationarity:</strong> Fitting statistical models to trending data produces <em>spurious regressions</em>—models predict high correlations purely because both series grow over time, leading to misleading forecasts and blown-up confidence intervals.</li>
        <li><strong>The Differencing Solution (&Delta;y<sub>t</sub> = y<sub>t</sub> - y<sub>t-1</sub>):</strong> Subtracting the previous day's or year's emission eliminates the stochastic trend while preserving day-to-day acceleration and volatility.</li>
        <li><strong>The ADF Hypothesis:</strong> The Augmented Dickey-Fuller (ADF) test tests the null hypothesis (H₀) that a unit-root exists (non-stationary). When the p-value is <strong>p &lt; 0.05</strong>, we reject H₀ and confirm stationarity.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# ADF TEST RESULTS TABLE WITH PASS/FAIL COLORING
# ==============================================================================
st.markdown('<div class="section-header">📊 Augmented Dickey-Fuller (ADF) Test Results</div>', unsafe_allow_html=True)

df_adf = load_adf_results()

if not df_adf.empty:
    # Filter controls for the table
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        diff_filter = st.selectbox(
            "Filter by Differencing State:",
            options=["All Tests", "Raw Series (d=0)", "1st Difference (d=1)"],
            index=0
        )

    df_display = df_adf.copy()
    if diff_filter == "Raw Series (d=0)":
        df_display = df_display[df_display["differencing_applied"].str.contains("None|Raw")]
    elif diff_filter == "1st Difference (d=1)":
        df_display = df_display[df_display["differencing_applied"].str.contains("1st Difference")]

    # Styling helper for pass/fail
    def highlight_stationary(val):
        if val == "Yes":
            return "background-color: #D1FAE5; color: #065F46; font-weight: bold;"
        elif val == "No":
            return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
        return ""

    def highlight_pvalue(val):
        try:
            p = float(val)
            if p < 0.05:
                return "color: #059669; font-weight: 600;"
            else:
                return "color: #DC2626; font-weight: 600;"
        except:
            return ""

    # Use Styler.map (or fallback to applymap for older pandas)
    map_func = getattr(df_display.style, "map", df_display.style.applymap)
    styled_df = map_func(highlight_stationary, subset=["stationary"])
    map_func2 = getattr(styled_df, "map", styled_df.applymap)
    styled_df = map_func2(highlight_pvalue, subset=["p_value"])\
        .format({"adf_statistic": "{:.3f}", "p_value": "{:.4f}"})

    st.dataframe(styled_df, use_container_width=True)

    render_chart_caption(
        "In the table above, green rows (Yes) indicate stationary series where p-value < 0.05 (rejecting null hypothesis of a unit-root). "
        "Notice that ALL raw series (d=0) fail the stationarity test due to trends, whereas applying first-order differencing (d=1) successfully achieves stationarity."
    )

# ==============================================================================
# BEFORE / AFTER DIFFERENCING INTERACTIVE CHART
# ==============================================================================
st.markdown('<div class="section-header">📉 Interactive Before vs After Differencing Diagnostics</div>', unsafe_allow_html=True)

series_choice = st.selectbox(
    "Select Series to Inspect Differencing Transformation:",
    options=[
        "Daily: China Total CO2",
        "Daily: India Total CO2",
        "Daily: USA Total CO2",
        "Daily: EU27+UK Total CO2",
        "Annual: China CO2",
        "Annual: India CO2",
        "Annual: USA CO2",
        "Annual: Germany CO2"
    ],
    index=0
)

# Extract series data based on user selection
if series_choice.startswith("Daily:"):
    region = series_choice.split(":")[1].replace("Total CO2", "").strip()
    df_raw = load_daily_emissions()
    df_sub = df_raw[df_raw["region"] == region].sort_values("date").copy()
    time_index = df_sub["date"]
    y_raw = df_sub["total"]
    unit = "MtCO₂/day"
    title_suffix = f"Daily Emissions — {region}"
else:
    country = series_choice.split(":")[1].replace("CO2", "").strip()
    if country == "USA":
        country = "United States"
    df_raw = load_annual_indicators()
    df_sub = df_raw[df_raw["country"] == country].sort_values("year").copy()
    time_index = df_sub["year"]
    y_raw = df_sub["co2"]
    unit = "MtCO₂/yr"
    title_suffix = f"Annual Emissions — {country}"

# Compute First Difference: Delta y_t = y_t - y_{t-1}
y_diff = y_raw.diff()

# Rolling statistics to demonstrate constant mean & variance
rolling_win = 30 if series_choice.startswith("Daily:") else 5
rolling_mean_raw = y_raw.rolling(window=rolling_win, min_periods=1).mean()
rolling_mean_diff = y_diff.rolling(window=rolling_win, min_periods=1).mean()
rolling_std_diff = y_diff.rolling(window=rolling_win, min_periods=1).std()

# Create 2-row subplot
fig_diff = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=(
        f"1. Before Differencing: Raw Series (Non-Stationary with Trend & Seasonality)",
        f"2. After Differencing: 1st Difference Δyₜ = yₜ - yₜ₋₁ (Stationary: Mean ≈ 0, Constant Variance)"
    )
)

# Trace 1: Raw Series
raw_color = "#38BDF8" if dark else "#1E3A8A"
diff_color = "#2DD4BF" if dark else "#0D9488"
paper_bg = "#1E293B" if dark else "#FFFFFF"
plot_bg = "#0B1120" if dark else "#FAFAFA"
grid_color = "#334155" if dark else "#E2E8F0"
font_color = "#F8FAFC" if dark else "#0F172A"

fig_diff.add_trace(
    go.Scatter(
        x=time_index,
        y=y_raw,
        name=f"Raw Series ({unit})",
        line=dict(color=raw_color, width=1.8),
        hovertemplate="<b>Raw Value:</b> %{y:.2f}<extra></extra>"
    ),
    row=1, col=1
)

# Trace 2: Rolling Mean on Raw
fig_diff.add_trace(
    go.Scatter(
        x=time_index,
        y=rolling_mean_raw,
        name="Rolling Mean (Shows Trend)",
        line=dict(color="#EF4444", width=2, dash="dash"),
        hovertemplate="<b>Rolling Mean:</b> %{y:.2f}<extra></extra>"
    ),
    row=1, col=1
)

# Trace 3: Differenced Series
fig_diff.add_trace(
    go.Scatter(
        x=time_index,
        y=y_diff,
        name=f"Differenced Δy ({unit})",
        line=dict(color=diff_color, width=1.4),
        hovertemplate="<b>Δy:</b> %{y:.2f}<extra></extra>"
    ),
    row=2, col=1
)

# Trace 4: Zero-mean reference line
fig_diff.add_trace(
    go.Scatter(
        x=time_index,
        y=[0] * len(time_index),
        name="Zero Baseline",
        line=dict(color="#94A3B8" if dark else "#64748B", width=1.5, dash="dot"),
        hoverinfo="skip"
    ),
    row=2, col=1
)

# Trace 5: Rolling Std on Diff
fig_diff.add_trace(
    go.Scatter(
        x=time_index,
        y=rolling_std_diff,
        name="Rolling Std Dev (Constant Variance)",
        line=dict(color="#F59E0B", width=1.8),
        hovertemplate="<b>Rolling Std:</b> ±%{y:.2f}<extra></extra>"
    ),
    row=2, col=1
)

fig_diff.update_layout(
    height=600,
    template="plotly_dark" if dark else "plotly_white",
    paper_bgcolor=paper_bg,
    plot_bgcolor=plot_bg,
    margin=dict(l=55, r=30, t=55, b=45),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center", font=dict(color=font_color))
)
fig_diff.update_yaxes(title_text=f"Raw ({unit})", row=1, col=1, gridcolor=grid_color, tickfont=dict(color=font_color), titlefont=dict(color=font_color))
fig_diff.update_yaxes(title_text=f"Differenced Δ ({unit})", row=2, col=1, gridcolor=grid_color, tickfont=dict(color=font_color), titlefont=dict(color=font_color))
fig_diff.update_xaxes(title_text="Time Horizon", row=2, col=1, gridcolor=grid_color, tickfont=dict(color=font_color), titlefont=dict(color=font_color))

st.plotly_chart(fig_diff, use_container_width=True)

render_chart_caption(
    f"Comparison for {title_suffix}. The top plot exhibits a trending mean (red dashed line), confirming non-stationarity. "
    "The bottom plot shows that after taking the first difference (d=1), the series oscillates tightly around zero (grey dotted line) "
    "with stable rolling standard deviation (amber line), satisfying the stationarity requirement for ARIMA/SARIMA baselines."
)
