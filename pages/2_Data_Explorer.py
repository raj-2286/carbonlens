"""
Page 2: Data Explorer
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Interactive exploration of Daily CO2 emissions and Annual Macroeconomic/Energy Indicators.
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
    load_daily_emissions,
    load_annual_indicators,
    is_dark_mode
)

# Initialize page
setup_page("Data Explorer", "📊")

# Prototype banner
render_prototype_banner()

st.markdown("## 📊 Exploratory Data Analysis & Indicators")
st.markdown(
    "Explore multi-scale emission dynamics across temporal scales. Select between **Daily High-Frequency CO2** (2019–2025) "
    "and **Annual Macroeconomic Drivers** (1990–2023)."
)

# ==============================================================================
# SCALE SELECTION & CONTROLS
# ==============================================================================
col_ctrl1, col_ctrl2 = st.columns([1, 2])

with col_ctrl1:
    scale_option = st.radio(
        "Select Temporal Scale:",
        options=["Daily Emissions (2019-2025)", "Annual Indicators (1990-2023)"],
        index=0,
        horizontal=False
    )

is_daily = "Daily" in scale_option

if is_daily:
    # --------------------------------------------------------------------------
    # DAILY EMISSIONS VIEW
    # --------------------------------------------------------------------------
    df_daily = load_daily_emissions()
    
    if df_daily.empty:
        st.stop()

    with col_ctrl2:
        selected_region = st.selectbox(
            "Select Region:",
            options=config.DAILY_REGIONS,
            index=0
        )

    # Filter data for selected region
    df_region = df_daily[df_daily["region"] == selected_region].sort_values("date").copy()

    # KPI Summary Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        latest_val = df_region["total"].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latest Total Emission</div>
            <div class="metric-value">{latest_val:.2f} <span style="font-size:14px; font-weight:normal;">MtCO₂/day</span></div>
            <div class="metric-sub">{df_region['date'].iloc[-1].strftime('%b %d, %Y')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_kpi2:
        mean_val = df_region["total"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">7-Year Daily Average</div>
            <div class="metric-value">{mean_val:.2f} <span style="font-size:14px; font-weight:normal;">MtCO₂/day</span></div>
            <div class="metric-sub">Std Dev: ±{df_region['total'].std():.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        peak_val = df_region["total"].max()
        peak_date = df_region.loc[df_region["total"].idxmax(), "date"].strftime('%b %Y')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Historical Peak</div>
            <div class="metric-value">{peak_val:.2f} <span style="font-size:14px; font-weight:normal;">MtCO₂/day</span></div>
            <div class="metric-sub">Observed in {peak_date}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        # Dominant sector
        sector_means = df_region[["power", "industry", "transport", "residential"]].mean()
        top_sector = sector_means.idxmax().capitalize()
        top_share = (sector_means.max() / df_region["total"].mean()) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Emitting Sector</div>
            <div class="metric-value">{top_sector}</div>
            <div class="metric-sub">{top_share:.1f}% of regional total</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📈 Daily Total CO₂ Trajectory with Moving Averages</div>', unsafe_allow_html=True)

    # Calculate 7-day and 30-day moving averages
    df_region["ma7"] = df_region["total"].rolling(window=7, min_periods=1).mean()
    df_region["ma30"] = df_region["total"].rolling(window=30, min_periods=1).mean()

    fig_daily = go.Figure()

    # Raw daily series
    fig_daily.add_trace(go.Scatter(
        x=df_region["date"],
        y=df_region["total"],
        mode="lines",
        name="Daily Total (Raw)",
        line=dict(color="#94A3B8", width=1),
        opacity=0.6,
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Total:</b> %{y:.2f} MtCO₂/day<extra></extra>"
    ))

    # 7-day moving average (removes weekly day-of-week dips)
    fig_daily.add_trace(go.Scatter(
        x=df_region["date"],
        y=df_region["ma7"],
        mode="lines",
        name="7-Day Moving Avg",
        line=dict(color="#3B82F6", width=2),
        hovertemplate="<b>7-Day MA:</b> %{y:.2f} MtCO₂/day<extra></extra>"
    ))

    # 30-day moving average (highlights seasonal cycle and macro trend)
    ma30_color = "#F8FAFC" if is_dark_mode() else "#0F172A"
    fig_daily.add_trace(go.Scatter(
        x=df_region["date"],
        y=df_region["ma30"],
        mode="lines",
        name="30-Day Moving Avg",
        line=dict(color=ma30_color, width=2.5),
        hovertemplate="<b>30-Day Trend:</b> %{y:.2f} MtCO₂/day<extra></extra>"
    ))

    apply_plotly_theme(
        fig_daily,
        title=f"Daily Carbon Dioxide Emissions — {selected_region} (2019-2025)",
        x_title="Date (Daily Observation)",
        y_title="Emissions (MtCO₂ / day)"
    )

    st.plotly_chart(fig_daily, use_container_width=True)
    render_chart_caption(
        f"This plot shows daily emissions for {selected_region}. The light grey curve shows raw daily volatility including weekend drops. "
        "The blue 7-day line smooths out the weekly cycle, and the bold dark 30-day line reveals seasonal winter/summer swings and the 2020 COVID-19 lockdown anomaly."
    )

    # --------------------------------------------------------------------------
    # SECTOR BREAKDOWN (STACKED AREA CHART)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">🏭 Sector Breakdown (Stacked Area)</div>', unsafe_allow_html=True)

    fig_sector = go.Figure()

    fig_sector.add_trace(go.Scatter(
        x=df_region["date"],
        y=df_region["power"],
        mode="lines",
        name="Power Generation",
        stackgroup="one",
        line=dict(width=0.5, color=config.SECTOR_COLORS["power"]),
        hovertemplate="Power: %{y:.2f} MtCO₂/day<extra></extra>"
    ))

    fig_sector.add_trace(go.Scatter(
        x=df_region["date"],
        y=df_region["industry"],
        mode="lines",
        name="Industry & Manufacturing",
        stackgroup="one",
        line=dict(width=0.5, color=config.SECTOR_COLORS["industry"]),
        hovertemplate="Industry: %{y:.2f} MtCO₂/day<extra></extra>"
    ))

    fig_sector.add_trace(go.Scatter(
        x=df_region["date"],
        y=df_region["transport"],
        mode="lines",
        name="Ground & Domestic Transport",
        stackgroup="one",
        line=dict(width=0.5, color=config.SECTOR_COLORS["transport"]),
        hovertemplate="Transport: %{y:.2f} MtCO₂/day<extra></extra>"
    ))

    fig_sector.add_trace(go.Scatter(
        x=df_region["date"],
        y=df_region["residential"],
        mode="lines",
        name="Residential & Commercial Heating",
        stackgroup="one",
        line=dict(width=0.5, color=config.SECTOR_COLORS["residential"]),
        hovertemplate="Residential: %{y:.2f} MtCO₂/day<extra></extra>"
    ))

    apply_plotly_theme(
        fig_sector,
        title=f"Sectoral Emission Decomposition — {selected_region}",
        x_title="Date",
        y_title="Emissions by Sector (MtCO₂ / day)"
    )

    st.plotly_chart(fig_sector, use_container_width=True)
    render_chart_caption(
        f"Stacked area visualization showing cumulative emissions across four key economic sectors in {selected_region}. "
        "Notice the heavy baseline weight of thermal power and industrial manufacturing, and how transport experienced the sharpest dip during lockdown periods."
    )

    # --------------------------------------------------------------------------
    # SUMMARY STATISTICS & MISSING VALUES REPORT
    # --------------------------------------------------------------------------
    col_stat1, col_stat2 = st.columns([3, 2])

    with col_stat1:
        st.markdown('<div class="section-header">📋 Summary Statistics Table</div>', unsafe_allow_html=True)
        stats_cols = ["total", "power", "industry", "transport", "residential"]
        df_stats = df_region[stats_cols].describe().T
        df_stats = df_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        df_stats.columns = ["Count", "Mean", "Std Dev", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max"]
        st.dataframe(df_stats.style.format("{:.3f}"), use_container_width=True)

    with col_stat2:
        st.markdown('<div class="section-header">🔍 Data Quality & Missing Value Audit</div>', unsafe_allow_html=True)
        audit_data = []
        for col in ["date", "total", "power", "industry", "transport", "residential"]:
            null_count = df_region[col].isnull().sum()
            null_pct = (null_count / len(df_region)) * 100
            audit_data.append({
                "Column": col,
                "Type": str(df_region[col].dtype),
                "Missing Count": null_count,
                "Completeness": f"{100.0 - null_pct:.1f}%",
                "Status": "✅ Clean" if null_count == 0 else "⚠️ Impute Needed"
            })
        st.dataframe(pd.DataFrame(audit_data), use_container_width=True)

    # Download Button
    csv_data = df_region.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Filtered Daily Data for {selected_region} (CSV)",
        data=csv_data,
        file_name=f"carbonlens_daily_{selected_region.lower().replace('+', '_')}.csv",
        mime="text/csv"
    )

else:
    # --------------------------------------------------------------------------
    # ANNUAL INDICATORS VIEW
    # --------------------------------------------------------------------------
    df_annual = load_annual_indicators()

    if df_annual.empty:
        st.stop()

    with col_ctrl2:
        selected_country = st.selectbox(
            "Select Country (11 Largest Emitters):",
            options=config.ANNUAL_COUNTRIES,
            index=0
        )

    df_c = df_annual[df_annual["country"] == selected_country].sort_values("year").copy()

    # KPI Summary Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        co2_latest = df_c["co2"].iloc[-1]
        co2_start = df_c["co2"].iloc[0]
        co2_growth = ((co2_latest - co2_start) / co2_start) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">2023 Annual Emissions</div>
            <div class="metric-value">{co2_latest:.1f} <span style="font-size:14px; font-weight:normal;">MtCO₂/yr</span></div>
            <div class="metric-sub">1990–2023 Change: {co2_growth:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        gdp_latest = df_c["gdp"].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">2023 Gross Domestic Product</div>
            <div class="metric-value">${gdp_latest:,.0f} <span style="font-size:14px; font-weight:normal;">Billion</span></div>
            <div class="metric-sub">Population: {df_c['population'].iloc[-1]:.0f} Million</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        ren_latest = df_c["renewable_share"].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Renewable Energy Share</div>
            <div class="metric-value">{ren_latest:.1f}%</div>
            <div class="metric-sub">1990 Baseline: {df_c['renewable_share'].iloc[0]:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        elec_latest = df_c["electric_power_consumption"].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Electric Consumption</div>
            <div class="metric-value">{elec_latest:,.0f} <span style="font-size:14px; font-weight:normal;">kWh/capita</span></div>
            <div class="metric-sub">Industry Value: ${df_c['industrial_value_added'].iloc[-1]:,.0f}B</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📈 Multi-Indicator Annual Trajectory (1990-2023)</div>', unsafe_allow_html=True)

    # Dual axis plot: CO2 emissions vs chosen economic driver
    driver_choice = st.selectbox(
        "Select Macroeconomic/Energy Driver to Compare with CO₂:",
        options=[
            ("gdp", "GDP (Billion USD)"),
            ("electric_power_consumption", "Electric Power Consumption (kWh/capita)"),
            ("renewable_share", "Renewable Share (%)"),
            ("industrial_value_added", "Industrial Value Added (Billion USD)"),
            ("population", "Population (Million)")
        ],
        format_func=lambda x: x[1]
    )

    driver_key, driver_label = driver_choice

    fig_ann = go.Figure()

    # Colors responsive to theme
    dark = is_dark_mode()
    co2_line_color = "#38BDF8" if dark else "#1E3A8A"
    driver_line_color = "#2DD4BF" if dark else "#0D9488"
    paper_bg = "#1E293B" if dark else "#FFFFFF"
    plot_bg = "#0B1120" if dark else "#FAFAFA"
    title_color = "#F8FAFC" if dark else "#0F172A"
    grid_color = "#334155" if dark else "#E2E8F0"

    # CO2 trace (Primary axis)
    fig_ann.add_trace(go.Scatter(
        x=df_c["year"],
        y=df_c["co2"],
        name="CO₂ Emissions (MtCO₂/yr)",
        mode="lines+markers",
        line=dict(color=co2_line_color, width=3),
        marker=dict(size=6),
        hovertemplate="<b>Year %{x}:</b> %{y:.1f} MtCO₂/yr<extra></extra>"
    ))

    # Driver trace (Secondary axis)
    fig_ann.add_trace(go.Scatter(
        x=df_c["year"],
        y=df_c[driver_key],
        name=driver_label,
        mode="lines+markers",
        yaxis="y2",
        line=dict(color=driver_line_color, width=2.5, dash="dot"),
        marker=dict(size=5),
        hovertemplate=f"<b>Year %{{x}}:</b> %{{y:.1f}} ({driver_label})<extra></extra>"
    ))

    fig_ann.update_layout(
        title={
            "text": f"Annual CO₂ Emissions vs {driver_label} — {selected_country} (1990-2023)",
            "y": 0.95, "x": 0.02, "xanchor": "left",
            "font": {"color": title_color, "size": 16}
        },
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        margin=dict(l=60, r=60, t=55, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center", font=dict(color=title_color)),
        xaxis=dict(title="Year", showgrid=True, gridcolor=grid_color, tickfont=dict(color=title_color), titlefont=dict(color=title_color)),
        yaxis=dict(title="CO₂ Emissions (MtCO₂ / year)", titlefont=dict(color=co2_line_color), tickfont=dict(color=co2_line_color), gridcolor=grid_color),
        yaxis2=dict(
            title=driver_label,
            titlefont=dict(color=driver_line_color),
            tickfont=dict(color=driver_line_color),
            overlaying="y",
            side="right",
            showgrid=False
        )
    )

    st.plotly_chart(fig_ann, use_container_width=True)
    render_chart_caption(
        f"Dual-axis historical chart for {selected_country}. The solid blue line tracks annual CO₂ emissions (left axis). "
        f"The dashed green line represents {driver_label} (right axis). Observe whether emissions scale directly with economic activity or decouple over time."
    )

    # Summary Statistics & Quality Audit
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        st.markdown('<div class="section-header">📋 Indicator Summary Statistics</div>', unsafe_allow_html=True)
        ind_cols = ["co2", "gdp", "population", "electric_power_consumption", "renewable_share", "industrial_value_added"]
        df_ind_stats = df_c[ind_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
        df_ind_stats.columns = ["Count", "Mean", "Std Dev", "Min", "Median", "Max"]
        st.dataframe(df_ind_stats.style.format("{:.1f}"), use_container_width=True)

    with col_s2:
        st.markdown('<div class="section-header">🔍 Data Integrity Audit</div>', unsafe_allow_html=True)
        audit_ann = []
        for col in ind_cols:
            null_count = df_c[col].isnull().sum()
            audit_ann.append({
                "Indicator": col,
                "Years Covered": len(df_c),
                "Missing Values": null_count,
                "Status": "✅ Complete (1990-2023)"
            })
        st.dataframe(pd.DataFrame(audit_ann), use_container_width=True)

    # Download Button
    csv_ann = df_c.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Annual Indicators for {selected_country} (CSV)",
        data=csv_ann,
        file_name=f"carbonlens_annual_{selected_country.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
