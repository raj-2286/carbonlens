"""
Utility functions for CarbonLens Streamlit Dashboard.
Includes cached data loaders, UI banners, Plotly chart theming (Light & Dark modes),
and graceful missing-file error handlers.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import config


# ==============================================================================
# 1. THEME DETECTION & HELPERS
# ==============================================================================
def get_current_theme() -> str:
    """Return 'Light' or 'Dark' from st.session_state, defaulting to 'Light'."""
    return st.session_state.get("theme", "Light")


def is_dark_mode() -> bool:
    """Check if Dark Mode is active."""
    return get_current_theme() == "Dark"


# ==============================================================================
# 2. PAGE SETUP & DUAL-THEME CSS INJECTION
# ==============================================================================
def setup_page(title: str, icon: str = "🌱"):
    """
    Standardize page configuration and inject custom CSS for both Light and Dark modes.
    Ensures headings, metric cards, and text are 100% visible and high-contrast in both modes.
    """
    try:
        st.set_page_config(
            page_title=f"{title} | CarbonLens",
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        # Already set by app.py entrypoint
        pass

    dark = is_dark_mode()

    if dark:
        # ---------------- DARK MODE STYLING ----------------
        css = """
        <style>
            /* Main container */
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
                max-width: 96%;
                color: #F8FAFC !important;
            }
            .stApp {
                background-color: #0B1120 !important;
                color: #F8FAFC !important;
            }

            /* ALL HEADINGS HIGH-CONTRAST IN DARK MODE */
            h1, h2, h3, h4, h5, h6,
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #F8FAFC !important;
                font-weight: 700 !important;
            }

            /* Section Callout Headers */
            .section-header {
                border-left: 5px solid #38BDF8 !important;
                padding-left: 12px !important;
                margin-top: 1.6rem !important;
                margin-bottom: 1.1rem !important;
                font-size: 1.35rem !important;
                font-weight: 800 !important;
                color: #38BDF8 !important;
                letter-spacing: 0.3px !important;
                text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            }

            /* Paragraphs, labels, and text */
            p, span, li, label, .stMarkdown p, .stMarkdown li {
                color: #E2E8F0 !important;
            }
            strong, b {
                color: #FFFFFF !important;
            }

            /* Metric Cards */
            .metric-card {
                background-color: #1E293B !important;
                border: 1px solid #334155 !important;
                border-radius: 10px;
                padding: 16px 20px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3) !important;
                transition: all 0.2s ease;
            }
            .metric-card:hover {
                border-color: #38BDF8 !important;
                box-shadow: 0 6px 12px rgba(0,0,0,0.5) !important;
            }
            .metric-label {
                font-size: 13px !important;
                font-weight: 700 !important;
                color: #94A3B8 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
            }
            .metric-value {
                font-size: 26px !important;
                font-weight: 800 !important;
                color: #F8FAFC !important;
                margin-top: 4px !important;
            }
            .metric-sub {
                font-size: 12px !important;
                color: #CBD5E1 !important;
                margin-top: 2px !important;
            }

            /* How to read this chart box */
            .chart-caption-box {
                background-color: #1E293B !important;
                border: 1px solid #334155 !important;
                border-left: 5px solid #14B8A6 !important;
                border-radius: 4px 8px 8px 4px;
                padding: 12px 16px !important;
                margin-top: 8px !important;
                margin-bottom: 20px !important;
                font-size: 13.5px !important;
                color: #F1F5F9 !important;
                line-height: 1.55 !important;
            }

            /* Custom inline card overrides for Dark Mode */
            div[style*="background: white"],
            div[style*="background-color: white"],
            div[style*="background: #FFFFFF"],
            div[style*="background-color: #FFFFFF"],
            div[style*="background:#FFFFFF"],
            div[style*="background-color:#FFFFFF"],
            div[style*="background: #F8FAFC"],
            div[style*="background-color: #F8FAFC"],
            div[style*="background: #EFF6FF"],
            div[style*="background-color: #EFF6FF"],
            div[style*="background: #ECFDF5"],
            div[style*="background-color: #ECFDF5"],
            div[style*="background: #F5F3FF"],
            div[style*="background-color: #F5F3FF"] {
                background-color: #1E293B !important;
                border-color: #334155 !important;
                color: #F8FAFC !important;
            }

            /* Fix inline text colors inside custom divs */
            div[style*="background"] h3,
            div[style*="background"] h4,
            div[style*="background"] strong {
                color: #F8FAFC !important;
            }
            div[style*="background"] p,
            div[style*="background"] ul,
            div[style*="background"] li,
            div[style*="background"] div {
                color: #E2E8F0 !important;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background-color: #0F172A !important;
                border-right: 1px solid #1E293B !important;
            }
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] label {
                color: #F8FAFC !important;
            }

            /* Tables & Dataframes */
            .stDataFrame {
                border: 1px solid #334155 !important;
                border-radius: 8px;
            }
            
            /* Radio and selectbox labels */
            .stRadio label, .stSelectbox label, .stMultiSelect label, .stDateInput label {
                color: #F8FAFC !important;
                font-weight: 600 !important;
            }
        </style>
        """
    else:
        # ---------------- LIGHT MODE STYLING ----------------
        css = """
        <style>
            /* Main container */
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
                max-width: 96%;
                color: #0F172A !important;
            }
            .stApp {
                background-color: #F8FAFC !important;
                color: #0F172A !important;
            }

            /* ALL HEADINGS IN LIGHT MODE */
            h1, h2, h3, h4, h5, h6,
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #0F172A !important;
                font-weight: 700 !important;
            }

            /* Section Callout Headers */
            .section-header {
                border-left: 5px solid #1E3A8A !important;
                padding-left: 12px !important;
                margin-top: 1.6rem !important;
                margin-bottom: 1.1rem !important;
                font-size: 1.35rem !important;
                font-weight: 800 !important;
                color: #0F172A !important;
                letter-spacing: 0.3px !important;
            }

            /* Paragraphs, labels, and text */
            p, span, li, label, .stMarkdown p, .stMarkdown li {
                color: #334155 !important;
            }
            strong, b {
                color: #0F172A !important;
            }

            /* Metric Cards */
            .metric-card {
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 10px;
                padding: 16px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
                transition: all 0.2s ease;
            }
            .metric-card:hover {
                border-color: #94A3B8 !important;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08) !important;
            }
            .metric-label {
                font-size: 13px !important;
                font-weight: 700 !important;
                color: #64748B !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
            }
            .metric-value {
                font-size: 26px !important;
                font-weight: 800 !important;
                color: #0F172A !important;
                margin-top: 4px !important;
            }
            .metric-sub {
                font-size: 12px !important;
                color: #64748B !important;
                margin-top: 2px !important;
            }

            /* How to read this chart box */
            .chart-caption-box {
                background-color: #F8FAFC !important;
                border: 1px solid #E2E8F0 !important;
                border-left: 5px solid #0D9488 !important;
                border-radius: 4px 8px 8px 4px;
                padding: 12px 16px !important;
                margin-top: 8px !important;
                margin-bottom: 20px !important;
                font-size: 13.5px !important;
                color: #334155 !important;
                line-height: 1.55 !important;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background-color: #FFFFFF !important;
                border-right: 1px solid #E2E8F0 !important;
            }

            /* Tables & Dataframes */
            .stDataFrame {
                border: 1px solid #E2E8F0 !important;
                border-radius: 8px;
            }

            /* Radio and selectbox labels */
            .stRadio label, .stSelectbox label, .stMultiSelect label, .stDateInput label {
                color: #0F172A !important;
                font-weight: 600 !important;
            }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)


def render_prototype_banner():
    """
    Renders the prominent warning banner on every page:
    'PROTOTYPE: sample data for demonstration, not real results'
    Dynamically adapts styling to Light or Dark mode for 100% readability.
    """
    dark = is_dark_mode()
    mode_text = f"data/{config.DATA_MODE}"

    if dark:
        banner_bg = "#221703"
        banner_border = "#B45309"
        banner_left = "#F59E0B"
        title_color = "#FDE047"       # Vibrant bright gold/yellow
        subtitle_color = "#FCD34D"    # Soft warm amber
        badge_bg = "#F59E0B"
        badge_fg = "#0F172A"          # Dark text on bright badge
        badge_label = "PROTOTYPE DATA" if config.DATA_MODE == "sample" else "REAL DATA ACTIVE"
    else:
        banner_bg = "#FFFBEB"
        banner_border = "#FCD34D"
        banner_left = "#F59E0B"
        title_color = "#92400E"       # Crisp dark amber
        subtitle_color = "#B45309"
        badge_bg = "#FDE68A" if config.DATA_MODE == "sample" else "#D1FAE5"
        badge_fg = "#92400E" if config.DATA_MODE == "sample" else "#065F46"
        badge_label = "PROTOTYPE DATA" if config.DATA_MODE == "sample" else "REAL DATA ACTIVE"

    st.markdown(f"""
    <div style="background-color: {banner_bg}; border: 1px solid {banner_border}; border-left: 6px solid {banner_left}; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 26px; line-height: 1;">⚠️</span>
            <div>
                <strong style="color: {title_color}; font-size: 15px; letter-spacing: 0.4px; text-transform: uppercase;">
                    PROTOTYPE: sample data for demonstration, not real results
                </strong>
                <div style="color: {subtitle_color}; font-size: 12.5px; margin-top: 2px;">
                    Source Directory: <code>{mode_text}</code> &bull; Config: <code>config.py &rarr; DATA_MODE = "{config.DATA_MODE}"</code> &bull; Prepared for B.Tech Major Project guide review.
                </div>
            </div>
        </div>
        <div style="background-color: {badge_bg}; color: {badge_fg}; font-size: 12px; font-weight: 800; padding: 5px 12px; border-radius: 12px; border: 1px solid #F59E0B; white-space: nowrap;">
            {badge_label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chart_caption(text: str):
    """
    Renders a clean 'How to read this chart' caption beneath main charts.
    """
    st.markdown(f"""
    <div class="chart-caption-box">
        <strong>💡 How to read this chart:</strong> {text}
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. CACHED DATA LOADERS WITH GRACEFUL ERROR HANDLING
# ==============================================================================
def handle_missing_file(file_path: Path, dataset_name: str):
    """Display a user-friendly error message if data file is missing."""
    st.error(
        f"❌ **Missing Dataset**: `{file_path.name}` not found in `{file_path.parent}`.\n\n"
        f"To generate the prototype dataset, run in your terminal:\n\n"
        f"```bash\npython src/make_sample_data.py\n```\n\n"
        f"Or verify `DATA_MODE` in `config.py`."
    )


@st.cache_data(show_spinner="Loading daily emissions data...")
def load_daily_emissions():
    """Load daily CO2 emissions dataset."""
    if not config.DAILY_EMISSIONS_FILE.exists():
        handle_missing_file(config.DAILY_EMISSIONS_FILE, "Daily Emissions")
        return pd.DataFrame()
    df = pd.read_csv(config.DAILY_EMISSIONS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="Loading annual indicators data...")
def load_annual_indicators():
    """Load annual macroeconomic and energy indicators dataset."""
    if not config.ANNUAL_INDICATORS_FILE.exists():
        handle_missing_file(config.ANNUAL_INDICATORS_FILE, "Annual Indicators")
        return pd.DataFrame()
    df = pd.read_csv(config.ANNUAL_INDICATORS_FILE)
    df["year"] = df["year"].astype(int)
    return df


@st.cache_data(show_spinner="Loading NDC targets data...")
def load_ndc_targets():
    """Load 2030 NDC targets dataset."""
    if not config.NDC_TARGETS_FILE.exists():
        handle_missing_file(config.NDC_TARGETS_FILE, "NDC Targets")
        return pd.DataFrame()
    return pd.read_csv(config.NDC_TARGETS_FILE)


@st.cache_data(show_spinner="Loading ADF test results...")
def load_adf_results():
    """Load stationarity ADF test results."""
    if not config.ADF_RESULTS_FILE.exists():
        handle_missing_file(config.ADF_RESULTS_FILE, "ADF Results")
        return pd.DataFrame()
    return pd.read_csv(config.ADF_RESULTS_FILE)


@st.cache_data(show_spinner="Loading lag and feature selection data...")
def load_selection_results():
    """Load Granger lag selection and feature importance results."""
    if not config.SELECTION_FILE.exists():
        handle_missing_file(config.SELECTION_FILE, "Lag/Feature Selection")
        return pd.DataFrame()
    return pd.read_csv(config.SELECTION_FILE)


@st.cache_data(show_spinner="Loading baseline model results...")
def load_baseline_results():
    """Load baseline model benchmark results."""
    if not config.BASELINE_RESULTS_FILE.exists():
        handle_missing_file(config.BASELINE_RESULTS_FILE, "Baseline Results")
        return pd.DataFrame()
    return pd.read_csv(config.BASELINE_RESULTS_FILE)


@st.cache_data(show_spinner="Loading model predictions...")
def load_predictions():
    """Load test period predictions for models."""
    if not config.PREDICTIONS_FILE.exists():
        handle_missing_file(config.PREDICTIONS_FILE, "Model Predictions")
        return pd.DataFrame()
    df = pd.read_csv(config.PREDICTIONS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ==============================================================================
# 4. PLOTLY THEME & HELPER FUNCTIONS (DUAL MODE AWARE)
# ==============================================================================
def apply_plotly_theme(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = ""):
    """
    Apply a consistent presentation layout to Plotly figures.
    Dynamically switches between Light and Dark mode styling.
    """
    dark = is_dark_mode()

    if dark:
        template = "plotly_dark"
        paper_bg = "#1E293B"     # Dark Slate Card
        plot_bg = "#0B1120"      # Deep Night
        text_color = "#F8FAFC"   # Crisp White-Slate
        grid_color = "#334155"   # Subtle Slate Grid
        axis_color = "#64748B"   # Subtle Border
        subtext_color = "#CBD5E1"
    else:
        template = "plotly_white"
        paper_bg = "#FFFFFF"     # Pure White
        plot_bg = "#FAFAFA"      # Very Light Slate
        text_color = "#0F172A"   # Deep Slate
        grid_color = "#E2E8F0"   # Light Grey Grid
        axis_color = "#CBD5E1"   # Clean Grey Line
        subtext_color = "#475569"

    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": text_color, "family": "Arial, sans-serif"}
        },
        template=template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        margin=dict(l=55, r=30, t=55, b=45),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=subtext_color)
        ),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor=grid_color,
            linecolor=axis_color,
            tickfont=dict(color=subtext_color, size=11),
            titlefont=dict(color=text_color, size=12, family="Arial, sans-serif")
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor=grid_color,
            linecolor=axis_color,
            tickfont=dict(color=subtext_color, size=11),
            titlefont=dict(color=text_color, size=12, family="Arial, sans-serif")
        )
    )
    return fig
