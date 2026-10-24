"""
Configuration file for CarbonLens Streamlit Dashboard.
Change DATA_MODE to "real" when real-world datasets are acquired.
All file names and column schemas remain identical between sample and real modes.
"""

from pathlib import Path

# ==============================================================================
# 1. DATA SOURCE CONFIGURATION
# Set to "sample" for prototype demonstration or "real" for production datasets
# ==============================================================================
DATA_MODE = "sample"  # Options: "sample" or "real"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / DATA_MODE

# Standard File Paths
DAILY_EMISSIONS_FILE = DATA_DIR / "daily_emissions.csv"
ANNUAL_INDICATORS_FILE = DATA_DIR / "annual_indicators.csv"
NDC_TARGETS_FILE = DATA_DIR / "ndc_targets.csv"
ADF_RESULTS_FILE = DATA_DIR / "adf_results.csv"
SELECTION_FILE = DATA_DIR / "selection.csv"
BASELINE_RESULTS_FILE = DATA_DIR / "baseline_results.csv"
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"

# ==============================================================================
# 2. PROJECT METADATA
# ==============================================================================
PROJECT_TITLE = "CarbonLens"
PROJECT_FULL_TITLE = "CarbonLens: AI-based Carbon Emission Analysis and Forecasting"
PROJECT_SUBTITLE = "B.Tech Final Year Project — Phase 1 Demonstration"
INSTITUTION = "Babu Banarasi Das Institute of Technology and Management (BBDITM), Lucknow"
SUPERVISOR = "Ms. Saloni Ahuja"

TEAM_MEMBERS = [
    {
        "name": "Rajat Singh",
        "role": "Models and Evaluation",
        "responsibilities": "Statistical baselines (ARIMA/SARIMA), ML models (SVR, RF, Boosting), cross-validation, and metrics bench.",
        "avatar": "🤖"
    },
    {
        "name": "Pranav Kumar Singh",
        "role": "Data Pipeline, SHAP, NDC & Dashboard",
        "responsibilities": "Multi-scale data ingestion, Streamlit dashboard architecture, NDC gap calculator, and Phase 2 SHAP framework.",
        "avatar": "💻"
    },
    {
        "name": "Pradhyumna Sharma",
        "role": "Data Collection, Results, Testing & Documentation",
        "responsibilities": "Historical indicator curation, stationarity & Granger causality tests, QA validation, and project reports.",
        "avatar": "📋"
    }
]

# ==============================================================================
# 3. DOMAIN CONSTANTS & METRICS
# ==============================================================================
DAILY_REGIONS = ["China", "India", "USA", "EU27+UK"]

ANNUAL_COUNTRIES = [
    "China", "United States", "India", "Russia", "Japan",
    "Germany", "Iran", "South Korea", "Saudi Arabia", "Indonesia", "Canada"
]

BASELINE_MODELS = [
    "ARIMA",
    "SARIMA",
    "SVR",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "Voting",
    "Bagging"
]

METRICS = ["MAE", "RMSE", "MAPE", "R2"]

# Consistent UI Colors
COLORS = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#0D9488",    # Teal
    "accent": "#F59E0B",       # Warm Amber
    "danger": "#EF4444",       # Crimson
    "success": "#10B981",      # Emerald Green
    "background": "#F8FAFC",   # Very Light Slate
    "card_bg": "#FFFFFF",      # Pure White
    "text": "#0F172A",         # Deep Slate
    "muted": "#64748B",        # Cool Grey
    "train": "#3B82F6",        # Blue
    "val": "#F59E0B",          # Amber
    "test": "#10B981"          # Green
}

# Sector Colors for Daily breakdown
SECTOR_COLORS = {
    "power": "#3B82F6",        # Blue
    "industry": "#F97316",     # Orange
    "transport": "#10B981",    # Emerald
    "residential": "#8B5CF6"   # Purple
}
