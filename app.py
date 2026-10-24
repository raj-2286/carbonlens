"""
CarbonLens: AI-based Carbon Emission Analysis and Forecasting
Main entry point for Streamlit Dashboard Prototype.

B.Tech Final Year Major Project Dashboard.
Supervisor: Ms. Saloni Ahuja | BBDITM Lucknow
Team: Rajat Singh, Pradhyumna Sharma, Pranav Kumar Singh
"""

import streamlit as st

# Configure global page settings as the very first command
st.set_page_config(
    page_title="CarbonLens: AI-based Carbon Emission Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

import config

# Initialize Theme in Session State
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

# Define Streamlit Multi-Page Navigation with exact 1-to-9 page numbering
pages = [
    st.Page("pages/1_Home.py", title="1. Home", icon="🏠", default=True),
    st.Page("pages/2_Data_Explorer.py", title="2. Data Explorer", icon="📊"),
    st.Page("pages/3_Preprocessing.py", title="3. Preprocessing (ADF)", icon="🔬"),
    st.Page("pages/4_Train_Test_Split.py", title="4. Train/Val/Test Split", icon="✂️"),
    st.Page("pages/5_Lag_and_Feature_Selection.py", title="5. Lag & Feature Selection", icon="🎯"),
    st.Page("pages/6_Baseline_Models.py", title="6. Baseline Models", icon="🤖"),
    st.Page("pages/7_Forecast_Viewer.py", title="7. Forecast Viewer", icon="📈"),
    st.Page("pages/8_NDC_Gap_Preview.py", title="8. NDC Gap Preview", icon="🎯"),
    st.Page("pages/9_Roadmap.py", title="9. Future Scope & Team", icon="👥"),
]

# Initialize navigation
pg = st.navigation(pages)

# Render shared sidebar information
with st.sidebar:
    st.markdown("### 🌍 CarbonLens")
    st.caption("AI-based Carbon Emission Analysis & Forecasting")
    
    # Theme Switcher Control (Light / Dark Mode)
    st.markdown("#### 🎨 Appearance")
    theme_selection = st.radio(
        "Select Appearance Theme:",
        options=["☀️ Light Mode", "🌙 Dark Mode"],
        index=0 if st.session_state["theme"] == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="app_theme_toggle"
    )
    new_theme = "Dark" if "Dark" in theme_selection else "Light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown("---")
    
    # Active data mode indicator
    mode_badge = "🟡 Sample Mode (Prototype)" if config.DATA_MODE == "sample" else "🟢 Real Data Active"
    st.info(f"**Data Source**: {mode_badge}\n\n*Change in `config.py`*")
    
    st.markdown("---")
    st.markdown("**B.Tech Major Project**")
    st.caption(f"**Supervisor**: {config.SUPERVISOR}\n\n**Institution**: {config.INSTITUTION}")
    
    st.markdown("---")
    st.markdown("**Team Members**:")
    st.caption("• **Rajat Singh** (Models & Eval)\n• **Pranav Kumar Singh** (Pipeline & App)\n• **Pradhyumna Sharma** (Data & QA)")

# Run the selected page
pg.run()
