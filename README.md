# 🌍 CarbonLens: AI-based Carbon Emission Analysis and Forecasting

> **B.Tech Final Year Project — CarbonLens AI Forecasting Framework**  
> **Supervisor:** Ms. Saloni Ahuja (Assistant Professor)  
> **Institution:** Babu Banarasi Das Institute of Technology and Management (BBDITM), Lucknow  
> **Team:**  
> - **Rajat Singh**: Models & Evaluation Lead (Statistical Baselines, ML Ensembles, Error Benchmarks)  
> - **Pranav Kumar Singh**: Data Pipeline, SHAP & App Lead (ETL Ingestion, Streamlit Dashboard, NDC Analysis)  
> - **Pradhyumna Sharma**: Data Collection & Testing Lead (Indicator Curation, ADF/Granger Tests, QA Documentation)

---

## 📌 Project Overview

**CarbonLens** is an end-to-end artificial intelligence framework designed to overcome the 1–2 year reporting lag in conventional greenhouse gas emissions accounting. The platform demonstrates multi-scale temporal modeling (coupling high-frequency daily emissions with annual macroeconomic drivers), rigorous statistical preprocessing (stationarity testing and Granger lag selection), baseline forecasting model benchmarking across 8 algorithms, and Paris Agreement NDC target gap quantification.

> ⚠️ **IMPORTANT NOTICE:** Prototype sample datasets are currently active for academic demonstration (`DATA_MODE = "sample"` in `config.py`). A prominent banner appears on every dashboard page indicating that sample data is used for demonstration rather than real results.

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
# Navigate to project directory
cd "g:\carbon emission"

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generate Prototype Datasets (Seed = 42)
To generate the realistic synthetic datasets with complete reproducibility:
```bash
python src/make_sample_data.py
```
This generates 7 CSV files in `data/sample/`:
- `daily_emissions.csv` (10,228 rows: China, India, USA, EU27+UK from 2019 to 2025 with trend, weekly cycles, winter seasonality, and COVID dip)
- `annual_indicators.csv` (374 rows: 11 largest emitters from 1990 to 2023 with GDP, population, electricity consumption, renewable share, industrial value added)
- `ndc_targets.csv` (2030 NDC targets and descriptions)
- `adf_results.csv` (Augmented Dickey-Fuller stationarity test statistics and p-values before and after differencing)
- `selection.csv` (Granger causality optimal lags and feature importance scores)
- `baseline_results.csv` (MAE, RMSE, MAPE, R², and training times for 8 baseline models across daily and annual scales)
- `predictions.csv` (Out-of-sample actuals vs model predictions on the 2024–2025 test window)

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚙️ Switching from Prototype to Real Data

The architecture is designed so that transitioning to real-world datasets requires modifying only **one line** in `config.py`:

```python
# In config.py:
DATA_MODE = "real"   # Change from "sample" to "real"
```

Save real CSV files with the exact same column schemas in `data/real/`:
- `data/real/daily_emissions.csv`
- `data/real/annual_indicators.csv`
- `data/real/ndc_targets.csv`
- `data/real/adf_results.csv`
- `data/real/selection.csv`
- `data/real/baseline_results.csv`
- `data/real/predictions.csv`

The application automatically points all loaders, statistics, and visualizations to the new path without altering any UI or model code.

---

## 📂 Project Directory Structure

```
g:\carbon emission\
├── app.py                      # Main entrypoint and sidebar navigation
├── config.py                   # Central configuration (DATA_MODE, paths, metadata)
├── utils.py                    # Cached loaders, UI banners, Plotly theming, error handling
├── requirements.txt            # Python dependencies (streamlit, pandas, numpy, plotly, etc.)
├── README.md                   # Project documentation and run guide
├── src/
│   └── make_sample_data.py     # Deterministic synthetic data generator (seed 42)
├── pages/
│   ├── 1_Home.py               # Team, guide, synopsis summary, checklist & pipeline diagram
│   ├── 2_Data_Explorer.py      # Daily & annual emission trajectories, stacked areas, stats & download
│   ├── 3_Preprocessing.py      # ADF stationarity results table and before/after differencing chart
│   ├── 4_Train_Test_Split.py   # Chronological timeline chart and "no leakage" guarantee
│   ├── 5_Lag_and_Feature_Selection.py # Granger lag bar chart, feature rankings & selection gain
│   ├── 6_Baseline_Models.py    # 8-model benchmark comparison, best model callout & training time
│   ├── 7_Forecast_Viewer.py    # Actual vs predicted multi-model curves and residual analysis
│   ├── 8_NDC_Gap_Preview.py    # 2030 NDC gap quantification (G = E_2030 - T_NDC) and gauge chart
│   └── 9_Roadmap.py            # Architecture pillars, future scope & team responsibility matrix
└── data/
    ├── sample/                 # Generated prototype datasets (active)
    └── real/                   # Directory reserved for real-world datasets
```

---

## 🧪 Page-by-Page Verification Guide

To verify all features for your project guide demonstration:

1. **Page 1 (Home)**:
   - Check the prominent prototype warning banner at top.
   - Verify project metadata: Title, Guide (*Ms. Saloni Ahuja*), Institution (*BBDITM Lucknow*).
   - Inspect team role cards (*Rajat, Pranav, Pradhyumna*).
   - Review the System Deliverables & Implementation Checklist (all tick marks active).
   - Inspect the 5-step visual pipeline diagram (*Data &rarr; Preprocessing &rarr; Lag/Features &rarr; Baselines &rarr; Evaluation*).

2. **Page 2 (Data Explorer)**:
   - Toggle between **Daily** and **Annual** scales.
   - For daily, select *China, India, USA, or EU27+UK*: observe raw volatility, 7-day, and 30-day moving average lines.
   - Check the stacked area chart decomposing Power, Industry, Transport, and Residential sectors.
   - Test the CSV download button.
   - For annual, toggle across macroeconomic indicators (GDP, Renewable Share, Electric Consumption).

3. **Page 3 (Preprocessing)**:
   - Read the plain-language explanation of stationarity.
   - Review the ADF table: confirm red rows for raw non-stationary series ($p > 0.05$) and green rows for differenced stationary series ($p < 0.01$).
   - Use the dropdown to select a series: verify the top raw series vs the bottom differenced series ($\Delta y_t$) centered at zero.

4. **Page 4 (Train/Val/Test Split)**:
   - Inspect the "Guaranteed No-Leakage" protocol box.
   - Observe the horizontal chronological timeline chart (Blue = Train, Amber = Validation, Green = Test).
   - Check sample counts and dates for daily (2019–2022, 2023, 2024–2025) and annual (1990–2014, 2015–2019, 2020–2023).

5. **Page 5 (Lag & Feature Selection)**:
   - Inspect the Granger Causality bar chart showing optimal lags per driver (e.g. GDP lag 2, Power lag 1).
   - Filter top selected features and importance scores for China, USA, India, EU27+UK.
   - Review the With vs Without Selection RMSE chart showing a ~17% error reduction.

6. **Page 6 (Baseline Models)**:
   - Filter by scale (Daily vs Annual) and region.
   - Change the metric dropdown (MAPE, RMSE, MAE, R²): observe the best model callout card dynamically update.
   - Inspect the training time chart showing computational speed (SARIMA 45s vs XGBoost 3s).
   - Review and download the complete benchmark table.

7. **Page 7 (Forecast Viewer)**:
   - Select a region and multi-select models (e.g. Voting, XGBoost, SARIMA).
   - Zoom in using the date pickers.
   - Verify dynamic recalculation of MAE, RMSE, MAPE, and R² for the selected models.
   - Inspect the residual error series and the residual bell-curve distribution.

8. **Page 8 (NDC Gap Preview)**:
   - Note the prominent Policy Simulation & Target Analysis Framework banner.
   - Select a country (*India, China, USA, etc.*).
   - Observe the gap calculation: $G = E_{2030} - T_{\text{NDC}}$ in absolute ($MtCO_2$) and percentage terms.
   - Inspect the interactive gauge and milestone comparison bars.

9. **Page 9 (Future Scope & Team)**:
   - Review the Core Architectural Pillars (Multi-scale Ingestion, Causal Lag Selection, AI Benchmarking & Policy Gap).
   - Inspect Future Scope & System Extensions (Deep Learning Informer/Transformers, SHAP XAI, Policy knobs).
   - Review the detailed team allocation table confirming responsibilities for Rajat, Pranav, and Pradhyumna.
