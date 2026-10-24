"""
Synthetic Data Generator for CarbonLens Prototype.
Generates realistic prototype datasets for Phase 1 demonstration.
Saved to data/sample/ with random seed 42 for complete reproducibility.

All datasets are synthetic and strictly intended for prototype demonstration.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Set random seed for full reproducibility
SEED = 42
np.random.seed(SEED)

# Resolve directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_daily_emissions():
    """
    Generate daily CO2 emissions (MtCO2/day) from 2019-01-01 to 2025-12-31 for:
    China, India, USA, EU27+UK.
    Includes:
    - Long-term trend (growth in China & India, gradual decarbonization in USA & EU)
    - 7-day weekly cycle (lower emissions on weekends)
    - Annual seasonal cycle (winter heating peaks, summer cooling peaks, India monsoon)
    - 2020 COVID lockdown anomaly (Q1 for China, Q2 for USA/EU/India)
    - Sector breakdown: power, industry, transport, residential
    """
    print("-> Generating daily_emissions.csv...")
    dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    n_days = len(dates)
    
    regions_config = {
        "China": {
            "base": 27.5,
            "trend": 0.0022,  # Strong growth, tapering in 2024-2025
            "taper_start": 1826,  # 2024
            "weekly_amp": 0.08,   # 8% weekend drop
            "annual_amp": 2.2,    # Winter heating
            "covid_start": pd.Timestamp("2020-01-23"),
            "covid_peak": pd.Timestamp("2020-02-15"),
            "covid_end": pd.Timestamp("2020-04-15"),
            "covid_drop": 0.28,   # 28% drop at peak
            "sector_ratios": {"power": 0.46, "industry": 0.38, "transport": 0.10, "residential": 0.06},
            "noise_scale": 0.45
        },
        "India": {
            "base": 6.2,
            "trend": 0.00095, # Steady ~3.8% annual growth
            "taper_start": 9999,
            "weekly_amp": 0.07,
            "annual_amp": 0.7,   # Monsoon dip in Jul-Aug
            "covid_start": pd.Timestamp("2020-03-24"),
            "covid_peak": pd.Timestamp("2020-04-14"),
            "covid_end": pd.Timestamp("2020-06-15"),
            "covid_drop": 0.32,   # Strict nationwide lockdown
            "sector_ratios": {"power": 0.52, "industry": 0.28, "transport": 0.12, "residential": 0.08},
            "noise_scale": 0.16
        },
        "USA": {
            "base": 14.6,
            "trend": -0.00045, # Gradual decarbonization
            "taper_start": 9999,
            "weekly_amp": 0.12, # Strong weekend dip
            "annual_amp": 1.4,  # Dual peak: winter heating + summer AC
            "covid_start": pd.Timestamp("2020-03-15"),
            "covid_peak": pd.Timestamp("2020-04-20"),
            "covid_end": pd.Timestamp("2020-07-01"),
            "covid_drop": 0.20,
            "sector_ratios": {"power": 0.26, "industry": 0.21, "transport": 0.35, "residential": 0.18},
            "noise_scale": 0.32
        },
        "EU27+UK": {
            "base": 8.6,
            "trend": -0.00052, # Ongoing climate policy decline
            "taper_start": 9999,
            "weekly_amp": 0.14,
            "annual_amp": 1.5,  # Winter peak
            "covid_start": pd.Timestamp("2020-03-10"),
            "covid_peak": pd.Timestamp("2020-04-10"),
            "covid_end": pd.Timestamp("2020-06-25"),
            "covid_drop": 0.24,
            "sector_ratios": {"power": 0.24, "industry": 0.24, "transport": 0.30, "residential": 0.22},
            "noise_scale": 0.22
        }
    }

    records = []
    t = np.arange(n_days)
    day_of_year = dates.dayofyear.values
    day_of_week = dates.dayofweek.values  # 0=Monday, 6=Sunday

    for region, cfg in regions_config.items():
        # 1. Base + Trend
        trend_effect = cfg["trend"] * t
        # If taper, slow down trend in late years
        if cfg["taper_start"] < n_days:
            late_idx = t > cfg["taper_start"]
            trend_effect[late_idx] = (
                cfg["trend"] * cfg["taper_start"] +
                (cfg["trend"] * 0.25) * (t[late_idx] - cfg["taper_start"])
            )
        base_series = cfg["base"] + trend_effect

        # 2. Seasonality
        if region == "USA":
            # Dual peak: Winter (Jan) and Summer (Jul)
            season = (
                cfg["annual_amp"] * 0.7 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25) +
                cfg["annual_amp"] * 0.4 * np.cos(4 * np.pi * (day_of_year - 200) / 365.25)
            )
        elif region == "India":
            # Winter high, Monsoon low (Jul-Aug ~ day 210)
            season = cfg["annual_amp"] * np.cos(2 * np.pi * (day_of_year - 40) / 365.25)
            monsoon_mask = (day_of_year >= 180) & (day_of_year <= 250)
            season[monsoon_mask] -= 0.35
        else:
            # Winter heating peak (Dec-Feb)
            season = cfg["annual_amp"] * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)

        # 3. Weekly Cycle (Saturday lower, Sunday lowest)
        weekly = np.zeros(n_days)
        weekly[day_of_week == 5] = -cfg["weekly_amp"] * cfg["base"] * 0.65
        weekly[day_of_week == 6] = -cfg["weekly_amp"] * cfg["base"]

        # 4. COVID Dip Modeling (smooth Gaussian-like depression)
        covid_dip = np.zeros(n_days)
        start_ts = cfg["covid_start"]
        peak_ts = cfg["covid_peak"]
        end_ts = cfg["covid_end"]
        
        for i, dt in enumerate(dates):
            if start_ts <= dt <= end_ts:
                # Approximate smooth recovery
                total_duration = (end_ts - start_ts).days
                elapsed = (dt - start_ts).days
                peak_day = (peak_ts - start_ts).days
                if elapsed <= peak_day:
                    intensity = elapsed / max(1, peak_day)
                else:
                    intensity = 1.0 - (elapsed - peak_day) / max(1, (total_duration - peak_day))
                covid_dip[i] = -cfg["covid_drop"] * cfg["base"] * max(0.0, intensity)

        # 5. Stochastic noise
        noise = np.random.normal(0, cfg["noise_scale"], n_days)

        # Total emissions
        total = base_series + season + weekly + covid_dip + noise
        total = np.maximum(total, cfg["base"] * 0.4) # Physical lower bound

        # Sectoral distribution with sector-specific weekly variations
        ratios = cfg["sector_ratios"]
        for i in range(n_days):
            tot = total[i]
            # Weekends hit industry and transport slightly harder
            is_wknd = day_of_week[i] in (5, 6)
            
            p_share = ratios["power"] * (0.96 if is_wknd else 1.01)
            i_share = ratios["industry"] * (0.90 if is_wknd else 1.04)
            t_share = ratios["transport"] * (0.93 if is_wknd else 1.02)
            r_share = ratios["residential"] * (1.18 if is_wknd else 0.95)
            
            norm = p_share + i_share + t_share + r_share
            power = round(float(tot * (p_share / norm)), 3)
            industry = round(float(tot * (i_share / norm)), 3)
            transport = round(float(tot * (t_share / norm)), 3)
            # Ensure exact balance
            residential = round(float(tot - (power + industry + transport)), 3)

            records.append({
                "date": dates[i].strftime("%Y-%m-%d"),
                "region": region,
                "total": round(float(tot), 3),
                "power": power,
                "industry": industry,
                "transport": transport,
                "residential": residential
            })

    df_daily = pd.DataFrame(records)
    df_daily.to_csv(OUTPUT_DIR / "daily_emissions.csv", index=False)
    print(f"   Saved {len(df_daily)} rows to {OUTPUT_DIR / 'daily_emissions.csv'}")
    return df_daily


def generate_annual_indicators():
    """
    Generate annual macroeconomic and energy indicators (1990-2023) for 11 largest emitters.
    Columns: country, year, co2, gdp, population, electric_power_consumption,
             renewable_share, industrial_value_added
    """
    print("-> Generating annual_indicators.csv...")
    years = np.arange(1990, 2024)
    countries = [
        "China", "United States", "India", "Russia", "Japan",
        "Germany", "Iran", "South Korea", "Saudi Arabia", "Indonesia", "Canada"
    ]

    # Baseline calibration parameters (circa 1990 & 2023 trajectories)
    meta = {
        "China": {
            "co2_1990": 2420, "co2_2023": 11400,
            "gdp_1990": 360, "gdp_2023": 17800,
            "pop_1990": 1135, "pop_2023": 1410,
            "elec_1990": 510, "elec_2023": 5800,
            "ren_1990": 5.2, "ren_2023": 31.5,
            "ind_1990": 145, "ind_2023": 7100
        },
        "United States": {
            "co2_1990": 5040, "co2_2023": 4750,
            "gdp_1990": 5960, "gdp_2023": 26900,
            "pop_1990": 250, "pop_2023": 339,
            "elec_1990": 11800, "elec_2023": 12900,
            "ren_1990": 7.4, "ren_2023": 23.4,
            "ind_1990": 1380, "ind_2023": 4800
        },
        "India": {
            "co2_1990": 580, "co2_2023": 2850,
            "gdp_1990": 320, "gdp_2023": 3750,
            "pop_1990": 870, "pop_2023": 1428,
            "elec_1990": 270, "elec_2023": 1250,
            "ren_1990": 15.1, "ren_2023": 22.8,
            "ind_1990": 85, "ind_2023": 950
        },
        "Russia": {
            "co2_1990": 2180, "co2_2023": 1680,
            "gdp_1990": 516, "gdp_2023": 2020,
            "pop_1990": 148, "pop_2023": 144,
            "elec_1990": 6300, "elec_2023": 6800,
            "ren_1990": 14.2, "ren_2023": 19.5,
            "ind_1990": 180, "ind_2023": 620
        },
        "Japan": {
            "co2_1990": 1150, "co2_2023": 1020,
            "gdp_1990": 3130, "gdp_2023": 4210,
            "pop_1990": 123, "pop_2023": 124,
            "elec_1990": 6800, "elec_2023": 7900,
            "ren_1990": 9.8, "ren_2023": 22.1,
            "ind_1990": 980, "ind_2023": 1240
        },
        "Germany": {
            "co2_1990": 1040, "co2_2023": 615,
            "gdp_1990": 1770, "gdp_2023": 4450,
            "pop_1990": 79, "pop_2023": 84,
            "elec_1990": 6700, "elec_2023": 6200,
            "ren_1990": 3.6, "ren_2023": 45.8,
            "ind_1990": 520, "ind_2023": 1180
        },
        "Iran": {
            "co2_1990": 210, "co2_2023": 720,
            "gdp_1990": 120, "gdp_2023": 410,
            "pop_1990": 56, "pop_2023": 89,
            "elec_1990": 1100, "elec_2023": 3300,
            "ren_1990": 5.1, "ren_2023": 7.4,
            "ind_1990": 35, "ind_2023": 150
        },
        "South Korea": {
            "co2_1990": 250, "co2_2023": 595,
            "gdp_1990": 280, "gdp_2023": 1710,
            "pop_1990": 43, "pop_2023": 51,
            "elec_1990": 2600, "elec_2023": 10500,
            "ren_1990": 1.5, "ren_2023": 9.2,
            "ind_1990": 95, "ind_2023": 540
        },
        "Saudi Arabia": {
            "co2_1990": 180, "co2_2023": 660,
            "gdp_1990": 117, "gdp_2023": 1070,
            "pop_1990": 16, "pop_2023": 36,
            "elec_1990": 4200, "elec_2023": 9800,
            "ren_1990": 0.05, "ren_2023": 2.8,
            "ind_1990": 60, "ind_2023": 490
        },
        "Indonesia": {
            "co2_1990": 160, "co2_2023": 710,
            "gdp_1990": 106, "gdp_2023": 1370,
            "pop_1990": 181, "pop_2023": 277,
            "elec_1990": 180, "elec_2023": 1150,
            "ren_1990": 18.5, "ren_2023": 15.2,
            "ind_1990": 42, "ind_2023": 530
        },
        "Canada": {
            "co2_1990": 460, "co2_2023": 540,
            "gdp_1990": 590, "gdp_2023": 2140,
            "pop_1990": 27, "pop_2023": 40,
            "elec_1990": 15500, "elec_2023": 14200,
            "ren_1990": 62.0, "ren_2023": 68.5,
            "ind_1990": 150, "ind_2023": 510
        }
    }

    records = []
    n_years = len(years)

    for country in countries:
        c_meta = meta[country]
        progress = np.linspace(0, 1, n_years)

        # Realistic non-linear paths (S-curve or growth/peak)
        if country == "China":
            # WTO entry in 2001 accelerated emissions
            wto_factor = 1.0 / (1.0 + np.exp(-10 * (progress - 0.45)))
            co2_trend = c_meta["co2_1990"] + (c_meta["co2_2023"] - c_meta["co2_1990"]) * wto_factor
        elif country == "United States":
            # Peaked in ~2007 (progress ~0.5) then declined
            peak_profile = np.sin(progress * np.pi) * 800 - (progress * 1100)
            co2_trend = c_meta["co2_1990"] + peak_profile
        elif country == "Germany":
            # Steady continuous decline
            co2_trend = c_meta["co2_1990"] - (c_meta["co2_1990"] - c_meta["co2_2023"]) * (progress ** 0.85)
        elif country == "Russia":
            # 1990s collapse, recovery post-2000
            profile = -700 * np.sin(progress * np.pi * 0.7) + (progress * 200)
            co2_trend = c_meta["co2_1990"] + profile
        else:
            co2_trend = c_meta["co2_1990"] + (c_meta["co2_2023"] - c_meta["co2_1990"]) * (progress ** 1.15)

        for i, year in enumerate(years):
            p = progress[i]
            # Add macroeconomic variations + 2008 recession + 2020 COVID
            shock = 0.0
            if year == 2009:
                shock = -0.04
            elif year == 2020:
                shock = -0.06

            noise = np.random.normal(0, 0.015)
            multiplier = 1.0 + shock + noise

            co2_val = round(max(50.0, float(co2_trend[i] * multiplier)), 1)
            gdp_val = round(c_meta["gdp_1990"] * ((c_meta["gdp_2023"] / c_meta["gdp_1990"]) ** p) * (1.0 + shock * 0.7), 1)
            pop_val = round(c_meta["pop_1990"] + (c_meta["pop_2023"] - c_meta["pop_1990"]) * p, 1)
            elec_val = round(c_meta["elec_1990"] + (c_meta["elec_2023"] - c_meta["elec_1990"]) * (p ** 1.1), 1)
            ren_val = round(min(95.0, max(0.1, c_meta["ren_1990"] + (c_meta["ren_2023"] - c_meta["ren_1990"]) * (p ** 1.8))), 1)
            ind_val = round(c_meta["ind_1990"] * ((c_meta["ind_2023"] / c_meta["ind_1990"]) ** p) * (1.0 + shock * 0.9), 1)

            records.append({
                "country": country,
                "year": int(year),
                "co2": co2_val,
                "gdp": gdp_val,
                "population": pop_val,
                "electric_power_consumption": elec_val,
                "renewable_share": ren_val,
                "industrial_value_added": ind_val
            })

    df_annual = pd.DataFrame(records)
    df_annual.to_csv(OUTPUT_DIR / "annual_indicators.csv", index=False)
    print(f"   Saved {len(df_annual)} rows to {OUTPUT_DIR / 'annual_indicators.csv'}")
    return df_annual


def generate_ndc_targets():
    """
    Generate NDC targets for 2030 (placeholder target values for Phase 1 demo).
    Columns: country, target_year, target_emission, description
    """
    print("-> Generating ndc_targets.csv...")
    targets = [
        {"country": "China", "target_year": 2030, "target_emission": 11200.0, "description": "Placeholder: Peak CO2 emissions before 2030 & 65% carbon intensity reduction"},
        {"country": "United States", "target_year": 2030, "target_emission": 3150.0, "description": "Placeholder: 50-52% net GHG emissions reduction below 2005 levels"},
        {"country": "India", "target_year": 2030, "target_emission": 3100.0, "description": "Placeholder: 45% emissions intensity reduction below 2005 levels & 500 GW non-fossil capacity"},
        {"country": "EU27+UK", "target_year": 2030, "target_emission": 2150.0, "description": "Placeholder: Fit for 55 commitment (at least 55% net GHG reduction vs 1990)"},
        {"country": "Russia", "target_year": 2030, "target_emission": 1650.0, "description": "Placeholder: Limit GHG emissions to max 70% of 1990 baseline"},
        {"country": "Japan", "target_year": 2030, "target_emission": 750.0, "description": "Placeholder: 46% emissions reduction below 2013 fiscal year levels"},
        {"country": "Germany", "target_year": 2030, "target_emission": 440.0, "description": "Placeholder: Federal Climate Action Act target (65% reduction vs 1990)"},
        {"country": "Iran", "target_year": 2030, "target_emission": 820.0, "description": "Placeholder: Unconditional 4% GHG reduction relative to business-as-usual"},
        {"country": "South Korea", "target_year": 2030, "target_emission": 430.0, "description": "Placeholder: Enhanced NDC target of 40% reduction below 2018 peak"},
        {"country": "Saudi Arabia", "target_year": 2030, "target_emission": 620.0, "description": "Placeholder: Circular Carbon Economy NDC reduction of 278 MtCO2eq/year"},
        {"country": "Indonesia", "target_year": 2030, "target_emission": 980.0, "description": "Placeholder: Unconditional 31.89% reduction vs baseline projection"},
        {"country": "Canada", "target_year": 2030, "target_emission": 440.0, "description": "Placeholder: 40-45% reduction below 2005 levels"}
    ]
    df_ndc = pd.DataFrame(targets)
    df_ndc.to_csv(OUTPUT_DIR / "ndc_targets.csv", index=False)
    print(f"   Saved {len(df_ndc)} rows to {OUTPUT_DIR / 'ndc_targets.csv'}")
    return df_ndc


def generate_adf_results():
    """
    Generate ADF (Augmented Dickey-Fuller) stationarity test results.
    Shows statistical status before and after first-order differencing.
    Columns: series_name, adf_statistic, p_value, stationary, differencing_applied
    """
    print("-> Generating adf_results.csv...")
    rows = [
        # Daily Raw Series (Non-Stationary due to trends and strong seasonality)
        {"series_name": "Daily CO2 (China Total)", "adf_statistic": -1.824, "p_value": 0.368, "stationary": "No", "differencing_applied": "None (Raw)"},
        {"series_name": "Daily CO2 (India Total)", "adf_statistic": -0.912, "p_value": 0.784, "stationary": "No", "differencing_applied": "None (Raw)"},
        {"series_name": "Daily CO2 (USA Total)", "adf_statistic": -2.145, "p_value": 0.227, "stationary": "No", "differencing_applied": "None (Raw)"},
        {"series_name": "Daily CO2 (EU27+UK Total)", "adf_statistic": -2.311, "p_value": 0.168, "stationary": "No", "differencing_applied": "None (Raw)"},

        # Daily Differenced Series (Stationary: p < 0.01)
        {"series_name": "Daily CO2 (China Total)", "adf_statistic": -24.891, "p_value": 0.0001, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"},
        {"series_name": "Daily CO2 (India Total)", "adf_statistic": -28.345, "p_value": 0.0001, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"},
        {"series_name": "Daily CO2 (USA Total)", "adf_statistic": -26.118, "p_value": 0.0001, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"},
        {"series_name": "Daily CO2 (EU27+UK Total)", "adf_statistic": -25.742, "p_value": 0.0001, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"},

        # Annual Raw Series (Non-Stationary due to multi-decade economic trajectory)
        {"series_name": "Annual CO2 (China)", "adf_statistic": -0.412, "p_value": 0.908, "stationary": "No", "differencing_applied": "None (Raw)"},
        {"series_name": "Annual CO2 (USA)", "adf_statistic": -1.654, "p_value": 0.455, "stationary": "No", "differencing_applied": "None (Raw)"},
        {"series_name": "Annual CO2 (India)", "adf_statistic": 0.285, "p_value": 0.976, "stationary": "No", "differencing_applied": "None (Raw)"},
        {"series_name": "Annual CO2 (Germany)", "adf_statistic": -1.321, "p_value": 0.621, "stationary": "No", "differencing_applied": "None (Raw)"},

        # Annual Differenced Series (Stationary after d=1)
        {"series_name": "Annual CO2 (China)", "adf_statistic": -4.218, "p_value": 0.0007, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"},
        {"series_name": "Annual CO2 (USA)", "adf_statistic": -5.112, "p_value": 0.0001, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"},
        {"series_name": "Annual CO2 (India)", "adf_statistic": -3.985, "p_value": 0.0015, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"},
        {"series_name": "Annual CO2 (Germany)", "adf_statistic": -4.872, "p_value": 0.0002, "stationary": "Yes", "differencing_applied": "1st Difference (d=1)"}
    ]
    df_adf = pd.DataFrame(rows)
    df_adf.to_csv(OUTPUT_DIR / "adf_results.csv", index=False)
    print(f"   Saved {len(df_adf)} rows to {OUTPUT_DIR / 'adf_results.csv'}")
    return df_adf


def generate_selection_results():
    """
    Generate Granger causality lag selection and feature selection results.
    Organized with a category column so the page can present:
    1. Granger causality selected lags per macroeconomic & energy driver
    2. Selected features per country with importance score
    3. With vs Without feature selection comparison metrics
    """
    print("-> Generating selection.csv...")
    rows = []

    # 1. Granger Causality Lags per Driver
    granger_drivers = [
        {"driver": "Electric Power Consumption (kWh/capita)", "optimal_lag": "1 year", "f_stat": 18.42, "p_val": 0.0002, "causality": "Confirmed (X -> CO2)"},
        {"driver": "Industrial Value Added (Billion USD)", "optimal_lag": "1 year", "f_stat": 14.85, "p_val": 0.0006, "causality": "Confirmed (X -> CO2)"},
        {"driver": "GDP (Billion USD)", "optimal_lag": "2 years", "f_stat": 11.34, "p_val": 0.0018, "causality": "Confirmed (X -> CO2)"},
        {"driver": "Renewable Share (%)", "optimal_lag": "2 years", "f_stat": 9.76, "p_val": 0.0041, "causality": "Confirmed (Negative X -> CO2)"},
        {"driver": "Population (Million)", "optimal_lag": "3 years", "f_stat": 4.12, "p_val": 0.0480, "causality": "Moderate (Long-term drift)"},
        {"driver": "Daily Power Sector Lag (Daily)", "optimal_lag": "7 days", "f_stat": 42.15, "p_val": 0.0000, "causality": "Confirmed (Weekly cycle)"},
        {"driver": "Daily Transport Sector Lag (Daily)", "optimal_lag": "1 day", "f_stat": 31.80, "p_val": 0.0000, "causality": "Confirmed (Immediate)"}
    ]
    for item in granger_drivers:
        rows.append({
            "category": "granger_lag",
            "entity": "Global / Multi-country",
            "driver_or_feature": item["driver"],
            "optimal_lag": item["optimal_lag"],
            "stat_or_score": item["f_stat"],
            "p_value": item["p_val"],
            "metric_all": np.nan,
            "metric_selected": np.nan,
            "improvement_pct": np.nan,
            "notes": item["causality"]
        })

    # 2. Selected Features per Country (Top Ranked by LASSO & Random Forest Gini Importance)
    top_features_per_country = {
        "China": [
            ("Industrial Value Added (t-1)", 0.38, "Top driver of production emissions"),
            ("Electric Power Consumption (t-1)", 0.31, "Heavy coal grid dependence"),
            ("GDP (t-2)", 0.18, "Macro demand signal"),
            ("Renewable Share (t-2)", 0.13, "Mitigating factor post-2015")
        ],
        "United States": [
            ("Electric Power Consumption (t-1)", 0.35, "Grid electricity & cooling"),
            ("GDP (t-1)", 0.26, "Service & consumer consumption"),
            ("Renewable Share (t-1)", 0.24, "Coal-to-gas & renewables shift"),
            ("Industrial Value Added (t-2)", 0.15, "Moderate manufacturing weight")
        ],
        "India": [
            ("Electric Power Consumption (t-1)", 0.42, "Rapid electrification & industrial demand"),
            ("GDP (t-2)", 0.28, "Strong economic development proxy"),
            ("Industrial Value Added (t-1)", 0.20, "Cement, steel, manufacturing"),
            ("Population (t-3)", 0.10, "Demographic expansion baseline")
        ],
        "EU27+UK": [
            ("Renewable Share (t-1)", 0.39, "Strongest driver of emissions abatement"),
            ("GDP (t-1)", 0.24, "Decoupled economic growth"),
            ("Industrial Value Added (t-2)", 0.22, "Efficiency gains in heavy sector"),
            ("Electric Power Consumption (t-1)", 0.15, "Electrification with cleaner grid")
        ]
    }
    for country, f_list in top_features_per_country.items():
        for feat, score, note in f_list:
            rows.append({
                "category": "selected_features",
                "entity": country,
                "driver_or_feature": feat,
                "optimal_lag": "Optimal",
                "stat_or_score": score,
                "p_value": np.nan,
                "metric_all": np.nan,
                "metric_selected": np.nan,
                "improvement_pct": np.nan,
                "notes": note
            })

    # 3. Model Performance Comparison: With Selection vs Without Selection (All Raw Features)
    comparison_bench = [
        {"entity": "China", "driver_or_feature": "XGBoost", "metric_all": 142.5, "metric_selected": 118.2, "improvement_pct": 17.1, "notes": "RMSE in MtCO2/yr. Selection removes multicollinearity."},
        {"entity": "China", "driver_or_feature": "Random Forest", "metric_all": 158.4, "metric_selected": 131.0, "improvement_pct": 17.3, "notes": "RMSE in MtCO2/yr."},
        {"entity": "United States", "driver_or_feature": "XGBoost", "metric_all": 98.4, "metric_selected": 82.1, "improvement_pct": 16.6, "notes": "RMSE in MtCO2/yr."},
        {"entity": "United States", "driver_or_feature": "Random Forest", "metric_all": 108.2, "metric_selected": 91.5, "improvement_pct": 15.4, "notes": "RMSE in MtCO2/yr."},
        {"entity": "India", "driver_or_feature": "XGBoost", "metric_all": 62.8, "metric_selected": 49.3, "improvement_pct": 21.5, "notes": "RMSE in MtCO2/yr. Strong noise suppression."},
        {"entity": "India", "driver_or_feature": "Random Forest", "metric_all": 69.1, "metric_selected": 55.4, "improvement_pct": 19.8, "notes": "RMSE in MtCO2/yr."},
        {"entity": "EU27+UK", "driver_or_feature": "XGBoost", "metric_all": 54.2, "metric_selected": 44.8, "improvement_pct": 17.3, "notes": "RMSE in MtCO2/yr."},
        {"entity": "EU27+UK", "driver_or_feature": "Random Forest", "metric_all": 59.5, "metric_selected": 49.6, "improvement_pct": 16.6, "notes": "RMSE in MtCO2/yr."}
    ]
    for comp in comparison_bench:
        rows.append({
            "category": "selection_comparison",
            "entity": comp["entity"],
            "driver_or_feature": comp["driver_or_feature"],
            "optimal_lag": "N/A",
            "stat_or_score": np.nan,
            "p_value": np.nan,
            "metric_all": comp["metric_all"],
            "metric_selected": comp["metric_selected"],
            "improvement_pct": comp["improvement_pct"],
            "notes": comp["notes"]
        })

    df_selection = pd.DataFrame(rows)
    df_selection.to_csv(OUTPUT_DIR / "selection.csv", index=False)
    print(f"   Saved {len(df_selection)} rows to {OUTPUT_DIR / 'selection.csv'}")
    return df_selection


def generate_baseline_results():
    """
    Generate baseline model benchmark metrics.
    Models: ARIMA, SARIMA, SVR, Random Forest, Gradient Boosting, XGBoost, Voting, Bagging.
    Columns: model, scale, region_or_country, MAE, RMSE, MAPE, R2, training_time_sec.
    Scale: daily and annual.
    Plausible results:
    - Statistical models (ARIMA/SARIMA) struggle with daily multi-seasonality and noise.
    - SVR and tree boosting show strong accuracy on annual data and competitive speed.
    - Values are realistic, NOT perfect (R2 ~0.72-0.95, MAPE ~3-14%).
    """
    print("-> Generating baseline_results.csv...")
    models = [
        "ARIMA",
        "SARIMA",
        "SVR",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "Voting",
        "Bagging"
    ]

    daily_regions = ["China", "India", "USA", "EU27+UK"]
    annual_countries = [
        "China", "United States", "India", "Russia", "Japan",
        "Germany", "Iran", "South Korea", "Saudi Arabia", "Indonesia", "Canada"
    ]

    records = []

    # 1. Daily Scale Benchmarks (Scale: daily)
    # Unit of MAE/RMSE: MtCO2/day
    daily_model_factors = {
        "ARIMA": {"mae_f": 1.45, "rmse_f": 1.95, "mape": 9.2, "r2": 0.74, "time": 16.4},
        "SARIMA": {"mae_f": 1.18, "rmse_f": 1.58, "mape": 7.4, "r2": 0.82, "time": 45.2},
        "SVR": {"mae_f": 0.98, "rmse_f": 1.34, "mape": 6.1, "r2": 0.86, "time": 2.8},
        "Random Forest": {"mae_f": 0.81, "rmse_f": 1.12, "mape": 4.9, "r2": 0.89, "time": 7.1},
        "Gradient Boosting": {"mae_f": 0.72, "rmse_f": 0.98, "mape": 4.3, "r2": 0.91, "time": 4.6},
        "XGBoost": {"mae_f": 0.62, "rmse_f": 0.85, "mape": 3.7, "r2": 0.93, "time": 3.1},
        "Voting": {"mae_f": 0.56, "rmse_f": 0.78, "mape": 3.4, "r2": 0.94, "time": 14.8},
        "Bagging": {"mae_f": 0.85, "rmse_f": 1.18, "mape": 5.2, "r2": 0.88, "time": 6.4}
    }

    region_multipliers = {
        "China": 2.2,    # Large absolute magnitude ~30 MtCO2/day
        "USA": 1.1,      # Medium magnitude ~14 MtCO2/day
        "EU27+UK": 0.7,  # Smaller magnitude ~8 MtCO2/day
        "India": 0.65    # Smaller magnitude ~7 MtCO2/day
    }

    for region in daily_regions:
        rm = region_multipliers[region]
        for model in models:
            mf = daily_model_factors[model]
            # Add minor region-specific realistic variation
            jitter = np.random.uniform(-0.02, 0.02)
            mae = round(float(mf["mae_f"] * rm * (1 + jitter)), 3)
            rmse = round(float(mf["rmse_f"] * rm * (1 + jitter)), 3)
            mape = round(float(mf["mape"] + jitter * 8.0), 2)
            r2 = round(float(min(0.96, max(0.68, mf["r2"] + jitter))), 3)
            train_time = round(float(mf["time"] * np.random.uniform(0.92, 1.08)), 2)

            records.append({
                "model": model,
                "scale": "daily",
                "region_or_country": region,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
                "training_time_sec": train_time
            })

    # 2. Annual Scale Benchmarks (Scale: annual)
    # Unit of MAE/RMSE: MtCO2/year
    annual_model_factors = {
        "ARIMA": {"mae_pct": 0.065, "rmse_pct": 0.082, "mape": 6.8, "r2": 0.83, "time": 1.2},
        "SARIMA": {"mae_pct": 0.061, "rmse_pct": 0.078, "mape": 6.4, "r2": 0.85, "time": 2.6},
        "SVR": {"mae_pct": 0.039, "rmse_pct": 0.049, "mape": 4.1, "r2": 0.91, "time": 0.4},
        "Random Forest": {"mae_pct": 0.043, "rmse_pct": 0.054, "mape": 4.5, "r2": 0.89, "time": 1.1},
        "Gradient Boosting": {"mae_pct": 0.036, "rmse_pct": 0.046, "mape": 3.8, "r2": 0.92, "time": 0.8},
        "XGBoost": {"mae_pct": 0.033, "rmse_pct": 0.042, "mape": 3.5, "r2": 0.93, "time": 0.6},
        "Voting": {"mae_pct": 0.030, "rmse_pct": 0.038, "mape": 3.1, "r2": 0.94, "time": 2.1},
        "Bagging": {"mae_pct": 0.045, "rmse_pct": 0.057, "mape": 4.7, "r2": 0.88, "time": 0.9}
    }

    country_scales = {
        "China": 10000.0, "United States": 5000.0, "India": 2400.0, "Russia": 1600.0,
        "Japan": 1100.0, "Germany": 750.0, "Iran": 650.0, "South Korea": 600.0,
        "Saudi Arabia": 580.0, "Indonesia": 620.0, "Canada": 520.0
    }

    for country in annual_countries:
        base_val = country_scales[country]
        for model in models:
            mf = annual_model_factors[model]
            jitter = np.random.uniform(-0.02, 0.02)
            mae = round(float(base_val * mf["mae_pct"] * (1 + jitter)), 1)
            rmse = round(float(base_val * mf["rmse_pct"] * (1 + jitter)), 1)
            mape = round(float(mf["mape"] + jitter * 6.0), 2)
            r2 = round(float(min(0.965, max(0.72, mf["r2"] + jitter))), 3)
            train_time = round(float(mf["time"] * np.random.uniform(0.9, 1.1)), 2)

            records.append({
                "model": model,
                "scale": "annual",
                "region_or_country": country,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
                "training_time_sec": train_time
            })

    df_baseline = pd.DataFrame(records)
    df_baseline.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    print(f"   Saved {len(df_baseline)} rows to {OUTPUT_DIR / 'baseline_results.csv'}")
    return df_baseline


def generate_predictions(df_daily):
    """
    Generate test period predictions for 2024-01-01 to 2025-12-31 (731 days) for each region.
    Columns: date, region, actual, ARIMA, SARIMA, SVR, Random_Forest,
             Gradient_Boosting, XGBoost, Voting, Bagging
    Produces model-specific behaviors:
    - ARIMA: smoother, minor phase lag on weekly peaks
    - SARIMA: captures weekly cycle well, struggles with extreme spikes
    - SVR: smooth non-linear fit
    - Tree models (RF, GB, XGBoost): accurate capture of sharp weekend drops
    - Voting: ensemble average with lowest variance and robust alignment
    """
    print("-> Generating predictions.csv...")
    test_dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    
    # Filter daily emissions to test period
    df_test_actual = df_daily[df_daily["date"].isin(test_dates.strftime("%Y-%m-%d"))].copy()

    records = []
    models = ["ARIMA", "SARIMA", "SVR", "Random_Forest", "Gradient_Boosting", "XGBoost", "Voting", "Bagging"]

    for region in ["China", "India", "USA", "EU27+UK"]:
        sub = df_test_actual[df_test_actual["region"] == region].sort_values("date").reset_index(drop=True)
        actuals = sub["total"].values
        n = len(actuals)

        # Baseline predictions generation with realistic errors
        # 1. ARIMA (Smoothed with moving average lag)
        arima_pred = pd.Series(actuals).rolling(window=4, min_periods=1).mean().values
        arima_pred = arima_pred * 0.985 + np.random.normal(0, actuals.mean() * 0.055, n)

        # 2. SARIMA (Captures 7-day pattern with some residual noise)
        sarima_pred = np.zeros(n)
        for i in range(n):
            lag7 = actuals[i - 7] if i >= 7 else actuals[i]
            sarima_pred[i] = 0.6 * actuals[i] + 0.38 * lag7 + np.random.normal(0, actuals.mean() * 0.04)

        # 3. SVR (Kernel regression behavior)
        svr_pred = actuals * 0.99 + np.random.normal(0, actuals.mean() * 0.038, n)

        # 4. Random Forest
        rf_pred = actuals * 0.995 + np.random.normal(0, actuals.mean() * 0.032, n)

        # 5. Gradient Boosting
        gb_pred = actuals * 0.998 + np.random.normal(0, actuals.mean() * 0.027, n)

        # 6. XGBoost
        xgb_pred = actuals * 0.999 + np.random.normal(0, actuals.mean() * 0.022, n)

        # 7. Bagging
        bagging_pred = actuals * 0.994 + np.random.normal(0, actuals.mean() * 0.034, n)

        # 8. Voting (Ensemble mean of top models)
        voting_pred = (xgb_pred * 0.35 + gb_pred * 0.30 + rf_pred * 0.20 + svr_pred * 0.15)

        for i in range(n):
            records.append({
                "date": sub["date"].iloc[i],
                "region": region,
                "actual": round(float(actuals[i]), 3),
                "ARIMA": round(float(arima_pred[i]), 3),
                "SARIMA": round(float(sarima_pred[i]), 3),
                "SVR": round(float(svr_pred[i]), 3),
                "Random_Forest": round(float(rf_pred[i]), 3),
                "Gradient_Boosting": round(float(gb_pred[i]), 3),
                "XGBoost": round(float(xgb_pred[i]), 3),
                "Voting": round(float(voting_pred[i]), 3),
                "Bagging": round(float(bagging_pred[i]), 3)
            })

    df_preds = pd.DataFrame(records)
    df_preds.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"   Saved {len(df_preds)} rows to {OUTPUT_DIR / 'predictions.csv'}")
    return df_preds


def main():
    print(f"============================================================")
    print(f"CarbonLens: Generating Prototype Datasets (Seed = {SEED})")
    print(f"Target Directory: {OUTPUT_DIR}")
    print(f"============================================================")

    df_daily = generate_daily_emissions()
    generate_annual_indicators()
    generate_ndc_targets()
    generate_adf_results()
    generate_selection_results()
    generate_baseline_results()
    generate_predictions(df_daily)

    print(f"============================================================")
    print("SUCCESS: All 7 prototype datasets generated successfully!")
    print(f"============================================================")


if __name__ == "__main__":
    main()
