# India CPI Inflation Decomposition & MPC Policy Signal Detector

An end-to-end econometric and machine learning pipeline to decompose Consumer Price Index (CPI) inflation dynamics in India (2015–2026) and analyze the Reserve Bank of India (RBI) Monetary Policy Committee (MPC) interest rate decision rules.

---

## 📌 Project Overview
Monetary policy transmission is highly sensitive to the *source* of inflation. Rate hikes are effective against demand-driven (core) inflation but impose growth costs without cooling supply-side shocks (like food or fuel spikes). 

This project built a reproducible pipeline to:
1. **Decompose CPI** into transient supply-side shocks (food/fuel) and persistent demand-side vectors (core) using Loess-based Seasonal-Trend decomposition (STL).
2. **Model Causal Transmission** using Augmented Dickey-Fuller (ADF) stationarity and Granger Causality tests between core inflation lags and policy interest rates.
3. **Analyze MPC Decisions** using an XGBoost Classifier with `TimeSeriesSplit` cross-validation to predict rate hikes, explained globally and locally with SHAP value Feature Importances.
4. **Forecast Future Inflation Paths** 6 months out using a Prophet additive time series model.
5. **Evaluate Policy Misattribution** by calculating how often rate hikes occurred during supply-shock (food-dominated) spikes.

---

## 📂 Repository Structure

```
rbi_cpi_project/
├── data/
│   ├── cleaned_cpi.csv          # Clean monthly CPI series with computed core & trend
│   └── processed_cpi_mpc.csv    # Merged dataset containing CPI inflation matched to MPC meetings
├── raw_data/
│   ├── CPI-Base-2012.xlsx       # Raw commodity-wise CPI index sheets (Base 2012=100)
│   └── mpc_decisions.xlsx       # Raw RBI voting and repo rate announcement records
├── notebooks/
│   ├── 01_data_prep.ipynb       # Loading raw Excel files, cleaning, merging and lag calculations
│   ├── 02_eda.ipynb             # Heatmaps, inflation component timelines, and correlations
│   ├── 03_decomposition.ipynb   # STL decomposition, ADF tests, Granger causality, and misattribution
│   └── 04_ai_model.ipynb        # Prophet forecasting, XGBoost rate hike classifier, and SHAP plots
├── outputs/
│   ├── 01_core_cpi_heatmap.png  # Month-Year core inflation trend heatmap
│   ├── 02_cpi_vs_repo_rate.png  # Timeline of CPI components vs Repo Rate hikes/cuts
│   ├── 03_stl_decomposition.png # STL decomposition panel (Observed, Trend, Seasonal, Residual)
│   └── yearly_cpi_components.csv# Annual averages table
└── policy_note/
    └── cpi_policy_brief.md      # A 1-page professional monetary policy advice brief
```

---

## 🚀 Setup & Installation

### Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet xgboost shap openpyxl
```

### Execution Flow
Run the notebooks in the `notebooks/` folder sequentially:
1. **[01_data_prep.ipynb](file:///notebooks/01_data_prep.ipynb)**: Cleans raw RBI files and creates the datasets.
2. **[02_eda.ipynb](file:///notebooks/02_eda.ipynb)**: Visualizes historical trends and exports correlation structures.
3. **[03_decomposition.ipynb](file:///notebooks/03_decomposition.ipynb)**: Performs the econometric time series analysis.
4. **[04_ai_model.ipynb](file:///notebooks/04_ai_model.ipynb)**: Runs the ML forecasting and classification models.

---

## 📈 Key Quantitative Findings

* **Rate Hike Misattribution (62.5%):** Out of the 8 rate hikes executed by the MPC since October 2016, 5 hikes (62.5%) occurred during environments where food inflation exceeded core inflation. This suggests a tight historical correlation between policy hikes and transient cost-push supply shocks.
* **Empirical Decision Rules (SHAP):** The XGBoost classification model shows that the MPC's rate hike decisions are statistically driven by **Core Inflation Lag 1 (`core_lag1`)** and **current Core Inflation (`cpi_core_yoy`)**, with mean absolute SHAP importances of `0.0514` and `0.0495` respectively. Volatile food and headline general inflation have very low directly predictive weight.
* **Granger Causality:** Granger causality tests confirm a highly significant directional relationship, where past Core CPI YoY inflation Granger-causes Repo Rate changes at Lag 1 ($p = 0.0003$) and Lag 2 ($p = 0.0030$).
* **Core Inflation Outlook (Jan–June 2026):** The Prophet forecasting model projects Core CPI to remain well-anchored, stabilizing between **4.15% and 4.37%** (well within the RBI's 2.0%–6.0% tolerance band), offering policy room to support post-COVID growth recovery.

---

## 🏛️ Policy Recommendation Highlights
Detailed recommendations can be found in the [cpi_policy_brief.md](file:///policy_note/cpi_policy_brief.md):
1. **Explicit Decomposed Inflation Communication:** The MPC should separate headline inflation in policy announcements to explicitly state that hikes target demand-side core trends, not transient supply food shocks. This minimizes market transmission lag.
2. **Core Inflation Anchoring:** Maintain core inflation as the primary empirical decision anchor to avoid growth contraction during supply shocks.
3. **Supportive Policy Stance:** Utilize the projected low-inflation window in 2026 to maintain a rate pause and support structural recovery.
