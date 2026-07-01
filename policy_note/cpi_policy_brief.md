# Supply vs. Demand Drivers of India's CPI Inflation (2015–2026): Implications for Monetary Policy Transmission

**Author:** Quantitative Monetary Policy Research Unit  
**Date:** July 1, 2026  

---

### Executive Summary
This policy brief evaluates the structural drivers of India’s Consumer Price Index (CPI) inflation and their historical interaction with the Reserve Bank of India’s (RBI) Monetary Policy Committee (MPC) repo rate decisions from 2015 to 2026. By isolating supply-side shocks (food and fuel) from demand-side pressures (core CPI) through STL decomposition, Granger causality, and explainable machine learning models (XGBoost + SHAP), we show that while the MPC's decision rule is structurally anchored on core inflation trends, 62.5% of historical rate hikes were executed during supply-dominated inflation spikes. This paper outlines the transmission risks of cost-push inflation responses and presents a 6-month forecasting outlook.

---

### 1. Context & Policy Problem
Monetary policy transmission in India faces a structural challenge: the CPI basket is heavily weighted toward food and beverages (45.86%) and fuel and light (6.84%), which are highly volatile and driven by supply-side shocks (e.g., monsoon variability, geopolitical energy disruptions). Traditional interest rate hikes are designed to cool aggregate demand (reflected in Core CPI). When rate hikes are deployed in response to headline inflation spikes driven purely by supply shocks, they impose severe economic growth costs (by raising borrowing costs for firms and consumers) without cooling the structural drivers of the price shock. Distinguishing between demand-driven and cost-push inflation is therefore the cornerstone of effective monetary policy.

---

### 2. Dataset & Methodology
* **Inflation Data:** Monthly CPI component data from the Database on Indian Economy (DBIE) and Ministry of Statistics and Programme Implementation (MOSPI) spanning January 2013 to December 2025. 
* **Policy Data:** Full history of the RBI MPC meetings and repo rate decisions (60 meetings) from October 2016 to June 2026.
* **Methodology:** 
  1. We computed Core CPI using official base-2012 weights ($\text{Core} = \text{General} - [0.4563 \times \text{Food} + 0.0666 \times \text{Fuel}]$).
  2. We conducted Seasonal-Trend decomposition using Loess (STL) on Core CPI.
  3. We ran Granger Causality tests on monthly-aligned series to verify the directional relationship.
  4. We trained an XGBoost classifier with sequential cross-validation and applied SHAP explainability to deconstruct the MPC's historical decision-making process.
  5. We fit a Prophet additive time series model to forecast the 6-month forward core inflation path.

---

### 3. Key Findings

#### Finding 1: Historical Inflation Epoc Heatmap Analysis
Analysis of the monthly inflation heatmap (*outputs/01_core_cpi_heatmap.png*) reveals clear cyclical patterns:
* **The Post-COVID Supply Shock (2022–2023):** Core CPI YoY inflation remained persistently elevated above the RBI's 6% upper tolerance limit, driven by global supply chain gridlocks and commodity price spikes following the outbreak of the Russia-Ukraine war.
* **The Food Spike (2019–2020):** General inflation spiked sharply due to domestic crop failures, while Core CPI remained relatively anchored (hovering between 4.0% and 5.0%), demonstrating a clear divergence between headline and core vectors.

#### Finding 2: Rate Hike Misattribution (Supply-Shock Context)
Our misattribution analysis of the 8 historical rate hikes executed by the MPC since October 2016 reveals that:
* **5 out of 8 hikes (62.5%)** were executed during periods where **Food inflation exceeded Core inflation** (e.g., May, June, August, and September of 2022, and February 2023). 
* **Implication:** Tightening monetary policy in these cycles coincided heavily with supply-side food shocks. This creates a communication risk, as markets may interpret hikes as a direct response to transient food price spikes rather than persistent demand-side core trends.

#### Finding 3: XGBoost Classifier & SHAP Explainability
The final XGBoost model trained on inflation indicators to predict rate hikes yielded a robust feature hierarchy. The mean absolute SHAP values (*outputs/06_shap_feature_importance.png*) show:
1. **Core Inflation Lag 1 (`core_lag1`):** SHAP value $= 0.0514$ (Highest importance)
2. **Current Core Inflation (`cpi_core_yoy`):** SHAP value $= 0.0495$ (Second highest)
3. **Core 3-Month Average (`core_3m_avg`):** SHAP value $= 0.0129$
4. **Headline General Inflation (`cpi_general_yoy`):** SHAP value $= 0.0031$
5. **Food Inflation (`cpi_food_yoy`):** SHAP value $= 0.0000$

* **Interpretation:** The MPC's empirical reaction function is highly disciplined. Hikes are statistically driven by the level and immediate lag of **Core Inflation**, not by food or headline spikes. This is further validated by our Granger Causality test, which shows past Core CPI YoY Granger-causes repo rate changes at Lag 1 ($p = 0.0003$) and Lag 2 ($p = 0.0030$). 

---

### 4. Inflation Forecasting Outlook (Jan–June 2026)
The Prophet time series model (*outputs/04_prophet_forecast.png*) predicts that Core CPI inflation will remain stable and well-anchored over the first half of 2026:
* **Forecasted Core CPI:** Projected to hover in a tight range between **4.15% (June 2026)** and **4.37% (February 2026)**.
* **Uncertainty Bounds:** The 90% confidence intervals range from a lower bound of $2.61\%$ to an upper bound of $5.95\%$.
* **Policy Implications:** With core inflation projected to remain close to the 4% target and comfortably below the 6% upper limit, the MPC has substantial policy room to maintain a stable or accommodative rate stance without risking demand-side overheating.

```
Prophet Forecasted Core CPI YoY (%):
- Jan 2026: 4.31% (90% CI: 2.71% - 5.92%)
- Feb 2026: 4.37% (90% CI: 2.75% - 5.95%)
- Mar 2026: 4.36% (90% CI: 2.78% - 5.92%)
- Apr 2026: 4.33% (90% CI: 2.66% - 5.87%)
- May 2026: 4.31% (90% CI: 2.66% - 5.84%)
- Jun 2026: 4.15% (90% CI: 2.61% - 5.68%)
```

---

### 5. Policy Recommendations
1. **Explicit Decomposed Inflation Communication:** The MPC should explicitly separate headline inflation in its policy statements into its supply-side (transient food/fuel) and demand-side (core trend) components. Explicitly communicating that "hikes are deployed to anchor core inflation trends, not in reaction to temporary vegetable price shocks" will help anchor market expectations and reduce transmission lags.
2. **Core Inflation Target Anchor:** While the primary legal mandate is Headline CPI (4% $\pm$ 2%), the MPC should continue to treat the STL-decomposed Core trend as its primary empirical decision anchor to prevent growth-destabilizing policy errors during cost-push shocks.
3. **Exploiting Policy Space in 2026:** Given that the Prophet model projects core inflation to stabilize near the 4.15%–4.37% band, the MPC should utilize this policy window to support growth recovery, maintaining a pause on rate hikes until global demand signals shift.

---

### 6. Limitations of the Analysis
1. **Small Sample Size:** The dataset contains only 60 MPC meetings since 2016, which limits the predictive power of the XGBoost classifier and makes it prone to overfitting. It should be treated as a historical pattern classifier.
2. **Fixed CPI Weights:** The analysis relies on 2012 base-year weights. Since consumption patterns have evolved, these weights may overstate the budget share of food in contemporary households.
3. **Absence of Output Gap Proxy:** The feature matrix lacks a direct quarterly GDP or output gap proxy, which restricts our ability to model real-economy demand-pull dynamics.
