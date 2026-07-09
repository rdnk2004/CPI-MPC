"""
Stage 3: Statistical Decomposition & Causal Policy Analysis.

1. STL decomposition of Core CPI YoY (trend / seasonal / residual).
2. ADF stationarity tests on Core CPI, General CPI, and the Repo Rate.
3. Granger causality: does Core CPI predict Repo Rate changes?
   IMPORTANT: all three series are non-stationary (verified below), so the
   test is run on *differenced* series, not raw levels. An earlier version of
   this analysis ran Granger causality directly on the non-stationary levels
   and found a "highly significant" result (p=0.0003 at lag 1) -- that result
   does not survive differencing (p=0.53 at lag 1) and was a false positive
   driven by both series sharing a long-term trend, not real predictive
   causality. See README changelog for details.
4. Supply vs. demand misattribution: what fraction of rate hikes happened
   while food inflation (supply-side) exceeded core inflation (demand-side).

Run standalone:
    python -m src.decomposition
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ADF_SIGNIFICANCE = 0.05
GRANGER_MAX_LAG = 6


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_cpi = pd.read_csv(config.CLEANED_CPI_FILE)
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
    df_mpc = pd.read_csv(config.PROCESSED_CPI_MPC_FILE)
    df_mpc["date"] = pd.to_datetime(df_mpc["date"])
    return df_cpi, df_mpc


def run_stl_decomposition(df_cpi: pd.DataFrame, out_path=None) -> pd.DataFrame:
    """Decompose Core CPI YoY into trend/seasonal/residual, plot, and persist
    the trend component back onto the cleaned CPI dataset."""
    out_path = out_path or config.OUTPUTS_DIR / "03_stl_decomposition.png"

    core_series = df_cpi.set_index("date")["cpi_core_yoy"].dropna()
    stl = STL(core_series, period=12, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(core_series, color="#2c3e50", linewidth=1.8)
    axes[0].set_ylabel("Observed (YoY %)")
    axes[0].set_title("STL Decomposition of India Core CPI Inflation", fontsize=12)

    axes[1].plot(result.trend, color="#e74c3c", linewidth=1.8)
    axes[1].set_ylabel("Trend (YoY %)")

    axes[2].plot(result.seasonal, color="#27ae60", linewidth=1.8)
    axes[2].set_ylabel("Seasonal (YoY %)")

    axes[3].plot(result.resid, color="#95a5a6", linewidth=1.5)
    axes[3].set_ylabel("Residual (YoY %)")
    axes[3].axhline(0, color="black", linewidth=0.8, linestyle=":")

    for ax in axes:
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)

    df_cpi = df_cpi.copy()
    df_cpi["core_trend"] = np.nan
    df_cpi.loc[df_cpi["date"].isin(core_series.index), "core_trend"] = result.trend.values
    df_cpi.to_csv(config.CLEANED_CPI_FILE, index=False)
    logger.info("core_trend column saved back to %s", config.CLEANED_CPI_FILE)
    return df_cpi


def check_stationarity(series: pd.Series, name: str) -> float:
    """Run ADF test, log a clear stationary/non-stationary verdict, return p-value."""
    p_value = adfuller(series.dropna())[1]
    verdict = "stationary" if p_value < ADF_SIGNIFICANCE else "non-stationary"
    logger.info("ADF test — %s: p=%.4f (%s)", name, p_value, verdict)
    return p_value


def run_granger_causality(df_cpi: pd.DataFrame, df_mpc: pd.DataFrame) -> dict:
    """Test whether Core CPI Granger-causes Repo Rate changes.

    Checks stationarity of Core CPI, General CPI, and Repo Rate first. Since
    all three are non-stationary in this dataset, the Granger test is run on
    first-differenced series (which pass ADF), not raw levels.
    """
    cpi_mpc = pd.merge_asof(
        df_cpi.sort_values("date"),
        df_mpc[["date", "repo_rate"]].sort_values("date"),
        on="date",
        direction="backward",
    ).dropna(subset=["cpi_core_yoy", "repo_rate"])

    logger.info("Merged monthly CPI-MPC dataset for Granger test: %d rows", len(cpi_mpc))

    stationarity = {
        "core_cpi_yoy": check_stationarity(cpi_mpc["cpi_core_yoy"], "Core CPI YoY"),
        "general_cpi_yoy": check_stationarity(df_cpi["cpi_general_yoy"], "General CPI YoY"),
        "repo_rate": check_stationarity(cpi_mpc["repo_rate"], "Repo Rate"),
    }

    cpi_mpc["core_diff"] = cpi_mpc["cpi_core_yoy"].diff()
    cpi_mpc["repo_diff"] = cpi_mpc["repo_rate"].diff()
    stationarity["core_diff"] = check_stationarity(cpi_mpc["core_diff"], "Differenced Core CPI YoY")
    stationarity["repo_diff"] = check_stationarity(cpi_mpc["repo_diff"], "Differenced Repo Rate")

    gc_data = cpi_mpc[["repo_diff", "core_diff"]].dropna()
    logger.info("Running Granger causality on DIFFERENCED (stationary) series: Core CPI -> Repo Rate")
    gc_results = grangercausalitytests(gc_data, maxlag=GRANGER_MAX_LAG, verbose=False)

    p_values_by_lag = {
        lag: round(gc_results[lag][0]["ssr_ftest"][1], 4) for lag in range(1, GRANGER_MAX_LAG + 1)
    }
    for lag, p in p_values_by_lag.items():
        logger.info("Granger causality lag %d: p=%.4f", lag, p)

    significant_lags = [lag for lag, p in p_values_by_lag.items() if p < ADF_SIGNIFICANCE]
    conclusion = (
        "No robust evidence that Core CPI Granger-causes Repo Rate changes "
        "after correcting for non-stationarity."
        if len(significant_lags) <= 1
        else f"Significant at lags {significant_lags} even after correction — investigate further."
    )
    logger.info("Granger causality conclusion: %s", conclusion)

    return {
        "stationarity_p_values": stationarity,
        "granger_p_values_by_lag": p_values_by_lag,
        "conclusion": conclusion,
    }


def run_misattribution_analysis(df_mpc: pd.DataFrame) -> dict:
    """What fraction of rate hikes happened while food (supply-side)
    inflation exceeded core (demand-side) inflation."""
    df_hikes = df_mpc[df_mpc["decision"] == "hike"].copy()
    df_hikes["food_dominant"] = df_hikes["cpi_food_yoy"] > df_hikes["cpi_core_yoy"]

    total_hikes = len(df_hikes)
    supply_hikes = int(df_hikes["food_dominant"].sum())
    pct = (supply_hikes / total_hikes * 100) if total_hikes else 0.0

    logger.info(
        "Misattribution: %d/%d hikes (%.1f%%) occurred while food inflation exceeded core inflation",
        supply_hikes, total_hikes, pct,
    )
    return {"total_hikes": total_hikes, "supply_dominant_hikes": supply_hikes, "pct_supply_dominant": round(pct, 1)}


def run() -> dict:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_cpi, df_mpc = load_data()

    df_cpi = run_stl_decomposition(df_cpi)
    granger_results = run_granger_causality(df_cpi, df_mpc)
    misattribution_results = run_misattribution_analysis(df_mpc)

    logger.info("Stage 3 complete.")
    return {"granger": granger_results, "misattribution": misattribution_results}


if __name__ == "__main__":
    run()