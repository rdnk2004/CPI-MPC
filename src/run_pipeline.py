"""
Runs the full CPI-MPC pipeline end to end, in order:
    1. data_prep     -- clean + merge raw CPI and MPC Excel files
    2. eda            -- heatmap, CPI-vs-repo-rate chart, correlations
    3. decomposition   -- STL decomposition, corrected Granger causality, misattribution
    4. ai_model         -- Prophet forecast (+ calibration check), classifier
                            evaluation (XGBoost vs. logistic baseline), SHAP

Each stage depends on the previous stage's saved CSV outputs in data/, so
they must run in this order on a fresh checkout.

Usage:
    python -m src.run_pipeline
"""

import logging

from src import ai_model, data_prep, decomposition, eda

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_all() -> dict:
    logger.info("=== Stage 1: Data Prep ===")
    df_cpi, df_merged = data_prep.run()

    logger.info("=== Stage 2: EDA ===")
    eda.run()

    logger.info("=== Stage 3: Decomposition & Causal Analysis ===")
    stage3_results = decomposition.run()

    logger.info("=== Stage 4: Forecasting & Classification ===")
    stage4_results = ai_model.run()

    logger.info("=== Pipeline complete ===")
    return {
        "cpi": df_cpi,
        "merged": df_merged,
        "stage3": stage3_results,
        "stage4": stage4_results,
    }


if __name__ == "__main__":
    run_all()