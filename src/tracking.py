"""
MLflow experiment tracking for the classifier evaluation and Prophet forecast.

Uses the local file-based tracking store (./mlruns) by default -- no server
required. Override with the MLFLOW_TRACKING_URI environment variable to point
at a remote tracking server instead.

View results with:
    mlflow ui
then open http://localhost:5000
"""

import logging

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "cpi_mpc_rate_hike_analysis"


def init_tracking() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_classifier_fold_runs(cv_results: pd.DataFrame, model_params: dict) -> None:
    """Log one MLflow run per (model, fold) combination, plus a summary run
    per model with the mean metrics across folds.

    Logging per-fold lets you later inspect *which* fold a model struggled
    on (e.g. fold 2's false-positive spike for logistic regression) rather
    than only ever seeing an averaged number.
    """
    for model_name, group in cv_results.groupby("model"):
        with mlflow.start_run(run_name=f"{model_name}_cv_summary"):
            mlflow.log_params({"model": model_name, **model_params.get(model_name, {})})
            mlflow.log_metrics(
                {
                    "mean_accuracy": group["accuracy"].mean(),
                    "mean_precision": group["precision"].mean(),
                    "mean_recall": group["recall"].mean(),
                    "mean_f1": group["f1"].mean(),
                }
            )
            for _, row in group.iterrows():
                with mlflow.start_run(run_name=f"{model_name}_fold{row['fold']}", nested=True):
                    mlflow.log_params({"model": model_name, "fold": int(row["fold"])})
                    mlflow.log_metrics(
                        {
                            "accuracy": row["accuracy"],
                            "precision": row["precision"],
                            "recall": row["recall"],
                            "f1": row["f1"],
                            "actual_hikes": row["actual_hikes"],
                            "predicted_hikes": row["predicted_hikes"],
                        }
                    )
    logger.info("Logged classifier CV results to MLflow experiment '%s'", EXPERIMENT_NAME)


def log_forecast_run(forecast_params: dict, calibration: pd.DataFrame | None, forecast_tail: pd.DataFrame) -> None:
    """Log the Prophet forecast configuration, headline forecast values, and
    (if available) the calibration diagnostic."""
    with mlflow.start_run(run_name="prophet_core_cpi_forecast"):
        mlflow.log_params(forecast_params)

        last_row = forecast_tail.iloc[-1]
        mlflow.log_metrics(
            {
                "forecast_yhat_last": last_row["yhat"],
                "forecast_yhat_lower_last": last_row["yhat_lower"],
                "forecast_yhat_upper_last": last_row["yhat_upper"],
            }
        )

        if calibration is not None and len(calibration):
            near_horizon = calibration[calibration["horizon"] >= pd.Timedelta(days=170)]
            if len(near_horizon):
                mlflow.log_metrics(
                    {
                        "backtest_mape": near_horizon["mape"].mean(),
                        "backtest_coverage": near_horizon["coverage"].mean(),
                        "claimed_interval_width": forecast_params.get("interval_width", 0.90),
                    }
                )
    logger.info("Logged Prophet forecast run to MLflow experiment '%s'", EXPERIMENT_NAME)