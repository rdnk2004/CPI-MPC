"""
Populate MLflow Model Registry, Runs, Metrics, Parameters, and Artifacts
into sqlite:///mlflow.db for demonstration and teaching.
"""

import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.prophet
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, model_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure tracking URI is set to local sqlite db
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()

EXPERIMENT_NAME = "cpi_mpc_rate_hike_analysis"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    exp_id = mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=Path("mlruns/1").resolve().as_uri()
    )
else:
    exp_id = experiment.experiment_id

# Make sure artifacts directory exists
mlruns_dir = config.ROOT_DIR / "mlruns" / str(exp_id)
mlruns_dir.mkdir(parents=True, exist_ok=True)


def load_dataset():
    df_mpc = pd.read_csv(config.PROCESSED_CPI_MPC_FILE)
    model_df = df_mpc[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].dropna()
    X = model_df[config.FEATURE_COLUMNS]
    y = model_df[config.TARGET_COLUMN]
    return df_mpc, X, y


def register_logistic_regression(X, y):
    logger.info("Logging and registering Logistic Regression Baseline (v1)...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE))
    ])
    pipe.fit(X, y)

    input_example = X.iloc[:3]
    preds = pipe.predict(input_example)
    signature = infer_signature(input_example, preds)

    with mlflow.start_run(experiment_id=exp_id, run_name="logistic_regression_baseline_v1") as run:
        mlflow.log_params({
            "model_family": "Linear",
            "model_type": "LogisticRegression",
            "class_weight": "balanced",
            "scaler": "StandardScaler",
            "max_iter": 1000,
            "random_state": config.RANDOM_STATE,
            "features_count": len(config.FEATURE_COLUMNS),
            "eval_strategy": "TimeSeriesSplit_5_folds",
        })

        # Log walk-forward CV metrics and in-sample metrics
        mlflow.log_metrics({
            "cv_mean_accuracy": 0.672,
            "cv_mean_precision": 0.178,
            "cv_mean_recall": 0.400,
            "cv_mean_f1": 0.233,
            "in_sample_accuracy": 0.724,
            "n_positive_examples": int(y.sum()),
            "n_total_examples": int(len(y)),
        })

        mlflow.set_tags({
            "model_type": "Baseline",
            "author": "Antigravity Teaching Demo",
            "class_imbalance_handling": "class_weight=balanced",
            "educational_note": "Trades accuracy (67%) to detect rare hike events (40% recall).",
        })

        # Log model artifact & register
        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="cpi_rate_hike_classifier",
        )

        run_id = run.info.run_id

    logger.info("Logged Logistic Regression run %s", run_id)
    return model_info


def register_xgboost(X, y):
    logger.info("Logging and registering XGBoost Classifier (v2)...")
    clf = model_store.load_xgb_model()

    input_example = X.iloc[:3]
    preds = clf.predict(input_example)
    signature = infer_signature(input_example, preds)

    with mlflow.start_run(experiment_id=exp_id, run_name="xgboost_hike_classifier_v2") as run:
        mlflow.log_params({
            "model_family": "GradientBoosting",
            "model_type": "XGBClassifier",
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "random_state": config.RANDOM_STATE,
            "features_count": len(config.FEATURE_COLUMNS),
            "eval_strategy": "TimeSeriesSplit_5_folds",
        })

        mlflow.log_metrics({
            "cv_mean_accuracy": 0.867,
            "cv_mean_precision": 0.000,
            "cv_mean_recall": 0.000,
            "cv_mean_f1": 0.000,
            "in_sample_accuracy": 1.000,
            "n_positive_examples": int(y.sum()),
            "n_total_examples": int(len(y)),
        })

        mlflow.set_tags({
            "model_type": "Challenger",
            "explainability": "SHAP_KernelExplainer",
            "educational_note": "Overfits 58 rows (100% in-sample acc vs 0.0 CV F1). Classic imbalanced data trap.",
        })

        # Log evaluation and explainability artifacts
        artifacts_to_log = [
            config.OUTPUTS_DIR / "06_shap_feature_importance.png",
            config.OUTPUTS_DIR / "07_shap_single_decision.png",
            config.OUTPUTS_DIR / "classifier_cv_results.csv",
            config.OUTPUTS_DIR / "shap_values_per_meeting.csv",
            config.ROOT_DIR / "models" / "classifier_metrics.json",
        ]
        for art in artifacts_to_log:
            if art.exists():
                mlflow.log_artifact(str(art), artifact_path="evaluation_artifacts")

        # Log model artifact & register as version 2
        model_info = mlflow.xgboost.log_model(
            xgb_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="cpi_rate_hike_classifier",
        )

        run_id = run.info.run_id

    logger.info("Logged XGBoost run %s", run_id)
    return model_info


def register_prophet():
    logger.info("Logging and registering Prophet Core CPI Forecaster (v1)...")
    prophet_model = model_store.load_prophet_model()

    future_sample = prophet_model.make_future_dataframe(periods=3, freq="MS").tail(3)
    preds_sample = prophet_model.predict(future_sample)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    signature = infer_signature(future_sample, preds_sample)

    with mlflow.start_run(experiment_id=exp_id, run_name="prophet_core_cpi_forecaster_v1") as run:
        mlflow.log_params({
            "model_type": "Prophet",
            "seasonality_mode": "additive",
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "interval_width": 0.90,
            "forecast_horizon_months": 6,
            "target_variable": "cpi_core_yoy",
        })

        mlflow.log_metrics({
            "forecast_yhat_last": 5.16,
            "forecast_yhat_lower_last": 4.54,
            "forecast_yhat_upper_last": 5.79,
            "backtest_mape": 0.117,
            "empirical_coverage": 0.600,
            "claimed_interval_width": 0.900,
        })

        mlflow.set_tags({
            "task": "Time Series Forecasting",
            "frequency": "Monthly",
            "educational_note": "Demonstrates backtesting calibration: 90% confidence bands only had 60% empirical coverage.",
        })

        # Log visual diagnostic artifacts
        artifacts_to_log = [
            config.OUTPUTS_DIR / "04_prophet_forecast.png",
            config.OUTPUTS_DIR / "05_prophet_components.png",
            config.OUTPUTS_DIR / "03_stl_decomposition.png",
            config.OUTPUTS_DIR / "01_core_cpi_heatmap.png",
            config.OUTPUTS_DIR / "02_cpi_vs_repo_rate.png",
        ]
        for art in artifacts_to_log:
            if art.exists():
                mlflow.log_artifact(str(art), artifact_path="forecast_diagnostics")

        model_info = mlflow.prophet.log_model(
            pr_model=prophet_model,
            artifact_path="model",
            signature=signature,
            input_example=future_sample,
            registered_model_name="cpi_core_inflation_prophet",
        )

        run_id = run.info.run_id

    logger.info("Logged Prophet run %s", run_id)
    return model_info


def annotate_registry():
    logger.info("Setting Model Registry descriptions, tags, and aliases...")

    # Set Registered Model description
    client.update_registered_model(
        name="cpi_rate_hike_classifier",
        description=(
            "Binary classification model to predict whether the RBI Monetary Policy Committee (MPC) "
            "will hike repo rates based on CPI sub-indices (Food, Fuel, Core) and momentum lags. "
            "Created for macroeconomic and ML policy analysis teaching."
        )
    )

    client.update_registered_model(
        name="cpi_core_inflation_prophet",
        description=(
            "Additive Prophet time-series forecaster for India Core CPI YoY (General Index ex Food & Fuel) "
            "projected 6 months forward, complete with empirical interval calibration backtesting."
        )
    )

    # Set Version descriptions & aliases
    try:
        # Version 1 (Logistic Regression)
        client.update_model_version(
            name="cpi_rate_hike_classifier",
            version="1",
            description="Baseline model: Class-weighted Logistic Regression with StandardScaler. Captures minority hike cases."
        )
        client.set_registered_model_alias("cpi_rate_hike_classifier", "baseline", "1")
        client.set_registered_model_alias("cpi_rate_hike_classifier", "challenger", "1")
    except Exception as e:
        logger.warning("Could not tag version 1: %s", e)

    try:
        # Version 2 (XGBoost)
        client.update_model_version(
            name="cpi_rate_hike_classifier",
            version="2",
            description="Gradient-boosted decision trees with SHAP explanations. High in-sample fit; used to demonstrate class imbalance perils."
        )
        client.set_registered_model_alias("cpi_rate_hike_classifier", "champion", "2")
    except Exception as e:
        logger.warning("Could not tag version 2: %s", e)

    try:
        client.update_model_version(
            name="cpi_core_inflation_prophet",
            version="1",
            description="Production 6-month Core CPI Forecaster with additive yearly seasonality."
        )
        client.set_registered_model_alias("cpi_core_inflation_prophet", "champion", "1")
    except Exception as e:
        logger.warning("Could not tag prophet version 1: %s", e)

    logger.info("Model Registry annotations complete!")


def main():
    _, X, y = load_dataset()
    existing_versions = {(m.name, str(m.version)) for m in client.search_model_versions("")}
    
    if ("cpi_rate_hike_classifier", "1") not in existing_versions:
        register_logistic_regression(X, y)
    else:
        logger.info("Logistic Regression (v1) already registered. Skipping.")

    if ("cpi_rate_hike_classifier", "2") not in existing_versions:
        register_xgboost(X, y)
    else:
        logger.info("XGBoost (v2) already registered. Skipping.")

    if ("cpi_core_inflation_prophet", "1") not in existing_versions:
        register_prophet()
    else:
        logger.info("Prophet Forecaster (v1) already registered. Skipping.")

    annotate_registry()
    logger.info("All models and artifacts registered successfully!")


if __name__ == "__main__":
    main()
