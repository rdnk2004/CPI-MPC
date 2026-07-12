"""
Trains the Prophet forecaster and the final XGBoost classifier once, and
persists them to disk -- so the API (src/api.py) loads pre-trained artifacts
on startup instead of retraining on every request.

Also persists the classifier's walk-forward CV metrics (F1, precision,
recall) alongside the model, so the API can honestly report the model's
demonstrated reliability at inference time, not just make a prediction.

Run standalone to (re)train and save all artifacts:
    python -m src.model_store
"""

import json
import logging

import joblib
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from src import ai_model, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = config.ROOT_DIR / "models"
PROPHET_MODEL_FILE = MODELS_DIR / "prophet_core_cpi.json"
XGB_MODEL_FILE = MODELS_DIR / "xgboost_hike_classifier.joblib"
CLASSIFIER_METRICS_FILE = MODELS_DIR / "classifier_metrics.json"


def train_and_save_prophet(df_cpi: pd.DataFrame) -> Prophet:
    core_prophet = df_cpi[["date", "cpi_core_yoy"]].rename(
        columns={"date": "ds", "cpi_core_yoy": "y"}
    ).dropna()

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
        seasonality_mode="additive", interval_width=0.90,
    )
    model.fit(core_prophet)

    with open(PROPHET_MODEL_FILE, "w") as f:
        f.write(model_to_json(model))
    logger.info("Saved Prophet model to %s", PROPHET_MODEL_FILE)
    return model


def train_and_save_classifier(df_mpc: pd.DataFrame, cv_results: pd.DataFrame) -> xgb.XGBClassifier:
    """Train the final XGBoost classifier on all data and save it, along
    with its walk-forward CV metrics -- the CV metrics are what the API
    actually reports as the model's reliability, since in-sample metrics on
    this final fit are inflated by overfitting (see ai_model.py).
    """
    model_df = df_mpc[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].dropna()
    X = model_df[config.FEATURE_COLUMNS]
    y = model_df[config.TARGET_COLUMN]

    clf = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=config.RANDOM_STATE,
    )
    clf.fit(X, y)
    joblib.dump(clf, XGB_MODEL_FILE)
    logger.info("Saved XGBoost classifier to %s", XGB_MODEL_FILE)

    xgb_cv = cv_results[cv_results["model"] == "xgboost"]
    metrics = {
        "mean_accuracy": round(xgb_cv["accuracy"].mean(), 3),
        "mean_precision": round(xgb_cv["precision"].mean(), 3),
        "mean_recall": round(xgb_cv["recall"].mean(), 3),
        "mean_f1": round(xgb_cv["f1"].mean(), 3),
        "n_positive_examples": int(y.sum()),
        "n_total_examples": int(len(y)),
        "reliability_note": (
            "This classifier shows no demonstrated predictive skill on held-out data "
            "(walk-forward F1 = 0.0 across all folds). Predictions are exploratory only "
            "and should not be treated as decision-grade."
            if xgb_cv["f1"].mean() == 0
            else "Evaluated via walk-forward TimeSeriesSplit; see mean_f1 for demonstrated skill."
        ),
    }
    with open(CLASSIFIER_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved classifier reliability metrics to %s", CLASSIFIER_METRICS_FILE)
    return clf


def load_prophet_model() -> Prophet:
    with open(PROPHET_MODEL_FILE, "r") as f:
        return model_from_json(f.read())


def load_xgb_model() -> xgb.XGBClassifier:
    return joblib.load(XGB_MODEL_FILE)


def load_classifier_metrics() -> dict:
    with open(CLASSIFIER_METRICS_FILE, "r") as f:
        return json.load(f)


def run() -> None:
    """Train and persist both model artifacts. Reuses ai_model.py's
    evaluate_classifiers() so the CV metrics saved here are guaranteed to be
    computed the exact same way as everywhere else in the project."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_cpi = pd.read_csv(config.CLEANED_CPI_FILE)
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
    df_mpc = pd.read_csv(config.PROCESSED_CPI_MPC_FILE)
    df_mpc["date"] = pd.to_datetime(df_mpc["date"])

    train_and_save_prophet(df_cpi)
    cv_results = ai_model.evaluate_classifiers(df_mpc)
    train_and_save_classifier(df_mpc, cv_results)

    logger.info("Model artifacts trained and saved to %s", MODELS_DIR)


if __name__ == "__main__":
    run()