"""
Stage 4: Inflation Forecasting & MPC Policy Classification (AI Layer).

1. Prophet forecast of Core CPI YoY, 6 months out -- plus a backtesting
   diagnostic (`cross_validation`) to check whether the claimed 90% interval
   is actually well-calibrated at that horizon (it wasn't, in the original
   run: ~60% empirical coverage at 180 days. See README changelog).
2. MPC hike classifier: XGBoost AND a class-weighted logistic regression
   baseline, evaluated with walk-forward TimeSeriesSplit reporting
   precision/recall/F1 -- not just accuracy, which is misleading here since
   only ~14% of meetings are hikes (a constant "never hike" predictor already
   scores ~87% accuracy).
3. SHAP explainability on a final full-data XGBoost fit, with an explicit
   caveat carried through from the CV results: this final model is prone to
   overfitting on ~58 rows, so treat SHAP output as descriptive, not as
   validated decision rules.

Run standalone:
    python -m src.ai_model
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src import config, tracking

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("shap").setLevel(logging.WARNING)

N_SPLITS = 5
FORECAST_HORIZON_MONTHS = 6
EXAMPLE_DECISION_DATE = "2022-05-04"  # off-cycle 40bps hike, used for the SHAP force plot


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_cpi = pd.read_csv(config.CLEANED_CPI_FILE)
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
    df_mpc = pd.read_csv(config.PROCESSED_CPI_MPC_FILE)
    df_mpc["date"] = pd.to_datetime(df_mpc["date"])
    return df_cpi, df_mpc


# --------------------------------------------------------------------------
# 1. Prophet forecast + calibration check
# --------------------------------------------------------------------------

def forecast_core_cpi(df_cpi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Prophet on Core CPI YoY, forecast N months out, and run a
    backtesting diagnostic to check whether the 90% interval is calibrated.
    """
    core_prophet = df_cpi[["date", "cpi_core_yoy"]].rename(
        columns={"date": "ds", "cpi_core_yoy": "y"}
    ).dropna()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.90,
    )
    model.fit(core_prophet)

    future = model.make_future_dataframe(periods=FORECAST_HORIZON_MONTHS, freq="MS")
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    fig1.suptitle(f"Prophet Forecast: India Core CPI (YoY %) — Next {FORECAST_HORIZON_MONTHS} Months", fontsize=11, y=1.02)
    fig1.savefig(config.OUTPUTS_DIR / "04_prophet_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2 = model.plot_components(forecast)
    fig2.savefig(config.OUTPUTS_DIR / "05_prophet_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(FORECAST_HORIZON_MONTHS).round(2)
    logger.info("Forecasted Core CPI YoY, next %d months:\n%s", FORECAST_HORIZON_MONTHS, tail.to_string(index=False))

    # Calibration check: does the 90% interval actually cover ~90% of real
    # outcomes at a 6-month (~180 day) horizon, historically?
    perf = None
    try:
        df_cv = cross_validation(
            model, initial="2555 days", period="180 days", horizon="180 days", disable_tqdm=True,
        )
        perf = performance_metrics(df_cv)
        near_horizon = perf[perf["horizon"] >= pd.Timedelta(days=170)]
        if len(near_horizon):
            mape = near_horizon["mape"].mean()
            coverage = near_horizon["coverage"].mean()
            logger.info(
                "Prophet calibration check at ~6-month horizon: MAPE=%.1f%%, empirical coverage=%.0f%% (claimed 90%%)",
                mape * 100, coverage * 100,
            )
            if coverage < 0.80:
                logger.warning(
                    "Forecast interval is overconfident at this horizon (empirical coverage %.0f%% < claimed 90%%). "
                    "Report the interval as indicative, not calibrated, in any write-up.", coverage * 100,
                )
    except Exception as exc:  # pragma: no cover - diagnostic only, shouldn't block the pipeline
        logger.warning("Prophet cross-validation diagnostic failed (%s) -- skipping calibration check.", exc)

    tracking.log_forecast_run(
        forecast_params={
            "yearly_seasonality": True,
            "seasonality_mode": "additive",
            "interval_width": 0.90,
            "horizon_months": FORECAST_HORIZON_MONTHS,
        },
        calibration=perf,
        forecast_tail=tail,
    )

    return forecast, perf


# --------------------------------------------------------------------------
# 2. Classifier evaluation: XGBoost vs. logistic regression baseline
# --------------------------------------------------------------------------

def evaluate_classifiers(df_mpc: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward evaluation of both models, reporting precision/recall/F1.

    Accuracy alone is misleading here: only ~14% of meetings are hikes, so a
    model that always predicts "no hike" already scores ~87% by accuracy
    without learning anything.
    """
    model_df = df_mpc[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].dropna().copy()
    X = model_df[config.FEATURE_COLUMNS]
    y = model_df[config.TARGET_COLUMN]

    hike_rate = y.mean()
    logger.info("Classifier dataset: %d meetings, hike rate=%.1f%% (naive 'always no-hike' baseline accuracy=%.1f%%)",
                len(model_df), hike_rate * 100, (1 - hike_rate) * 100)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for model_name in ["xgboost", "logistic_regression"]:
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if model_name == "logistic_regression":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE)
            else:
                clf = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    eval_metric="logloss", random_state=config.RANDOM_STATE,
                )

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            rows.append({
                "model": model_name, "fold": fold + 1, "n_test": len(y_test),
                "actual_hikes": int(y_test.sum()), "predicted_hikes": int(preds.sum()),
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
            })

    results = pd.DataFrame(rows)
    results.to_csv(config.OUTPUTS_DIR / "classifier_cv_results.csv", index=False)

    summary = results.groupby("model")[["accuracy", "precision", "recall", "f1"]].mean().round(3)
    logger.info("Mean CV metrics by model:\n%s", summary.to_string())

    if summary.loc["xgboost", "f1"] == 0:
        logger.warning(
            "XGBoost F1=0.000 across all folds -- it never predicts a positive (hike) class on held-out data. "
            "With only %d positive examples total, treat any classifier result here as exploratory, not decision-grade.",
            int(y.sum()),
        )

    model_params = {
        "xgboost": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        "logistic_regression": {"class_weight": "balanced", "max_iter": 1000},
    }
    tracking.log_classifier_fold_runs(results, model_params)

    return results


# --------------------------------------------------------------------------
# 3. SHAP explainability on the final full-data fit
# --------------------------------------------------------------------------

def fit_final_model_and_explain(df_mpc: pd.DataFrame, cv_results: pd.DataFrame) -> dict:
    """Fit XGBoost on all available data and compute SHAP values.

    NOTE: this final model is fit on 100% of the data (no held-out set), so
    its own in-sample accuracy is not a reliability signal -- it will tend to
    memorize a small dataset like this one. The CV results above are the
    honest measure of predictive skill; SHAP here should be read as
    "what does this model lean on to reproduce the historical labels",
    which is a valid descriptive question, not evidence of a validated
    decision rule.
    """
    model_df = df_mpc[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].dropna().copy()
    X = model_df[config.FEATURE_COLUMNS]
    y = model_df[config.TARGET_COLUMN]

    clf_final = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=config.RANDOM_STATE,
    )
    clf_final.fit(X, y)

    in_sample_acc = accuracy_score(y, clf_final.predict(X))
    xgb_cv_f1 = cv_results.loc[cv_results["model"] == "xgboost", "f1"].mean()
    logger.info("Final model in-sample accuracy: %.3f (vs. mean out-of-sample F1 from CV: %.3f)", in_sample_acc, xgb_cv_f1)
    if in_sample_acc > 0.95 and xgb_cv_f1 < 0.2:
        logger.warning(
            "Large gap between in-sample accuracy (%.2f) and out-of-sample CV F1 (%.2f) indicates overfitting. "
            "The SHAP explanations below describe this overfit model, not a validated predictive relationship.",
            in_sample_acc, xgb_cv_f1,
        )

    logger.info("Computing SHAP values (KernelExplainer, single-row background)...")
    background = X.median().to_frame().T
    explainer = shap.KernelExplainer(lambda x: clf_final.predict_proba(x), background)
    shap_values = explainer.shap_values(X)

    pos_shap_values = shap_values[:, :, 1]
    expected_val = explainer.expected_value[1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(pos_shap_values, X, feature_names=config.FEATURE_COLUMNS, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance: What Drives RBI Rate Hike Predictions?", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "06_shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "model": clf_final,
        "X": X,
        "model_df": model_df,
        "pos_shap_values": pos_shap_values,
        "expected_val": expected_val,
    }


def explain_single_decision(df_mpc: pd.DataFrame, shap_bundle: dict, decision_date: str = EXAMPLE_DECISION_DATE) -> None:
    """Force plot for one specific MPC decision, showing which features
    pushed the model's predicted hike probability up or down."""
    X = shap_bundle["X"]
    model_df = shap_bundle["model_df"]
    pos_shap_values = shap_bundle["pos_shap_values"]
    expected_val = shap_bundle["expected_val"]

    hike_idx = df_mpc[df_mpc["date"] == decision_date].index
    if len(hike_idx) == 0:
        logger.warning("Decision date %s not found in dataset -- skipping force plot.", decision_date)
        return

    idx_val = hike_idx[0]
    x_idx = model_df.index.get_loc(idx_val)

    logger.info("Explaining decision on %s. Features:\n%s", decision_date, X.iloc[x_idx].round(2).to_string())

    # Note: shap.initjs() is only needed for the interactive JS widget in a
    # notebook; it requires IPython and isn't needed for this static
    # matplotlib=True force plot, so it's omitted here.
    shap.force_plot(expected_val, pos_shap_values[x_idx], X.iloc[x_idx], matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot: Deconstructing the {decision_date} Rate Hike Decision", fontsize=11, pad=20)
    plt.savefig(config.OUTPUTS_DIR / "07_shap_single_decision.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", config.OUTPUTS_DIR / "07_shap_single_decision.png")


def run() -> dict:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tracking.init_tracking()
    df_cpi, df_mpc = load_data()

    forecast, calibration = forecast_core_cpi(df_cpi)
    cv_results = evaluate_classifiers(df_mpc)
    shap_bundle = fit_final_model_and_explain(df_mpc, cv_results)
    explain_single_decision(df_mpc, shap_bundle)

    logger.info("Stage 4 complete.")
    return {"forecast": forecast, "calibration": calibration, "cv_results": cv_results}


if __name__ == "__main__":
    run()