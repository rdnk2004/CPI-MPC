"""
FastAPI service for the CPI-MPC project. Loads pre-trained model artifacts
on startup (see src/model_store.py) rather than retraining per request.

Run:
    uvicorn src.api:app --reload

Then visit http://localhost:8000/docs for interactive API docs.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config, model_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup. If artifacts are missing
    (e.g. first-ever run), train them on the fly rather than fail to start --
    but this should normally not happen outside first setup, since
    `python -m src.model_store` is meant to be run ahead of time."""
    try:
        _models["prophet"] = model_store.load_prophet_model()
        _models["xgb"] = model_store.load_xgb_model()
        _models["classifier_metrics"] = model_store.load_classifier_metrics()
        logger.info("Loaded pre-trained model artifacts.")
    except FileNotFoundError:
        logger.warning("Model artifacts not found -- training now. Run `python -m src.model_store` ahead of time to avoid this at startup.")
        model_store.run()
        _models["prophet"] = model_store.load_prophet_model()
        _models["xgb"] = model_store.load_xgb_model()
        _models["classifier_metrics"] = model_store.load_classifier_metrics()
    yield
    _models.clear()


app = FastAPI(
    title="CPI-MPC API",
    description="Core CPI inflation forecasting and RBI rate-hike classification.",
    version="1.0",
    lifespan=lifespan,
)


class HikeFeatures(BaseModel):
    """Input features for a hike prediction, matching config.FEATURE_COLUMNS."""
    cpi_general_yoy: float = Field(..., description="Headline CPI YoY %")
    cpi_food_yoy: float = Field(..., description="Food CPI YoY %")
    cpi_fuel_yoy: float = Field(..., description="Fuel CPI YoY %")
    cpi_core_yoy: float = Field(..., description="Core CPI YoY %")
    core_lag1: float = Field(..., description="Core CPI YoY %, 1 meeting prior")
    core_lag2: float = Field(..., description="Core CPI YoY %, 2 meetings prior")
    core_3m_avg: float = Field(..., description="3-month rolling average of Core CPI YoY %")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.get("/forecast/core-cpi")
def forecast_core_cpi(months: int = 6) -> dict:
    """Forecast Core CPI YoY inflation N months ahead using the pre-trained
    Prophet model."""
    if months < 1 or months > 24:
        raise HTTPException(status_code=400, detail="months must be between 1 and 24")

    model = _models["prophet"]
    future = model.make_future_dataframe(periods=months, freq="MS")
    forecast = model.predict(future)
    tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(months)

    return {
        "forecast": [
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "yhat": round(row["yhat"], 2),
                "yhat_lower": round(row["yhat_lower"], 2),
                "yhat_upper": round(row["yhat_upper"], 2),
            }
            for _, row in tail.iterrows()
        ],
        "calibration_note": (
            "Backtesting showed this interval's empirical coverage is ~60% at a "
            "6-month horizon against a claimed 90% -- treat the interval as "
            "indicative, not a calibrated confidence band."
        ),
    }


@app.post("/predict/hike")
def predict_hike(features: HikeFeatures) -> dict:
    """Predict the probability of an MPC rate hike given current CPI features.

    IMPORTANT: this classifier demonstrated no predictive skill beyond the
    majority-class baseline in walk-forward cross-validation (see
    classifier_metrics.reliability_note in the response). It is served here
    for demonstration/exploratory purposes, not as a decision-grade signal.
    """
    model = _models["xgb"]
    X = pd.DataFrame([features.model_dump()])[config.FEATURE_COLUMNS]

    proba = model.predict_proba(X)[0]
    prediction = int(model.predict(X)[0])

    return {
        "prediction": "hike" if prediction == 1 else "no_hike",
        "hike_probability": round(float(proba[1]), 4),
        "reliability": "exploratory_not_decision_grade",
        "classifier_metrics": _models["classifier_metrics"],
    }


@app.get("/model-info")
def model_info() -> dict:
    """Full transparency endpoint: what's being served, and how reliable it
    actually is, in one place."""
    return {
        "forecast_model": "Prophet (Core CPI YoY, additive seasonality)",
        "classifier_model": "XGBoost (n_estimators=50, max_depth=3)",
        "classifier_reliability": _models["classifier_metrics"],
    }