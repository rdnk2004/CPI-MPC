"""
Streamlit dashboard for the CPI-MPC project. Talks to the FastAPI service
(src/api.py) over HTTP for the forecast and hike-probability predictions --
it does not retrain or reimplement any model logic itself, so the dashboard
can never drift from what the API actually serves.

Run (with the API already running separately):
    uvicorn src.api:app --reload &
    streamlit run src/dashboard.py
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src import config

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="CPI-MPC Dashboard", layout="wide")


@st.cache_data
def load_historical_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_cpi = pd.read_csv(config.CLEANED_CPI_FILE)
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
    df_mpc = pd.read_csv(config.PROCESSED_CPI_MPC_FILE)
    df_mpc["date"] = pd.to_datetime(df_mpc["date"])
    return df_cpi, df_mpc


def call_api(method: str, path: str, **kwargs):
    """Thin wrapper so every API call fails loudly and visibly in the UI
    (via st.error) rather than crashing the whole dashboard."""
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", timeout=10, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_BASE_URL}. Is `uvicorn src.api:app` running?")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API returned an error: {exc}")
        return None


st.title("🇮🇳 India CPI & RBI Monetary Policy Dashboard")
st.caption(f"Backed by the FastAPI service at `{API_BASE_URL}`")

df_cpi, df_mpc = load_historical_data()
latest = df_cpi.iloc[-1]

# --------------------------------------------------------------------------
# Top-line metrics
# --------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Headline CPI YoY", f"{latest['cpi_general_yoy']:.2f}%")
col2.metric("Latest Core CPI YoY", f"{latest['cpi_core_yoy']:.2f}%")
col3.metric("Latest Food CPI YoY", f"{latest['cpi_food_yoy']:.2f}%")
latest_mpc = df_mpc.iloc[-1]
col4.metric("Current Repo Rate", f"{latest_mpc['repo_rate']:.2f}%", help=f"As of {latest_mpc['date'].date()}")

st.divider()

# --------------------------------------------------------------------------
# Historical trend
# --------------------------------------------------------------------------
st.subheader("Historical Inflation vs. Repo Rate")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["cpi_general_yoy"], name="Headline CPI", line=dict(width=2)))
fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["cpi_core_yoy"], name="Core CPI", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=df_cpi["date"], y=df_cpi["cpi_food_yoy"], name="Food CPI", opacity=0.6))
fig.add_hline(y=4, line_dash="dot", line_color="grey", annotation_text="RBI 4% target")
fig.add_hline(y=6, line_dash="dot", line_color="red", annotation_text="Upper tolerance band")
fig.update_layout(height=400, legend=dict(orientation="h", y=1.1), margin=dict(t=30))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Forecast (from the API)
# --------------------------------------------------------------------------
st.subheader("Core CPI Forecast")
months = st.slider("Forecast horizon (months)", min_value=1, max_value=12, value=6)

forecast_data = call_api("GET", f"/forecast/core-cpi?months={months}")
if forecast_data:
    forecast_df = pd.DataFrame(forecast_data["forecast"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
        y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"])[::-1],
        fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0),
        name="90% interval (see calibration note)", showlegend=True,
    ))
    fig2.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["yhat"], name="Forecast", line=dict(width=3)))
    fig2.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.info(forecast_data["calibration_note"], icon="⚠️")

st.divider()

# --------------------------------------------------------------------------
# Hike probability calculator (from the API)
# --------------------------------------------------------------------------
st.subheader("Rate Hike Probability Calculator")
st.caption("Adjust CPI features below to see the classifier's predicted hike probability for a hypothetical MPC meeting.")

with st.form("hike_form"):
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        cpi_general_yoy = st.number_input("Headline CPI YoY %", value=float(latest["cpi_general_yoy"]))
        cpi_food_yoy = st.number_input("Food CPI YoY %", value=float(latest["cpi_food_yoy"]))
        cpi_fuel_yoy = st.number_input("Fuel CPI YoY %", value=float(latest["cpi_fuel_yoy"]))
        cpi_core_yoy = st.number_input("Core CPI YoY %", value=float(latest["cpi_core_yoy"]))
    with fcol2:
        core_lag1 = st.number_input("Core CPI YoY %, 1 meeting prior", value=float(latest["cpi_core_yoy"]))
        core_lag2 = st.number_input("Core CPI YoY %, 2 meetings prior", value=float(latest["cpi_core_yoy"]))
        core_3m_avg = st.number_input("Core CPI 3-month rolling avg %", value=float(latest["cpi_core_yoy"]))
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "cpi_general_yoy": cpi_general_yoy, "cpi_food_yoy": cpi_food_yoy,
        "cpi_fuel_yoy": cpi_fuel_yoy, "cpi_core_yoy": cpi_core_yoy,
        "core_lag1": core_lag1, "core_lag2": core_lag2, "core_3m_avg": core_3m_avg,
    }
    result = call_api("POST", "/predict/hike", json=payload)
    if result:
        pcol1, pcol2 = st.columns([1, 2])
        pcol1.metric("Predicted Hike Probability", f"{result['hike_probability']*100:.1f}%")
        pcol1.metric("Prediction", result["prediction"].replace("_", " ").title())
        with pcol2:
            st.warning(
                f"**Reliability: {result['reliability'].replace('_', ' ')}**\n\n"
                f"{result['classifier_metrics']['reliability_note']}\n\n"
                f"Walk-forward CV metrics: F1={result['classifier_metrics']['mean_f1']}, "
                f"Precision={result['classifier_metrics']['mean_precision']}, "
                f"Recall={result['classifier_metrics']['mean_recall']} "
                f"(from {result['classifier_metrics']['n_positive_examples']} positive examples "
                f"out of {result['classifier_metrics']['n_total_examples']} total meetings).",
                icon="⚠️",
            )

st.divider()

# --------------------------------------------------------------------------
# Model transparency
# --------------------------------------------------------------------------
st.subheader("Model Transparency")
info = call_api("GET", "/model-info")
if info:
    st.json(info)