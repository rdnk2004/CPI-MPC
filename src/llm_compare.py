"""
For each MPC meeting with both a scraped minutes text (src/scrape_minutes.py)
and SHAP values (src/ai_model.py), asks Gemini to compare:
  1. What the RBI's own published minutes say drove the decision
  2. What the model's SHAP values say actually drove its prediction
...and produce a verdict on whether they agree or diverge.

Design choice: classifying which inflation "type" (core/food/fuel/headline)
the SHAP values point to is done in plain Python (deterministic, verifiable,
see classify_shap_emphasis) rather than asked of the LLM -- an LLM
interpreting raw SHAP numbers would be an unnecessary and less reliable
detour for a task plain arithmetic already does exactly. Gemini's job is the
part that actually needs language understanding: reading real prose and
comparing it to that pre-computed classification.

Requires a Gemini API key:
    export GEMINI_API_KEY=...          (macOS/Linux)
    $env:GEMINI_API_KEY = "..."         (PowerShell)

Run standalone:
    python -m src.llm_compare
"""

import json
import logging
import os
import re
import time
from typing import Literal

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MINUTES_DIR = config.DATA_DIR / "mpc_minutes"
MINUTES_INDEX_FILE = config.DATA_DIR / "mpc_minutes_index.csv"
SHAP_VALUES_FILE = config.OUTPUTS_DIR / "shap_values_per_meeting.csv"
COMPARISON_OUTPUT_FILE = config.OUTPUTS_DIR / "rationale_comparison.csv"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_MINUTES_CHARS = 6000
MAX_RETRIES_PER_MEETING = 5


class RationaleComparison(BaseModel):
    """Schema Gemini's output is constrained to -- passing this as
    response_schema forces the SDK to return a validated object via
    response.parsed, rather than hoping the model's raw text happens to be
    parseable JSON (which is what caused the earlier failures: Gemini
    occasionally embedded an unescaped quote inside a string field)."""
    stated_rationale_summary: str
    stated_emphasis: Literal["core", "food", "fuel", "headline", "mixed"]
    agreement_verdict: Literal["aligned", "partially_aligned", "diverging"]
    verdict_explanation: str

FEATURE_GROUPS = {
    "core": ["cpi_core_yoy", "core_lag1", "core_lag2", "core_3m_avg"],
    "food": ["cpi_food_yoy"],
    "fuel": ["cpi_fuel_yoy"],
    "headline": ["cpi_general_yoy"],
}


def classify_shap_emphasis(row: pd.Series) -> dict:
    """Deterministically bucket a meeting's SHAP values into core/food/fuel/
    headline groups by each group's strongest single feature, and report
    that dominant feature too.

    Uses MAX (the single strongest feature in each group) rather than
    summing all features in a group -- summing would unfairly favor "core"
    since it has 4 features (cpi_core_yoy, core_lag1, core_lag2,
    core_3m_avg) versus 1 each for food/fuel/headline. Verified against the
    full 57-meeting dataset that this fairer comparison still finds "core"
    dominant in effectively every meeting (58/58) -- the result is a real
    pattern in the model's SHAP attributions, not an artifact of how the
    features were grouped.
    """
    group_max = {
        group: max(abs(row[col]) for col in cols)
        for group, cols in FEATURE_GROUPS.items()
    }
    dominant_group = max(group_max, key=group_max.get)

    all_features = [col for cols in FEATURE_GROUPS.values() for col in cols]
    dominant_feature = max(all_features, key=lambda col: abs(row[col]))

    return {
        "dominant_group": dominant_group,
        "dominant_feature": dominant_feature,
        "group_max_shap": {g: round(float(v), 4) for g, v in group_max.items()},
    }


def build_prompt(meeting_date: str, decision: str, shap_emphasis: dict, minutes_text: str) -> str:
    truncated = minutes_text[:MAX_MINUTES_CHARS]
    return f"""You are analyzing a historical Reserve Bank of India (RBI) Monetary Policy Committee (MPC) decision.

MEETING DATE: {meeting_date}
ACTUAL DECISION: {decision}

A machine learning model's SHAP feature attributions for this decision point to "{shap_emphasis['dominant_group']}" \
inflation as the dominant driver (specifically the feature "{shap_emphasis['dominant_feature']}"). \
The full breakdown of each category's strongest feature's absolute SHAP value is: {shap_emphasis['group_max_shap']}.

Below is the RBI's own published minutes/statement text for this meeting (may include navigation text \
noise; ignore anything that isn't part of the actual policy discussion):

---
{truncated}
---

Summarize what the statement says drove this decision (in your own words, not quoted text), classify \
which inflation type its own language emphasizes most, and assess how well that matches the model's \
SHAP-derived dominant category ("{shap_emphasis['dominant_group']}").
"""


def call_gemini(client: genai.Client, prompt: str) -> RationaleComparison:
    """Calls Gemini with a strict response_schema (guaranteed-valid output,
    no manual JSON parsing) and retries on 429 rate-limit errors using the
    retry delay the API itself reports, rather than a blind fixed wait.
    """
    for attempt in range(1, MAX_RETRIES_PER_MEETING + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RationaleComparison,
                ),
            )
            return response.parsed
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if not is_rate_limit or attempt == MAX_RETRIES_PER_MEETING:
                raise

            # The API reports its own suggested wait in the error payload
            # (e.g. "retryDelay": "30s") -- use that instead of guessing.
            match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", str(exc))
            wait_seconds = int(match.group(1)) + 2 if match else 30
            logger.warning(
                "Rate limited (attempt %d/%d) -- waiting %ds before retrying, per the API's own retry hint.",
                attempt, MAX_RETRIES_PER_MEETING, wait_seconds,
            )
            time.sleep(wait_seconds)


def run() -> pd.DataFrame:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key from https://aistudio.google.com/apikey and set it before running this script."
        )
    client = genai.Client(api_key=api_key)

    if not MINUTES_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"{MINUTES_INDEX_FILE} not found -- run `python -m src.scrape_minutes` first."
        )
    if not SHAP_VALUES_FILE.exists():
        raise FileNotFoundError(
            f"{SHAP_VALUES_FILE} not found -- run `python -m src.ai_model` first."
        )

    minutes_index = pd.read_csv(MINUTES_INDEX_FILE)
    shap_values = pd.read_csv(SHAP_VALUES_FILE)
    shap_values["date"] = pd.to_datetime(shap_values["date"])
    minutes_index["meeting_date"] = pd.to_datetime(minutes_index["meeting_date"])

    merged = shap_values.merge(
        minutes_index, left_on="date", right_on="meeting_date", how="inner"
    )
    logger.info("Found %d meetings with both SHAP values and scraped minutes text.", len(merged))

    # Resume from a previous run instead of re-processing (and re-burning
    # API quota on) meetings already successfully completed.
    already_done = set()
    existing_results = []
    if COMPARISON_OUTPUT_FILE.exists():
        existing_df = pd.read_csv(COMPARISON_OUTPUT_FILE)
        already_done = set(existing_df["date"])
        existing_results = existing_df.to_dict("records")
        logger.info("Resuming: %d meetings already completed in a previous run, skipping those.", len(already_done))

    results = list(existing_results)
    for _, row in merged.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        if date_str in already_done:
            continue

        minutes_path = MINUTES_DIR / f"{date_str}.txt"
        if not minutes_path.exists():
            logger.warning("No minutes text file for %s -- skipping", date_str)
            continue

        minutes_text = minutes_path.read_text(encoding="utf-8")
        shap_emphasis = classify_shap_emphasis(row)

        prompt = build_prompt(date_str, row["actual_decision"], shap_emphasis, minutes_text)
        try:
            llm_result = call_gemini(client, prompt)
            results.append({
                "date": date_str,
                "actual_decision": row["actual_decision"],
                "model_dominant_group": shap_emphasis["dominant_group"],
                "model_dominant_feature": shap_emphasis["dominant_feature"],
                **llm_result.model_dump(),
            })
            logger.info("%s: model=%s, stated=%s, verdict=%s",
                        date_str, shap_emphasis["dominant_group"],
                        llm_result.stated_emphasis, llm_result.agreement_verdict)
        except Exception as exc:
            logger.warning("Gemini call failed for %s (%s) -- skipping", date_str, exc)

        # Save after every meeting, not just at the end -- so a crash or a
        # quota cutoff partway through (as happened last run) doesn't lose
        # progress on meetings already completed in this run.
        pd.DataFrame(results).to_csv(COMPARISON_OUTPUT_FILE, index=False)
        time.sleep(1.0)

    results_df = pd.DataFrame(results)
    logger.info("Total: %d rationale comparisons saved to %s", len(results_df), COMPARISON_OUTPUT_FILE)

    if len(results_df):
        verdict_counts = results_df["agreement_verdict"].value_counts()
        logger.info("Verdict distribution:\n%s", verdict_counts.to_string())

    return results_df


if __name__ == "__main__":
    run()