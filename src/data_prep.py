"""
Stage 1: CPI Inflation & MPC Decisions Data Preparation.

Cleans the raw CPI index Excel file and the raw MPC decisions Excel file,
computes YoY inflation rates, then merges MPC meeting dates onto the most
recently available CPI data using a backward as-of merge (no lookahead bias).

Run standalone:
    python -m src.data_prep
"""

import logging

import numpy as np
import pandas as pd

from src import config, db, validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean_cpi(raw_file=config.CPI_RAW_FILE) -> pd.DataFrame:
    """Parse the raw CPI Excel sheet into a tidy General/Food/Fuel/Core dataframe.

    The raw sheet has several title rows before the real header, and each
    commodity group is a separate block of rows rather than a column -- so we
    filter by the `commodity` label rather than relying on column position
    alone for the values we extract.
    """
    df_raw = pd.read_excel(raw_file)
    df_raw = df_raw.dropna(how="all").reset_index(drop=True)

    df_filtered = df_raw.iloc[4:].copy()
    df_filtered = df_filtered.rename(
        columns={
            "Unnamed: 1": "month_str",
            "Unnamed: 2": "commodity",
            "Unnamed: 8": "value",
        }
    )
    df_filtered = df_filtered[df_filtered["month_str"].notna() & df_filtered["commodity"].notna()]
    df_filtered["value"] = pd.to_numeric(df_filtered["value"], errors="coerce")

    def extract(label: str, col_name: str) -> pd.DataFrame:
        return df_filtered[df_filtered["commodity"] == label][["month_str", "value"]].rename(
            columns={"value": col_name}
        )

    general_cpi = extract("A) General Index", "cpi_general")
    food_cpi = extract("A.1) Food and beverages", "cpi_food")
    fuel_cpi = extract("A.5) Fuel and light", "cpi_fuel")

    df_cpi = general_cpi.merge(food_cpi, on="month_str").merge(fuel_cpi, on="month_str")

    df_cpi["date"] = pd.to_datetime(df_cpi["month_str"], format="%b-%Y", errors="coerce")
    df_cpi = df_cpi.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Core CPI = General Index minus the weighted Food & Fuel contribution.
    # See config.py note: this is a constant-scaled version of a properly
    # renormalized core index, which is fine for YoY % calculations below.
    df_cpi["cpi_core"] = df_cpi["cpi_general"] - (
        df_cpi["cpi_food"] * config.FOOD_WEIGHT + df_cpi["cpi_fuel"] * config.FUEL_WEIGHT
    )

    for col in ["cpi_general", "cpi_food", "cpi_fuel", "cpi_core"]:
        df_cpi[f"{col}_yoy"] = df_cpi[col].pct_change(12) * 100

    logger.info("Cleaned CPI data: %d rows (%s to %s)", len(df_cpi), df_cpi["date"].min().date(), df_cpi["date"].max().date())
    return df_cpi


def clean_bps(val) -> float:
    """Parse the MPC voting-decision column into a signed bps float.

    Handles every format observed in the raw file: 'P' (pause/hold), blank,
    '(+)25', '(-)50', bare '+'/'-' prefixed numbers, and plain numbers.
    Anything unparseable falls back to 0.0 (treated as a hold) rather than
    raising, since a handful of footnote rows can slip through the date filter.
    """
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    if val_str in ["P", "", "nan"]:
        return 0.0
    if "(+)" in val_str:
        return float(val_str.replace("(+)", "").strip())
    if "(-)" in val_str:
        return -float(val_str.replace("(-)", "").strip())
    if val_str.startswith("+"):
        return float(val_str[1:])
    if val_str.startswith("-"):
        return float(val_str)
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def load_and_clean_mpc(raw_file=config.MPC_RAW_FILE) -> pd.DataFrame:
    """Parse the raw MPC decisions Excel sheet into a tidy dataframe."""
    df_raw = pd.read_excel(raw_file)
    df_raw = df_raw.dropna(how="all").reset_index(drop=True)

    df_raw.columns = df_raw.iloc[2]
    df_raw = df_raw.iloc[3:].reset_index(drop=True)

    df_mpc = pd.DataFrame(
        {
            "date": df_raw.iloc[:, 1],
            "repo_rate": df_raw.iloc[:, 2],
            "raw_decision": df_raw.iloc[:, 3],
        }
    )

    df_mpc["date"] = pd.to_datetime(df_mpc["date"], errors="coerce")
    df_mpc = df_mpc.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    df_mpc["bps_change"] = df_mpc["raw_decision"].apply(clean_bps)
    df_mpc["repo_rate"] = pd.to_numeric(df_mpc["repo_rate"], errors="coerce")

    df_mpc["is_hike"] = (df_mpc["bps_change"] > 0).astype(int)
    df_mpc["decision"] = np.select(
        [df_mpc["bps_change"] > 0, df_mpc["bps_change"] < 0],
        ["hike", "cut"],
        default="hold",
    )

    logger.info("Parsed MPC decisions: %d meetings, distribution=%s", len(df_mpc), df_mpc["decision"].value_counts().to_dict())
    return df_mpc


def merge_cpi_mpc(df_cpi: pd.DataFrame, df_mpc: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of merge: each MPC meeting gets the most recent CPI print
    available *at that date* -- never a future one. Adds lag/rolling features
    used by the Stage 4 classifier.
    """
    merged = pd.merge_asof(
        df_mpc,
        df_cpi[["date", "cpi_general_yoy", "cpi_food_yoy", "cpi_fuel_yoy", "cpi_core_yoy"]],
        on="date",
        direction="backward",
    )

    merged["core_lag1"] = merged["cpi_core_yoy"].shift(1)
    merged["core_lag2"] = merged["cpi_core_yoy"].shift(2)
    merged["core_3m_avg"] = merged["cpi_core_yoy"].rolling(3).mean()

    logger.info("Merged CPI-MPC dataset: %d rows", len(merged))
    return merged


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Stage 1 pipeline: clean, validate, persist to CSV, and
    (best-effort) persist to the database.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_cpi = load_and_clean_cpi()
    validation.validate_cleaned_cpi(df_cpi)
    df_cpi.to_csv(config.CLEANED_CPI_FILE, index=False)

    df_mpc = load_and_clean_mpc()
    merged = merge_cpi_mpc(df_cpi, df_mpc)
    validation.validate_processed_cpi_mpc(merged)
    merged.to_csv(config.PROCESSED_CPI_MPC_FILE, index=False)

    logger.info("Stage 1 complete. Saved %s and %s", config.CLEANED_CPI_FILE, config.PROCESSED_CPI_MPC_FILE)

    # Database write is best-effort: CSVs remain the source of truth that
    # every downstream stage actually reads, so a missing/unreachable DB
    # (e.g. docker-compose not running locally) should not break the
    # pipeline -- just skip the DB write and continue.
    try:
        engine = db.get_engine()
        db.write_table(df_cpi, "cleaned_cpi", engine)
        db.write_table(merged, "processed_cpi_mpc", engine)
    except Exception as exc:
        logger.warning("Database write skipped (%s). CSV outputs are unaffected.", exc)

    return df_cpi, merged


if __name__ == "__main__":
    run()