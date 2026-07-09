"""
Shared configuration for the CPI-MPC pipeline: file paths and constants.
Centralized here so every stage references the same values instead of
hardcoding weights/paths in multiple places.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "raw_data"
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

CPI_RAW_FILE = RAW_DATA_DIR / "CPI-Base-2012.xlsx"
MPC_RAW_FILE = RAW_DATA_DIR / "mpc_decisions.xlsx"

CLEANED_CPI_FILE = DATA_DIR / "cleaned_cpi.csv"
PROCESSED_CPI_MPC_FILE = DATA_DIR / "processed_cpi_mpc.csv"

# --- CPI basket weights (Base 2012=100, published RBI/MOSPI combined weights) --
# Core CPI = General Index excluding Food & Fuel groups.
# NOTE: subtracting (weight * sub-index) from the general index does NOT
# renormalize to a 0-100 base -- the resulting "cpi_core" level is scaled by
# a constant factor (1 - FOOD_WEIGHT - FUEL_WEIGHT). This does not affect the
# YoY % change calculation (pct_change is invariant to constant scaling), but
# `cpi_core` itself should not be read as a "base=100" index. It's kept this
# way to match the original project methodology.
FOOD_WEIGHT = 0.4563
FUEL_WEIGHT = 0.0666

# --- Feature columns used by the Stage 4 models -------------------------
FEATURE_COLUMNS = [
    "cpi_general_yoy",
    "cpi_food_yoy",
    "cpi_fuel_yoy",
    "cpi_core_yoy",
    "core_lag1",
    "core_lag2",
    "core_3m_avg",
]

TARGET_COLUMN = "is_hike"

RANDOM_STATE = 42