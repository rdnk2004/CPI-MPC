"""
Data quality validation using pandera. Run after Stage 1 to catch structural
or range issues before they propagate into Stages 2-4 (e.g. a raw-file format
change silently producing NaNs, duplicate rows, or an out-of-range value).

Usage:
    from src.validation import validate_cleaned_cpi, validate_processed_cpi_mpc
    validate_cleaned_cpi(df_cpi)   # raises pandera.errors.SchemaError on failure
"""

import logging

import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

logger = logging.getLogger(__name__)

# YoY inflation figures for this dataset historically range roughly -6% to 15%
# (see outputs/yearly_cpi_components.csv); -20/30 gives headroom without
# letting a genuinely broken parse (e.g. an index value of 0) slip through.
_YOY_RANGE = Check.in_range(-20, 30, ignore_na=True)

cleaned_cpi_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, unique=True, nullable=False),
        "cpi_general": Column(float, Check.gt(0), nullable=False),
        "cpi_food": Column(float, Check.gt(0), nullable=False),
        "cpi_fuel": Column(float, Check.gt(0), nullable=False),
        "cpi_core": Column(float, nullable=False),
        "cpi_general_yoy": Column(float, _YOY_RANGE, nullable=True),
        "cpi_food_yoy": Column(float, _YOY_RANGE, nullable=True),
        "cpi_fuel_yoy": Column(float, _YOY_RANGE, nullable=True),
        "cpi_core_yoy": Column(float, _YOY_RANGE, nullable=True),
    },
    strict=False,  # allow extra columns (e.g. core_trend, added later by Stage 3)
    coerce=True,
)

processed_cpi_mpc_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, unique=True, nullable=False),
        # RBI repo rate has never been outside roughly 3-10% in the modern
        # inflation-targeting era; a value outside this range signals a
        # parsing error, not a real policy rate.
        "repo_rate": Column(float, Check.in_range(2, 12), nullable=False),
        "bps_change": Column(float, Check.in_range(-200, 200), nullable=False),
        "is_hike": Column(int, Check.isin([0, 1]), nullable=False),
        "decision": Column(str, Check.isin(["hike", "cut", "hold"]), nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_cleaned_cpi(df) -> None:
    cleaned_cpi_schema.validate(df)
    logger.info("cleaned_cpi passed schema validation (%d rows).", len(df))


def validate_processed_cpi_mpc(df) -> None:
    processed_cpi_mpc_schema.validate(df)
    logger.info("processed_cpi_mpc passed schema validation (%d rows).", len(df))