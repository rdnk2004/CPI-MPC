"""
Database connection layer. Defaults to the docker-compose Postgres service,
overridable via the DATABASE_URL environment variable so the same code works
against Postgres in production/docker and SQLite in local testing.

Usage:
    from src.db import get_engine, write_table
    engine = get_engine()
    write_table(df_cpi, "cleaned_cpi", engine)
"""

import logging
import os

import pandas as pd
from sqlalchemy import Engine, create_engine

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "postgresql+psycopg2://cpi_user:cpi_pass@localhost:5432/cpi_mpc"


def get_engine() -> Engine:
    """Create a SQLAlchemy engine from DATABASE_URL, or fall back to the
    default local docker-compose Postgres connection string."""
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    engine = create_engine(url)
    logger.info("Connected to database: %s", engine.url.render_as_string(hide_password=True))
    return engine


def write_table(df: pd.DataFrame, table_name: str, engine: Engine) -> None:
    """Write a dataframe to a table, replacing it fully each run.

    Full-replace (not append) is intentional here -- each pipeline run
    re-derives cleaned_cpi and processed_cpi_mpc from the raw Excel files
    from scratch, so the table should always mirror the latest run exactly
    rather than accumulate duplicate historical rows.
    """
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    logger.info("Wrote %d rows to table '%s'", len(df), table_name)


def read_table(table_name: str, engine: Engine) -> pd.DataFrame:
    return pd.read_sql_table(table_name, engine)