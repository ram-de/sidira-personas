"""Snowflake reader. Env-var driven."""
from __future__ import annotations

import os
import pandas as pd


def read_snowflake(database: str, schema: str, table: str) -> pd.DataFrame:
    import snowflake.connector
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "ADHOC"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
        database=database,
        schema=schema,
    )
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {database}.{schema}.{table}")
        df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df
    finally:
        conn.close()


def read_local(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
