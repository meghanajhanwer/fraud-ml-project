from typing import Optional
import pandas as pd
from google.cloud import bigquery
from google.cloud import bigquery_storage
from config.config import PROJECT_ID, CURATED_DATASET, CURATED_TABLE, BQ_LOCATION

def load_curated(limit: Optional[int] = None, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Reads from curated table in BigQuery using the Storage API for efficiency.
    Optionally selects subset of columns and/or limits rows.
    """
    table_fqn = f"`{PROJECT_ID}.{CURATED_DATASET}.{CURATED_TABLE}`"
    if columns:
        select_cols = ", ".join([f"`{c}`" for c in columns])
    else:
        select_cols = "*"

    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
    SELECT {select_cols}
    FROM {table_fqn}
    {limit_clause}
    """
    bqclient = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    bqsclient = bigquery_storage.BigQueryReadClient()
    df = bqclient.query(sql).result().to_dataframe(bqstorage_client=bqsclient)
    return df
