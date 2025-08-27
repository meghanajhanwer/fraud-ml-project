import io
import pandas as pd
from google.cloud import bigquery, storage
from config.config import PROJECT_ID, BQ_LOCATION,DATASET, ARTIFACTS_GCS_PREFIX,RAW_DATASET, CURATED_DATASET, CURATED_TABLE,TARGET_COL, ID_COL, TIME_COL,ULB_CSV_LOCAL, ULB_GCS_PATH
from .data_extraction_ulb import load_ulb_dataframe

def _load_primary_curated() -> pd.DataFrame:
    """Load your existing curated BigQuery dataset/view."""
    bqclient = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    sql = f"SELECT * FROM `{PROJECT_ID}.{CURATED_DATASET}.{CURATED_TABLE}`"
    df = bqclient.query(sql).result().to_dataframe()
    return df

def load_curated() -> pd.DataFrame:
    """
    Unified loader. Uses DATASET from config:
      - 'primary' => BigQuery curated
      - 'ulb'     => ULB CSV adapter (local or GCS)
    """
    if DATASET == "primary":
        return _load_primary_curated()
    elif DATASET == "ulb":
        return load_ulb_dataframe()
    else:
        raise ValueError(f"Unknown DATASET '{DATASET}'. Use 'primary' or 'ulb'.")
