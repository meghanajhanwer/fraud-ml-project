# src/data_extraction_ulb.py
import io
import os
import pandas as pd
from google.cloud import storage
from config.config import (
    PROJECT_ID, TARGET_COL, ID_COL, TIME_COL,
    ULB_CSV_LOCAL, ULB_GCS_PATH
)

def _read_csv_local_or_gcs() -> pd.DataFrame:
    """Try local CSV first; fall back to GCS path."""
    if os.path.exists(ULB_CSV_LOCAL):
        return pd.read_csv(ULB_CSV_LOCAL)

    # Read from GCS
    assert ULB_GCS_PATH.startswith("gs://")
    _, rest = ULB_GCS_PATH.split("gs://", 1)
    bucket_name, blob_name = rest.split("/", 1)
    client = storage.Client(project=PROJECT_ID)
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(
            f"ULB CSV not found at {ULB_GCS_PATH}. Put file at {ULB_CSV_LOCAL} or upload to GCS."
        )
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))

def load_ulb_dataframe() -> pd.DataFrame:
    """
    Map ULB columns to your canonical minimal schema.
    ULB columns: Time, V1..V28, Amount, Class (1=fraud)
    We create:
      - TARGET_COL = 'is_fraud' from Class
      - ID_COL     = synthetic 'ULB_<row>'
      - TIME_COL   = Timestamp from 'Time' seconds offset
      - TransactionAmount from 'Amount'
      - AccountID  = synthetic grouping
      - PreviousTransactionDate (optional for FE; safe if missing)
      - keep V1..V28 as numerics
    """
    df = _read_csv_local_or_gcs()

    # Basic renames
    df = df.rename(columns={
        "Amount": "TransactionAmount",
        "Class": TARGET_COL,
    })

    # Create required IDs and timestamps
    df[ID_COL] = (df.index).astype(str).map(lambda s: f"ULB_{s}")
    # ULB 'Time' is seconds since first transaction in the dataset
    base = pd.Timestamp("2019-01-01")
    df[TIME_COL] = base + pd.to_timedelta(df["Time"], unit="s")
    # Synthetic account buckets (optional; useful for grouping)
    df["AccountID"] = (df.index % 10000).astype(str).map(lambda s: f"ULB_ACC_{s}")

    # Optional: previous timestamp per account
    df = df.sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    df["PreviousTransactionDate"] = df.groupby("AccountID")[TIME_COL].shift(1)

    # Ensure target is int (0/1)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Create empty NLP text column to keep pipeline happy if it exists
    if "nlp_text" not in df.columns:
        df["nlp_text"] = ""

    return df
