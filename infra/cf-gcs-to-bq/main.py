import os
from google.cloud import bigquery

BQ_DATASET   = os.getenv("BQ_DATASET", "ingestion")
BQ_TABLE     = os.getenv("BQ_TABLE", "bank_transactions_raw")
BQ_LOCATION  = os.getenv("BQ_LOCATION", "us-central1")
FILE_PREFIX  = os.getenv("FILE_PREFIX", "incoming/")
DELIMITER    = os.getenv("DELIMITER", ",")
SKIP_ROWS    = int(os.getenv("SKIP_ROWS", "1"))
WRITE_MODE   = os.getenv("WRITE_MODE", "WRITE_APPEND")

bq = bigquery.Client()

def load_csv_to_bq(uri: str, table_id: str) -> int:
    job_cfg = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        autodetect=False,
        skip_leading_rows=SKIP_ROWS,
        field_delimiter=DELIMITER,
        quote_character='"',
        allow_quoted_newlines=True,
        write_disposition=WRITE_MODE,
    )
    job = bq.load_table_from_uri(uri, table_id, job_config=job_cfg, location=BQ_LOCATION)
    job.result()
    return job.output_rows or 0

def gcs_to_bq(event, context):
    bucket = event.get("bucket")
    name   = event.get("name")
    gen    = event.get("generation")

    if not bucket or not name:
        print("Missing bucket/name in event")
        return

    if FILE_PREFIX and not name.startswith(FILE_PREFIX):
        print(f"Skip (prefix): {name}")
        return
    if not (name.endswith(".csv") or name.endswith(".csv.gz")):
        print(f"Skip (not CSV): {name}")
        return

    table_id = f"{bq.project}.{BQ_DATASET}.{BQ_TABLE}"
    uri = f"gs://{bucket}/{name}"
    rows = load_csv_to_bq(uri, table_id)
    print(f"[OK] Loaded {rows} rows from {uri} (gen={gen}) → {table_id}")
