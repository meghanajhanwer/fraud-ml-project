import os, os.path
PROJECT_ID = "resonant-idea-467410-u9"
REGION = "us-central1"
BQ_LOCATION = "us-central1"
RAW_DATASET = "ingestion"
RAW_TABLE = "bank_transactions_raw"
CURATED_DATASET = "curated"
CURATED_TABLE = "bank_transactions_modelready"
BUCKET = "resonant-idea-467410-u9-gcs-to-bq"
ARTIFACTS_GCS_BASE = f"gs://{BUCKET}/artifacts"
DATASET = os.getenv("DATASET", "primary").lower()
ARTIFACTS_GCS_PREFIX = f"{ARTIFACTS_GCS_BASE}/datasets/{DATASET}"
ULB_CSV_LOCAL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "creditcard.csv")
ULB_GCS_PATH  = f"gs://{BUCKET}/external/ulb/creditcard.csv"
TARGET_COL = "is_fraud"
ID_COL     = "TransactionID"
TIME_COL   = "TransactionDate"
VAL_SIZE    = 0.15
TEST_SIZE   = 0.15
RANDOM_SEED = 42
SPLIT_STRATEGY = os.getenv("SPLIT_STRATEGY", "random").lower()
LSTM_SEQ_LEN = 10
SMOTE_POLICY = os.getenv("SMOTE_POLICY", "auto").lower()
XGB_LIGHT = os.getenv("XGB_LIGHT", "1") == "1"

SKLEARN_IMAGE_URI  = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
XGBOOST_IMAGE_URI  = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-0:latest"
TENSORFLOW_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-14:latest"

XGB_DISPLAY_NAME = "fraud-xgb"
NLP_DISPLAY_NAME = "fraud-nlp"
LSTM_DISPLAY_NAME = "fraud-lstm"

ENDPOINT_DISPLAY_NAME   = "fraud-endpoint"
ENDPOINT_MACHINE_TYPE   = "n1-standard-2"

def artifacts_prefix_for(ds: str) -> str:
    return f"{ARTIFACTS_GCS_BASE}/datasets/{ds}"
