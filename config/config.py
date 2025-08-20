PROJECT_ID = "resonant-idea-467410-u9"
REGION = "us-central1"
BQ_LOCATION = "us-central1"

# Buckets
INGEST_BUCKET = "resonant-idea-467410-u9-gcs-to-bq"
ARTIFACTS_GCS_PREFIX = f"gs://{INGEST_BUCKET}/artifacts"

# BigQuery
RAW_DATASET = "ingestion"
RAW_TABLE = "bank_transactions_raw"  # landing
CURATED_DATASET = "curated"
CURATED_TABLE = "bank_transactions_modelready"  # modeling-ready table

# Columns
ID_COL = "TransactionID"
TIME_COL = "TransactionDate"
TARGET_COL = "is_fraud"        # *** Assumed present ***
POSITIVE_CLASS = 1

# Modeling
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15   # train remainder is 0.70
SEQUENCE_LENGTH = 10

# Registrations / Endpoint
XGB_DISPLAY_NAME = "fraud-xgb"
LSTM_DISPLAY_NAME = "fraud-lstm"
NLP_DISPLAY_NAME = "fraud-nlp"
ENDPOINT_DISPLAY_NAME = "fraud-endpoint"

# Vertex AI serving containers (prebuilt)
SKLEARN_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
XGBOOST_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
TENSORFLOW_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"

# Compute
ENDPOINT_MACHINE_TYPE = "n1-standard-2"

# Split strategy: "random" (existing split_data), "group", or "time"
SPLIT_STRATEGY = "random"   # change to "group" or "time" when you want
