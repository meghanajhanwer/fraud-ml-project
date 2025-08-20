# GCS → Cloud Function → BigQuery → Curated → ML (XGB, LSTM, NLP) → Vertex AI

Hardcoded for project: `resonant-idea-467410-u9` (region/location `us-central1`).

## Flow
GCS (`incoming/`) ➜ Cloud Function (Gen2) ➜ BigQuery `ingestion.bank_transactions_raw` ➜ Curated table `curated.bank_transactions_modelready` ➜ EDA/FE/Preprocessing (SMOTE on train only) ➜ Train XGBoost, LSTM, TF-IDF+LogReg ➜ Compare ➜ Register best model ➜ Deploy to Vertex AI endpoint.

## Quickstart
```bash
# 1) Infra
cd infra
bash setup_infra.sh    # enables APIs, creates bucket/datasets/tables, curated table

# 2) Deploy ingestion function
cd ../cloud_function
bash deploy.sh

# 3) Ingest a CSV
gsutil cp ~/bank_transactions_data.csv gs://resonant-idea-467410-u9-gcs-to-bq/incoming/bank_transactions_data.csv
gcloud functions logs read gcs-to-bq --region=us-central1 --gen2 --limit=50

# 4) Create virtual env and install deps
cd ..
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 5) Run pipeline (pulls from curated table)
python pipeline/run_pipeline.py
