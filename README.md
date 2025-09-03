Fraud ML on GCP — End-to-End

GCS → Cloud Function (Gen2) → BigQuery (ingestion → curated) → ML (XGBoost, LSTM, NLP) → Vertex AI → FastAPI Dashboard (Cloud Run)

This project ingests CSV transactions from Cloud Storage, lands them in BigQuery, curates a modeling table, trains multiple models (XGBoost, LSTM/LATM, and a TF-IDF+LogReg “NLP” model), evaluates/compares them, uploads artifacts to GCS, and optionally deploys the best model to a Vertex AI endpoint. A FastAPI dashboard reads artifacts from GCS and calls the Vertex endpoint for live predictions. The dashboard can be run locally or on Cloud Run.

Architecture -
┌────────────┐         ┌────────────────┐     ┌─────────────────────┐
│  CSV files │  put    │  Cloud Storage │     │  Eventarc (Cloud    │
│ incoming/  ├────────►│  bucket (GCS)  │────►│  Storage finalised) │
└────────────┘         └────────────────┘     └─────────┬───────────┘
                                                         │
                                                 triggers│
                                                         ▼
                                                ┌────────────────────┐
                                                │ Cloud Functions v2 │
                                                │  (gcs-to-bq)       │
                                                └─────────┬──────────┘
                                                          │ bq load
                                                          ▼
       ┌──────────────────────────┐       ┌───────────────────────────┐
       │ BigQuery dataset:        │       │ BigQuery dataset:         │
       │  ingestion               │       │  curated                  │
       │  bank_transactions_raw   │──────►│  bank_transactions_...    │
       └──────────────────────────┘  SQL  └───────────────────────────┘

                  ┌────────────────────────────────────────────────┐
                  │  pipeline/run_pipeline.py                      │
                  │  - EDA/FE/SMOTE                                │
                  │  - Train XGB, LSTM, NLP                        │
                  │  - Save metrics/plots to GCS                   │
                  │  - Deploy best to Vertex AI (optional)         │
                  └────────────────────────────────────────────────┘
                                   │
                             artifacts in GCS
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │ FastAPI Dashboard (app) │
                     │ - reads metrics/plots    │
                     │ - live predict via       │
                     │   Vertex AI endpoint     │
                     └─────────┬────────────────┘
                               │
                               ▼
                         Cloud Run (optional)
