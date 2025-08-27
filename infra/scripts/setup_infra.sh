#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="resonant-idea-467410-u9"
REGION="us-central1"
BQ_LOCATION="us-central1"
BUCKET="resonant-idea-467410-u9-gcs-to-bq"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_JSON="${SCRIPT_DIR}/schema.json"
SQL_CURATE="${SCRIPT_DIR}/../sql/curate_transactions.sql"

echo "Setting gcloud project"
gcloud config set project "${PROJECT_ID}"

echo "Enabling APIs"
gcloud services enable \
  storage.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com \
  cloudfunctions.googleapis.com \
  eventarc.googleapis.com \
  run.googleapis.com \
  logging.googleapis.com \
  pubsub.googleapis.com \
  aiplatform.googleapis.com

echo "Creating GCS bucket (if not exists)"
gsutil mb -l "${REGION}" "gs://${BUCKET}" || true
gsutil cp /dev/null "gs://${BUCKET}/incoming/.keep" || true
gsutil cp /dev/null "gs://${BUCKET}/artifacts/.keep" || true

echo "Creating BigQuery datasets"
bq --location="${BQ_LOCATION}" mk -d ingestion || true
bq --location="${BQ_LOCATION}" mk -d curated || true

echo "Creating RAW table (if not exists)"
bq ls ingestion | grep -q bank_transactions_raw || \
bq mk --table ingestion.bank_transactions_raw "${SCHEMA_JSON}"

echo "Creating curated modeling table"
bq query --use_legacy_sql=false < "${SQL_CURATE}"

echo "Setting IAM for Eventarc, GCS Pub/Sub, and Cloud Functions runtime"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-eventarc.iam.gserviceaccount.com" \
  --role="roles/storage.legacyBucketReader" || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher" || true

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer" || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/bigquery.jobUser" || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor" || true

echo "Infra setup complete."
