#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="mythic-producer-471010-k0"
REGION="us-central1"        
BQ_LOCATION="us-central1"   
BUCKET="mythic-producer-471010-k0-gcs-to-bq"

echo ">> Using project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}" >/dev/null
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          
SCHEMA_JSON="${INFRA_DIR}/bq/schema.json"
SQL_CURATE="${INFRA_DIR}/bq/curate_transactions.sql"

[ -f "${SCHEMA_JSON}" ] || { echo "FATAL: Missing schema file: ${SCHEMA_JSON}"; exit 1; }
[ -f "${SQL_CURATE}" ]   || { echo "FATAL: Missing SQL file:    ${SQL_CURATE}"; exit 1; }

echo ">> Enabling required APIs"
gcloud services enable \
  storage.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com \
  cloudfunctions.googleapis.com \
  eventarc.googleapis.com \
  run.googleapis.com \
  logging.googleapis.com \
  pubsub.googleapis.com \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com >/dev/null

echo ">> Creating GCS bucket"
gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1 || gsutil mb -l "${REGION}" "gs://${BUCKET}"
gsutil cp /dev/null "gs://${BUCKET}/incoming/.keep"  >/dev/null 2>&1 || true
gsutil cp /dev/null "gs://${BUCKET}/artifacts/.keep" >/dev/null 2>&1 || true

echo ">> Creating BigQuery datasets"
bq --location="${BQ_LOCATION}" mk -d ingestion >/dev/null 2>&1 || true
bq --location="${BQ_LOCATION}" mk -d curated   >/dev/null 2>&1 || true

echo ">> Creating RAW table"
if ! bq ls ingestion | grep -q "^ *bank_transactions_raw *$"; then
  bq mk --table --location="${BQ_LOCATION}" \
    "${PROJECT_ID}:ingestion.bank_transactions_raw" \
    "${SCHEMA_JSON}"
else
  echo "   RAW table already exists: ${PROJECT_ID}:ingestion.bank_transactions_raw"
fi

echo ">> Creating curated modeling table via SQL"
bq query --location="${BQ_LOCATION}" --use_legacy_sql=false < "${SQL_CURATE}"

echo ">> Setting minimal IAM for Eventarc & function runtime (best-effort)"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
EVENTARC_SA="service-${PROJECT_NUMBER}@gcp-sa-eventarc.iam.gserviceaccount.com"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud beta services identity create --service=eventarc.googleapis.com --project="${PROJECT_ID}" >/dev/null 2>&1 || true

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${EVENTARC_SA}" \
  --role="roles/eventarc.serviceAgent" >/dev/null || true

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${EVENTARC_SA}" \
  --role="roles/storage.objectViewer" >/dev/null || true

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.objectViewer" >/dev/null || true
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/bigquery.jobUser" >/dev/null || true
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/bigquery.dataEditor" >/dev/null || true
echo ">> Infra setup complete."
