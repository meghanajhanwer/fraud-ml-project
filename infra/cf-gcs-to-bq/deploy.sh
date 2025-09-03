#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="mythic-producer-471010-k0"
REGION="us-central1"
BUCKET="mythic-producer-471010-k0-gcs-to-bq"

FUNC_NAME="gcs-to-bq"
ENTRY_POINT="gcs_to_bq"
RUNTIME="python311"
MEMORY="512Mi"
TIMEOUT="300s"
BQ_DATASET="ingestion"
BQ_TABLE="bank_transactions_raw"
BQ_LOCATION="us-central1"  
FILE_PREFIX="incoming/"
DELIMITER=","
SKIP_ROWS="1"
WRITE_MODE="WRITE_APPEND"

echo ">> Project: ${PROJECT_ID} | Region: ${REGION} | Bucket: ${BUCKET}"
gcloud config set project "${PROJECT_ID}" >/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUNCTION_DIR="${SCRIPT_DIR}"
[[ -f "${FUNCTION_DIR}/main.py" ]] || { echo "ERROR: main.py not found in ${FUNCTION_DIR}"; exit 1; }
[[ -f "${FUNCTION_DIR}/requirements.txt" ]] || { echo "ERROR: requirements.txt not found in ${FUNCTION_DIR}"; exit 1; }

echo ">> Enabling required services"
gcloud services enable \
  cloudbuild.googleapis.com \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  eventarc.googleapis.com \
  storage.googleapis.com \
  pubsub.googleapis.com >/dev/null

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
EVENTARC_SA="service-${PROJECT_NUMBER}@gcp-sa-eventarc.iam.gserviceaccount.com"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo ">> Ensuring Eventarc service agent exists"
gcloud beta services identity create --service=eventarc.googleapis.com --project="${PROJECT_ID}" >/dev/null 2>&1 || true

echo ">> Granting minimal IAM (best-effort)"
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

echo ">> Deploying Cloud Function with GCS trigger"
gcloud functions deploy "${FUNC_NAME}" \
  --gen2 \
  --region="${REGION}" \
  --runtime="${RUNTIME}" \
  --source="${FUNCTION_DIR}" \
  --entry-point="${ENTRY_POINT}" \
  --memory="${MEMORY}" \
  --timeout="${TIMEOUT}" \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=${BUCKET}" \
  --trigger-location="${REGION}" \
  --set-env-vars=^:^BQ_DATASET=${BQ_DATASET}:BQ_TABLE=${BQ_TABLE}:BQ_LOCATION=${BQ_LOCATION}:FILE_PREFIX=${FILE_PREFIX}:DELIMITER=${DELIMITER}:SKIP_ROWS=${SKIP_ROWS}:WRITE_MODE=${WRITE_MODE}

echo ">> Deployed. Quick status:"
gcloud functions describe "${FUNC_NAME}" --region="${REGION}" --format="yaml(status,serviceConfig.service,environment)"
