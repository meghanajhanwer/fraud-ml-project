#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="resonant-idea-467410-u9"
REGION="us-central1"
BUCKET="resonant-idea-467410-u9-gcs-to-bq"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUNCTION_DIR="${SCRIPT_DIR}" 
[[ -f "${FUNCTION_DIR}/main.py" ]] || { echo "ERROR: main.py not found in ${FUNCTION_DIR}"; exit 1; }
[[ -f "${FUNCTION_DIR}/requirements.txt" ]] || { echo "ERROR: requirements.txt not found in ${FUNCTION_DIR}"; exit 1; }
gcloud config set project "${PROJECT_ID}"

gcloud functions deploy gcs-to-bq \
  --gen2 \
  --runtime=python311 \
  --region="${REGION}" \
  --entry-point=gcs_to_bq \
  --memory=512Mi \
  --timeout=300s \
  --source="${FUNCTION_DIR}" \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=${BUCKET}" \
  --set-env-vars=^:^BQ_DATASET=ingestion:BQ_TABLE=bank_transactions_raw:BQ_LOCATION=us-central1:FILE_PREFIX=incoming/:DELIMITER=,:SKIP_ROWS=1:WRITE_MODE=WRITE_APPEND

echo "Cloud Function deployed from ${FUNCTION_DIR}"
