#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="resonant-idea-467410-u9"
REGION="us-central1"
BUCKET="resonant-idea-467410-u9-gcs-to-bq"

gcloud config set project "${PROJECT_ID}"

gcloud functions deploy gcs-to-bq \
  --gen2 \
  --runtime=python311 \
  --region="${REGION}" \
  --entry-point=gcs_to_bq \
  --memory=512Mi \
  --timeout=300s \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=${BUCKET}" \
  --set-env-vars=^:^BQ_DATASET=ingestion:BQ_TABLE=bank_transactions_raw:BQ_LOCATION=us-central1:FILE_PREFIX=incoming/:DELIMITER=,:SKIP_ROWS=1:WRITE_MODE=WRITE_APPEND

echo "Cloud Function deployed."
