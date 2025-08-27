import os
from google.cloud import storage
from config.config import PROJECT_ID, ULB_CSV_LOCAL, ULB_GCS_PATH

def main():
    if not os.path.exists(ULB_CSV_LOCAL):
        raise SystemExit(f"Local CSV not found: {ULB_CSV_LOCAL}")

    assert ULB_GCS_PATH.startswith("gs://")
    _, rest = ULB_GCS_PATH.split("gs://", 1)
    bucket, blob_name = rest.split("/", 1)

    client = storage.Client(project=PROJECT_ID)
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_filename(ULB_CSV_LOCAL)
    print(f"Uploaded {ULB_CSV_LOCAL} → {ULB_GCS_PATH}")

if __name__ == "__main__":
    main()
