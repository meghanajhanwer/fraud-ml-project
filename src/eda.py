import io
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import storage
from config.config import ARTIFACTS_GCS_PREFIX, PROJECT_ID

def _gcs_upload_bytes(content: bytes, gcs_uri: str):
    assert gcs_uri.startswith("gs://")
    _, path = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path.split("/", 1)
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)

def save_class_balance_plot(df: pd.DataFrame, target_col: str, title: str = "Class Balance"):
    counts = df[target_col].value_counts().sort_index()
    fig = plt.figure()
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0)
    _gcs_upload_bytes(buf.read(), f"{ARTIFACTS_GCS_PREFIX}/eda/class_balance.png")

def profile_basic(df: pd.DataFrame, n_head: int = 5):
    # Save simple head and separate numeric/categorical describes
    _gcs_upload_bytes(df.head(n_head).to_csv(index=False).encode(),
                      f"{ARTIFACTS_GCS_PREFIX}/eda/head.csv")

    num = df.select_dtypes(include=["number"])
    if not num.empty:
        _gcs_upload_bytes(num.describe().to_csv().encode(),
                          f"{ARTIFACTS_GCS_PREFIX}/eda/describe_numeric.csv")

    cat = df.select_dtypes(exclude=["number"])
    if not cat.empty:
        _gcs_upload_bytes(cat.describe().to_csv().encode(),
                          f"{ARTIFACTS_GCS_PREFIX}/eda/describe_categorical.csv")
