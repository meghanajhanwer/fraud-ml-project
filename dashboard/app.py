import io
import streamlit as st; st.write("Dashboard loaded")
import json
from typing import Optional, List
import pandas as pd
import streamlit as st
from google.cloud import storage

PROJECT_ID = "mythic-producer-471010-k0"
BUCKET = "mythic-producer-471010-k0-gcs-to-bq"
ARTIFACTS_PREFIX = "artifacts"
def _client():
    return storage.Client(project=PROJECT_ID)
def gcs_text(uri: str) -> Optional[str]:
    try:
        assert uri.startswith("gs://")
        _, path = uri.split("gs://", 1)
        bucket_name, blob_name = path.split("/", 1)
        blob = _client().bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            return None
        return blob.download_as_text()
    except Exception:
        return None

def gcs_bytes(uri: str) -> Optional[bytes]:
    try:
        assert uri.startswith("gs://")
        _, path = uri.split("gs://", 1)
        bucket_name, blob_name = path.split("/", 1)
        blob = _client().bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    except Exception:
        return None

def gcs_csv(uri: str) -> Optional[pd.DataFrame]:
    raw = gcs_bytes(uri)
    if not raw:
        return None
    return pd.read_csv(io.BytesIO(raw))

def show_image_if_exists(title: str, uri: str, width: int = 700):
    img = gcs_bytes(uri)
    if img:
        st.subheader(title)
        st.image(img, width=width)
    else:
        st.info(f"Not found: `{uri}`")

def show_table_if_exists(title: str, uri: str, is_json=False):
    st.subheader(title)
    if is_json:
        txt = gcs_text(uri)
        if txt:
            try:
                data = json.loads(txt)
                st.json(data)
                return data
            except Exception:
                st.code(txt)
        else:
            st.info(f"Not found: `{uri}`")
        return None
    else:
        df = gcs_csv(uri)
        if df is not None and not df.empty:
            st.dataframe(df)
        else:
            st.info(f"Not found or empty: `{uri}`")
        return df

st.set_page_config(page_title="Fraud ML Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")
st.caption(f"Project: **{PROJECT_ID}** — Artifacts from **gs://{BUCKET}/{ARTIFACTS_PREFIX}**")

st.header("Model Metrics & Leader")
metrics_uri = f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/metrics/metrics.json"
metrics = show_table_if_exists("Raw Metrics JSON", metrics_uri, is_json=True)

show_image_if_exists("F1 Comparison", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/metrics/f1_comparison.png", width=600)

if isinstance(metrics, dict):
    import itertools
    rows = []
    for model_type, vals in metrics.items():
        row = {"model_type": model_type}
        row.update(vals)
        rows.append(row)
    if rows:
        dfm = pd.DataFrame(rows)
        st.subheader("Metrics Table")
        st.dataframe(dfm)
        best_row = dfm.sort_values("f1", ascending=False).head(1)
        if not best_row.empty:
            best = best_row.iloc[0]["model_type"]
            st.success(f"Best model by F1: **{best}**")

st.header("EDA")
show_image_if_exists("Class Balance", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/eda/class_balance.png", width=500)
show_table_if_exists("Head (first rows)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/eda/head.csv")
col1, col2 = st.columns(2)
with col1:
    show_table_if_exists("Describe (numeric)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/eda/describe_numeric.csv")
with col2:
    show_table_if_exists("Describe (categorical)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/eda/describe_categorical.csv")

st.header("Evaluation Plots")
plots = [
    ("Confusion (XGB)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/plots/confusion_xgb.png"),
    ("ROC (XGB)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/plots/roc_xgb.png"),
    ("PR (XGB)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/plots/pr_xgb.png"),
    ("Confusion (NLP)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/plots/confusion_nlp.png"),
    ("ROC (NLP)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/plots/roc_nlp.png"),
    ("PR (NLP)", f"gs://{BUCKET}/{ARTIFACTS_PREFIX}/plots/pr_nlp.png"),
]
for title, uri in plots:
    show_image_if_exists(title, uri, width=700)

st.caption("If a plot/table is missing, re-run the training pipeline to regenerate artifacts.")
