import json, os, tempfile
import numpy as np
import pandas as pd
import joblib
from google.cloud import storage
from config.config import PROJECT_ID, ARTIFACTS_GCS_PREFIX,TARGET_COL, ID_COL, TIME_COL
from src.data_extraction import load_curated
from src.feature_engineering import add_simple_features

def _download_preprocessor(local_path: str):
    assert ARTIFACTS_GCS_PREFIX.startswith("gs://")
    bucket = ARTIFACTS_GCS_PREFIX.split("gs://",1)[1].split("/",1)[0]
    prefix = ARTIFACTS_GCS_PREFIX.split(bucket + "/", 1)[1]
    blob = storage.Client(project=PROJECT_ID).bucket(bucket).blob(
        f"{prefix}/preprocessor/preprocessor.joblib"
    )
    if not blob.exists():
        raise SystemExit("Preprocessor not found in GCS")
    blob.download_to_filename(local_path)

def main(n=1, out="instances.json"):
    df = load_curated()
    if df.empty:
        raise SystemExit("Curated tableis empty")
    df = df.sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    df = add_simple_features(df)
    drop_cols = [TARGET_COL, ID_COL, TIME_COL, "PreviousTransactionDate", "nlp_text"]
    tabular_cols = [c for c in df.columns if c not in drop_cols]
    with tempfile.TemporaryDirectory() as td:
        pre_path = os.path.join(td, "preprocessor.joblib")
        _download_preprocessor(pre_path)
        pre = joblib.load(pre_path)

        X = pre.transform(df.iloc[:max(1,n)][tabular_cols])
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                X = X.toarray()
        except Exception:
            pass

    instances = X.tolist()
    with open(out, "w") as f:
        json.dump({"instances": instances}, f)
    print(f"Wrote {out} | instances: {len(instances)} | feature_len: {len(instances[0])}")

if __name__ == "__main__":
    main(n=1, out="instances.json")
