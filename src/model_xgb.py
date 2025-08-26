# src/model_xgb.py
import os
import json
import tempfile
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
import xgboost as xgb
from google.cloud import storage

from config.config import (
    PROJECT_ID, BUCKET, ARTIFACTS_GCS_PREFIX, DATASET, XGB_LIGHT
)

def _upload_bytes(data: bytes, gcs_path: str, content_type="application/octet-stream"):
    assert gcs_path.startswith("gs://")
    _, rest = gcs_path.split("gs://", 1)
    bucket, blob = rest.split("/", 1)
    client = storage.Client(project=PROJECT_ID)
    b = client.bucket(bucket).blob(blob)
    b.upload_from_string(data, content_type=content_type)

def _save_text(txt: str, rel: str):
    _upload_bytes(txt.encode(), f"{ARTIFACTS_GCS_PREFIX}/{rel}", "text/plain")

def _metrics_dict(y_true, y_pred, y_proba) -> Dict[str, Any]:
    return {
        "model_type": "xgb",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
    }

def train_eval_xgb(X_tr, y_tr, X_va, y_va) -> Tuple[Dict[str, Any], xgb.XGBClassifier]:
    # If upstream disabled SMOTE, set scale_pos_weight = #neg/#pos on the *training* fold.
    pos = int(np.sum(y_tr == 1))
    neg = int(np.sum(y_tr == 0))
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    # Light defaults for free tier (hist grows fast & is memory friendly)
    base_params = dict(
        n_estimators=300 if not XGB_LIGHT else 200,
        max_depth=6 if not XGB_LIGHT else 4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=0,
    )

    # If SMOTE was used upstream, classes are closer → scale_pos_weight ~1 is fine.
    # We can detect by class ratio; if very imbalanced, apply spw.
    ratio = spw  # reuse computed value
    use_spw = ratio > 3.0  # simple heuristic
    if use_spw:
        base_params["scale_pos_weight"] = ratio

    model = xgb.XGBClassifier(**base_params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    try:
        proba = model.predict_proba(X_va)[:, 1]
    except Exception:
        proba = model.predict(X_va)
        if proba.ndim > 1:
            proba = proba[:, -1]
    pred = (proba >= 0.5).astype(int)

    metrics = _metrics_dict(y_va, pred, proba)

    # Save booster as .bst for Vertex AI XGBoost container
    booster = model.get_booster()
    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "xgb_model.bst")
        booster.save_model(local)
        with open(local, "rb") as f:
            _upload_bytes(f.read(), f"{ARTIFACTS_GCS_PREFIX}/models/xgb_model.bst", "application/octet-stream")

    # Also drop a params JSON
    _save_text(json.dumps({"params": base_params, "class_ratio": {"neg": neg, "pos": pos}}, indent=2),
               "models/xgb_params.json")

    return metrics, model
