import os
import json
import tempfile
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, average_precision_score
import xgboost as xgb
from google.cloud import storage
from config.config import PROJECT_ID, ARTIFACTS_GCS_PREFIX, XGB_LIGHT

def _upload_bytes(data: bytes, gcs_path: str, content_type="application/octet-stream"):
    assert gcs_path.startswith("gs://")
    _, rest = gcs_path.split("gs://", 1)
    bucket, blob = rest.split("/", 1)
    storage.Client(project=PROJECT_ID).bucket(bucket).blob(blob).upload_from_string(
        data, content_type=content_type
    )

def _save_text(txt: str, rel: str):
    _upload_bytes(txt.encode(), f"{ARTIFACTS_GCS_PREFIX}/{rel}", "application/json")

def _metrics_dict(y_true, y_pred, y_proba) -> Dict[str, Any]:
    has_pos = len(np.unique(y_true)) > 1
    return {
        "model_type": "xgb",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if has_pos else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if has_pos else 0.0,
    }

def train_eval_xgb(X_tr, y_tr, X_va, y_va) -> Tuple[Dict[str, Any], xgb.XGBClassifier]:
    pos = int(np.sum(y_tr == 1))
    neg = int(np.sum(y_tr == 0))
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    params = dict(
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
    if spw > 3.0:
        params["scale_pos_weight"] = spw

    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    try:
        proba = model.predict_proba(X_va)[:, 1]
    except Exception:
        proba = model.predict(X_va)
        if getattr(proba, "ndim", 1) > 1:
            proba = proba[:, -1]
    pred = (proba >= 0.5).astype(int)
    metrics = _metrics_dict(y_va, pred, proba)
    booster = model.get_booster()
    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "model.bst")
        booster.save_model(local)
        with open(local, "rb") as f:
            _upload_bytes(
                f.read(),
                f"{ARTIFACTS_GCS_PREFIX}/models/xgb/model.bst",
                "application/octet-stream",
            )

    _save_text(json.dumps({"params": params, "class_ratio": {"neg": neg, "pos": pos}}, indent=2),
               "models/xgb_params.json")
    return metrics, model
