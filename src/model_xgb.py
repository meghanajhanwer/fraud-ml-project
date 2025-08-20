from typing import Dict, Any, Tuple
import tempfile, os, json
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
)
from xgboost import XGBClassifier
from google.cloud import storage
from config.config import RANDOM_SEED, ARTIFACTS_GCS_PREFIX, PROJECT_ID

def _load_best_params_if_any():
    """Try to load tuned params from gs://.../artifacts/models/xgb/best_params.json."""
    try:
        bucket_name = ARTIFACTS_GCS_PREFIX.split("gs://",1)[1].split("/",1)[0]
        prefix      = ARTIFACTS_GCS_PREFIX.split(bucket_name + "/", 1)[1]
        blob = storage.Client(project=PROJECT_ID).bucket(bucket_name)\
            .blob(f"{prefix}/models/xgb/best_params.json")
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception:
        pass
    return None

def _upload_used_params(params: dict):
    try:
        bucket_name = ARTIFACTS_GCS_PREFIX.split("gs://",1)[1].split("/",1)[0]
        prefix      = ARTIFACTS_GCS_PREFIX.split(bucket_name + "/", 1)[1]
        storage.Client(project=PROJECT_ID).bucket(bucket_name)\
            .blob(f"{prefix}/models/xgb/used_params.json")\
            .upload_from_string(json.dumps(params, indent=2))
    except Exception:
        pass

def train_eval_xgb(X_train, y_train, X_val, y_val) -> Tuple[Dict[str, Any], Any]:
    tuned = _load_best_params_if_any()
    if tuned:
        # Ensure critical safety defaults remain
        base = dict(tree_method="hist", n_jobs=2, random_state=RANDOM_SEED, reg_lambda=1.0)
        base.update(tuned)
        params = base
    else:
        params = dict(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_SEED, tree_method="hist", n_jobs=2,
            eval_metric="aucpr",
        )

    model = XGBClassifier(**params)
    # Early stopping when we have a validation set
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=30)
    except Exception:
        model.fit(X_train, y_train, verbose=False)

    # Predictions + metrics
    y_pred = model.predict(X_val)
    try:
        y_proba = model.predict_proba(X_val)[:, 1]
    except Exception:
        y_proba = None

    pr, rc, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_val, y_pred)
    metrics = {
        "model_type": "xgb",
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_val, y_proba)) if y_proba is not None else 0.0,
        "pr_auc": float(average_precision_score(y_val, y_proba)) if y_proba is not None else 0.0,
    }

    # Save the Booster as model.bst under artifacts/models/xgb/
    booster = model.get_booster()
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "model.bst")
        booster.save_model(lp)
        bucket_name = ARTIFACTS_GCS_PREFIX.split("gs://",1)[1].split("/",1)[0]
        prefix      = ARTIFACTS_GCS_PREFIX.split(bucket_name + "/", 1)[1]
        storage.Client(project=PROJECT_ID).bucket(bucket_name)\
            .blob(f"{prefix}/models/xgb/model.bst").upload_from_filename(lp)

    # Save the params actually used (for traceability)
    _upload_used_params(params)

    return metrics, model
