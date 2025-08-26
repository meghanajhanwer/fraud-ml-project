# scripts/run_cross_domain.py
import os, json, tempfile
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse

from config.config import (
    PROJECT_ID, ARTIFACTS_GCS_BASE,
    TARGET_COL, ID_COL, TIME_COL, RANDOM_SEED
)
from src.data_extraction import _load_primary_curated   # primary loader
from src.data_extraction_ulb import load_ulb_dataframe  # ulb loader
import xgboost as xgb


# ---------- GCS helpers ----------
def _save_json_gcs(obj, gcs_uri: str):
    assert gcs_uri.startswith("gs://")
    bucket, path = gcs_uri[5:].split("/", 1)
    b = storage.Client(project=PROJECT_ID).bucket(bucket).blob(path)
    b.upload_from_string(json.dumps(obj, indent=2), content_type="application/json")


# ---------- Feature selection (numeric-only, intersection) ----------
def _select_safe_numeric_columns(train_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 drop_cols: List[str]) -> List[str]:
    # numeric columns in train, excluding drop_cols
    num_train = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_train = [c for c in num_train if c not in drop_cols]
    # keep only those that also exist in test and are numeric there too
    safe = []
    for c in num_train:
        if c in test_df.columns and pd.api.types.is_numeric_dtype(test_df[c]):
            safe.append(c)
    return sorted(safe)


# ---------- Preprocessor: numeric only ----------
def _build_numeric_preproc(feature_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
    ])
    preproc = ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
        n_jobs=None,
    )
    return preproc


# ---------- Train on TRAIN, return (preproc, model, used_cols, val_metrics) ----------
def _train_on(train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              train_name: str) -> Tuple[ColumnTransformer, xgb.XGBClassifier, List[str], Dict[str, Any]]:
    # Drop columns not used for tabular numeric modelling
    drop_cols = [TARGET_COL, ID_COL, TIME_COL, "PreviousTransactionDate", "nlp_text"]

    # Pick numeric-only columns present in both train and test
    feature_cols = _select_safe_numeric_columns(train_df, test_df, drop_cols)
    if not feature_cols:
        raise RuntimeError("No common numeric feature columns found between train and test.")

    # Split train into train/val (simple random split; cross-domain script is for external evaluation)
    # To keep memory light, do a 80/20 split inline
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED, stratify=train_df[TARGET_COL])
    y_tr = tr[TARGET_COL].values
    y_va = va[TARGET_COL].values

    preproc = _build_numeric_preproc(feature_cols)
    X_tr = preproc.fit_transform(tr[feature_cols])
    X_va = preproc.transform(va[feature_cols])

    # Decide SMOTE usage: skip on large datasets (e.g., ULB) to avoid OOM
    use_smote = len(tr) < 120_000  # heuristic: skip if very large
    if use_smote:
        sm = SMOTE(random_state=RANDOM_SEED)
        if issparse(X_tr):
            X_tr = X_tr.toarray()
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    # Class weighting if still imbalanced
    pos = int(np.sum(y_tr == 1))
    neg = int(np.sum(y_tr == 0))
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0
    base_params = dict(
        n_estimators=200 if len(tr) > 120_000 else 300,
        max_depth=4 if len(tr) > 120_000 else 6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=0,
    )
    # apply scale_pos_weight when imbalance is notable
    if spw > 3.0:
        base_params["scale_pos_weight"] = spw

    model = xgb.XGBClassifier(**base_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Validation metrics
    proba_va = model.predict_proba(X_va)[:, 1]
    pred_va = (proba_va >= 0.5).astype(int)
    val_metrics = dict(
        accuracy=float(accuracy_score(y_va, pred_va)),
        precision=float(precision_score(y_va, pred_va, zero_division=0)),
        recall=float(recall_score(y_va, pred_va, zero_division=0)),
        f1=float(f1_score(y_va, pred_va, zero_division=0)),
        roc_auc=float(roc_auc_score(y_va, proba_va)) if len(np.unique(y_va)) > 1 else 0.0,
        pr_auc=float(average_precision_score(y_va, proba_va)) if len(np.unique(y_va)) > 1 else 0.0,
        support=int(y_va.sum()),
        n=int(len(y_va)),
    )
    return preproc, model, feature_cols, val_metrics


# ---------- Evaluate trained model on TEST (cross-domain) ----------
def _eval_on(preproc: ColumnTransformer,
             model: xgb.XGBClassifier,
             feature_cols: List[str],
             test_df: pd.DataFrame) -> Dict[str, Any]:
    # Ensure all feature columns exist in test; if missing, fill with zeros
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0
    X_te = preproc.transform(test_df[feature_cols])
    y_te = test_df[TARGET_COL].values

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return dict(
        accuracy=float(accuracy_score(y_te, pred)),
        precision=float(precision_score(y_te, pred, zero_division=0)),
        recall=float(recall_score(y_te, pred, zero_division=0)),
        f1=float(f1_score(y_te, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else 0.0,
        pr_auc=float(average_precision_score(y_te, proba)) if len(np.unique(y_te)) > 1 else 0.0,
        support=int(y_te.sum()),
        n=int(len(y_te)),
    )


# ---------- Dataset loader ----------
def _load_by_name(name: str) -> pd.DataFrame:
    if name == "primary":
        return _load_primary_curated()
    elif name == "ulb":
        return load_ulb_dataframe()
    else:
        raise ValueError(name)


# ---------- Main ----------
def main():
    pairs = [("primary", "ulb"), ("ulb", "primary")]
    results = {}

    for train_name, test_name in pairs:
        print(f"=== Train on {train_name} → Test on {test_name} ===")
        df_train = _load_by_name(train_name)
        df_test  = _load_by_name(test_name)

        # Train with numeric-only intersection features
        preproc, model, cols, val_metrics = _train_on(df_train, df_test, train_name)
        test_metrics = _eval_on(preproc, model, cols, df_test)

        key = f"{train_name}_to_{test_name}"
        results[key] = {"val": val_metrics, "test": test_metrics}

        pair_uri = f"{ARTIFACTS_GCS_BASE}/cross_domain/{key}.json"
        _save_json_gcs(results[key], pair_uri)
        print(f"Saved {pair_uri}")

    summary_uri = f"{ARTIFACTS_GCS_BASE}/cross_domain/summary.json"
    _save_json_gcs(results, summary_uri)
    print(f"✅ Saved {summary_uri}")


if __name__ == "__main__":
    main()
