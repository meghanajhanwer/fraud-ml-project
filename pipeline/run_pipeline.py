import os
import io
import json
import warnings
import pandas as pd
from typing import Dict, Any, Tuple
import config.config as CFG
from src.data_extraction import load_curated
from src.eda import save_class_balance_plot, profile_basic
from src.feature_engineering import add_simple_features
from src.preprocessing import (
    split_data,
    split_data_group_by_account,
    split_data_time_based,
    build_tabular_transformer,
    smote_fit_resample,
)
from src.model_xgb import train_eval_xgb
from src.model_nlp import train_eval_nlp
from src.evaluation import save_metrics_comparison
from src.visualization import save_confusion, save_roc, save_pr
from sklearn.metrics import confusion_matrix
from google.cloud import storage
import joblib

def _gcs_client():
    return storage.Client(project=CFG.PROJECT_ID)

def _upload_bytes(b: bytes, uri: str, content_type: str = "application/octet-stream") -> None:
    assert uri.startswith("gs://")
    bucket, path = uri[5:].split("/", 1)
    _gcs_client().bucket(bucket).blob(path).upload_from_file(io.BytesIO(b), rewind=True, content_type=content_type)

def _upload_json(obj: Any, uri: str) -> None: 
    _upload_bytes(json.dumps(obj, indent=2).encode("utf-8"), uri, "application/json")

def _sf(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

def _pick_threshold(metrics: Dict[str, Any], default: float = 0.5) -> float:
    for k in ("best_threshold", "threshold", "thr", "opt_threshold"):
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                pass
    return default

def main():
    warnings.filterwarnings("ignore")
    SKIP_DEPLOY = os.getenv("SKIP_DEPLOY", "0") == "1"

    print(
        f"Fraud pipeline start | DATASET={CFG.DATASET} | "
        f"SPLIT_STRATEGY={CFG.SPLIT_STRATEGY} | SKIP_DEPLOY={SKIP_DEPLOY}"
    )
    #EDA
    df = load_curated()
    if CFG.TARGET_COL not in df.columns:
        raise RuntimeError(f"Expected target column '{CFG.TARGET_COL}' not found.")
    print(
        f"Loaded rows: {len(df):,} | cols: {len(df.columns)} | "
        f"positives: {int(df[CFG.TARGET_COL].sum()):,}"
    )

    try:
        profile_basic(df)
    except Exception as e:
        print(f"profile_basic skipped: {e}")

    try:
        save_class_balance_plot(df, CFG.TARGET_COL)
    except Exception as e:
        print(f"class balance skipped: {e}")

    #Order for time/group splits
    if "AccountID" in df.columns and CFG.TIME_COL in df.columns:
        df = df.sort_values(["AccountID", CFG.TIME_COL]).reset_index(drop=True)
    elif CFG.TIME_COL in df.columns:
        df = df.sort_values([CFG.TIME_COL]).reset_index(drop=True)

    df_fe = add_simple_features(df)

    #Split
    ss = (CFG.SPLIT_STRATEGY or "random").lower()
    if ss == "group":
        df_train, df_val, df_test, y_train, y_val, y_test = split_data_group_by_account(df_fe)
        print("Group-by-AccountID split.")
    elif ss == "time":
        df_train, df_val, df_test, y_train, y_val, y_test = split_data_time_based(df_fe)
        print("Time-based split.")
    else:
        df_train, df_val, df_test, y_train, y_val, y_test = split_data(df_fe)
        print("Random (stratified) split.")

    #Columns & preprocessor
    drop_cols = [CFG.TARGET_COL, CFG.ID_COL, CFG.TIME_COL, "PreviousTransactionDate", "nlp_text"]
    LEAKY = [
        "flag_high_amount",
        "flag_high_ratio",
        "flag_many_logins",
        "flag_night",
        "flag_rapid_repeat_high",
        "flag_weekend_high",
        "risk_score",
    ]
    drop_cols.extend([c for c in LEAKY if c in df_fe.columns])

    if CFG.DATASET == "ulb":
        obj_cols = df_fe.select_dtypes(include=["object", "category"]).columns.tolist()
        hi = [c for c in obj_cols if df_fe[c].nunique(dropna=True) > 100]
        if hi:
            print(f"Dropping high-cardinality cols (ULB): {hi}")
            drop_cols.extend(hi)

    tabular_cols = [c for c in df_fe.columns if c not in drop_cols]
    preproc, _ = build_tabular_transformer(df_fe[tabular_cols])
    X_train = preproc.fit_transform(df_train[tabular_cols])
    X_val   = preproc.transform(df_val[tabular_cols])

    # Save preprocessor
    try:
        base = f"{CFG.ARTIFACTS_GCS_PREFIX}/preproc"
        buf = io.BytesIO(); joblib.dump(preproc, buf); buf.seek(0)
        _upload_bytes(buf.getvalue(), f"{base}/preprocessor.joblib")
        try:
            _upload_json(preproc.get_feature_names_out().tolist(), f"{base}/feature_names.json")
        except Exception:
            _upload_json([], f"{base}/feature_names.json")

        try:
            _upload_json(preproc.feature_names_in_.tolist(), f"{base}/columns_in.json")
        except Exception:
            _upload_json(tabular_cols, f"{base}/columns_in.json")

        print(f"Preprocessor saved to {base}/")
    except Exception as e:
        print(f"Save preprocessor skipped: {e}")

    #Balance (SMOTE) or class weighting
    use_smote = (CFG.SMOTE_POLICY == "always") or (CFG.SMOTE_POLICY == "auto" and CFG.DATASET != "ulb")
    if use_smote:
        print(f"SMOTE enabled (policy={CFG.SMOTE_POLICY}, dataset={CFG.DATASET})")
        X_train_bal, y_train_bal = smote_fit_resample(X_train, y_train)
    else:
        print(f"SMOTE disabled (policy={CFG.SMOTE_POLICY}, dataset={CFG.DATASET}); using class weighting in XGB")
        X_train_bal, y_train_bal = X_train, y_train

    #Train models
    metrics_all_list = []          
    per_model: Dict[str, Dict[str, Any]] = {} 
    thresholds_payload: Dict[str, Dict[str, Any]] = {}

    #XGB Model
    xgb_metrics, xgb_model = train_eval_xgb(X_train_bal, y_train_bal, X_val, y_val)
    try:
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
    except Exception:
        xgb_proba = None
    xgb_pred = xgb_model.predict(X_val)
    try:
        save_confusion(y_val, xgb_pred, "xgb")
        save_roc(y_val, xgb_proba, "xgb")
        save_pr(y_val, xgb_proba, "xgb")
    except Exception as e:
        print(f"XGB plots skipped: {e}")
    metrics_all_list.append(xgb_metrics)

    # Persist per-model metrics JSON
    try:
        tn, fp, fn, tp = confusion_matrix(y_val, xgb_pred, labels=[0,1]).ravel()
        m = {
            "name": "xgb",
            "roc_auc": _sf(xgb_metrics.get("roc_auc") or xgb_metrics.get("roc")),
            "pr_auc": _sf(xgb_metrics.get("pr_auc") or xgb_metrics.get("pr")),
            "f1": _sf(xgb_metrics.get("f1")),
            "precision": _sf(xgb_metrics.get("precision")),
            "recall": _sf(xgb_metrics.get("recall")),
            "accuracy": _sf(xgb_metrics.get("accuracy")),
            "threshold": _pick_threshold(xgb_metrics, 0.5),
            "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        per_model["xgb"] = m
        thresholds_payload["xgb"] = {"best_threshold": m["threshold"]}
        _upload_json(m, f"{CFG.ARTIFACTS_GCS_PREFIX}/metrics/xgb.json")
    except Exception as e:
        print(f"xgb.json save skipped: {e}")

    #NLP Model
    try:
        train_text = df_train.get("nlp_text", pd.Series([""] * len(df_train))).fillna("")
        val_text   = df_val.get("nlp_text", pd.Series([""] * len(df_val))).fillna("")
        nlp_metrics, nlp_model = train_eval_nlp(train_text, y_train, val_text, y_val)

        try:
            nlp_proba = nlp_model.predict_proba(val_text)[:, 1]
        except Exception:
            nlp_proba = None
        nlp_pred = nlp_model.predict(val_text)
        try:
            save_confusion(y_val, nlp_pred, "nlp")
            save_roc(y_val, nlp_proba, "nlp")
            save_pr(y_val, nlp_proba, "nlp")
        except Exception as e:
            print(f"NLP plots skipped: {e}")

        metrics_all_list.append(nlp_metrics)

        # per-model JSON
        try:
            tn, fp, fn, tp = confusion_matrix(y_val, nlp_pred, labels=[0,1]).ravel()
            m = {
                "name": "nlp",
                "roc_auc": _sf(nlp_metrics.get("roc_auc") or nlp_metrics.get("roc")),
                "pr_auc": _sf(nlp_metrics.get("pr_auc") or nlp_metrics.get("pr")),
                "f1": _sf(nlp_metrics.get("f1")),
                "precision": _sf(nlp_metrics.get("precision")),
                "recall": _sf(nlp_metrics.get("recall")),
                "accuracy": _sf(nlp_metrics.get("accuracy")),
                "threshold": _pick_threshold(nlp_metrics, 0.5),
                "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }
            per_model["nlp"] = m
            thresholds_payload["nlp"] = {"best_threshold": m["threshold"]}
            _upload_json(m, f"{CFG.ARTIFACTS_GCS_PREFIX}/metrics/nlp.json")
        except Exception as e:
            print(f"nlp.json save skipped: {e}")
    except Exception as e:
        print(f"NLP skipped: {e}")

    #Legacy combined metrics
    try:
        save_metrics_comparison(metrics_all_list)
    except Exception as e:
        print(f"metrics comparison save skipped: {e}")

    #thresholds.json
    try:
        _upload_json(thresholds_payload, f"{CFG.ARTIFACTS_GCS_PREFIX}/metrics/thresholds.json")
    except Exception as e:
        print(f"thresholds.json save skipped: {e}")

    #best.json
    try:
        if per_model:
            def keyer(m: Dict[str, Any]) -> Tuple[float, float, float]:
                return (
                    _sf(m.get("pr_auc"), -1.0) or -1.0,
                    _sf(m.get("f1"), -1.0) or -1.0,
                    _sf(m.get("roc_auc"), -1.0) or -1.0,
                )
            best_name = max(per_model.keys(), key=lambda k: keyer(per_model[k]))
            best = per_model[best_name]
            criterion = "pr_auc"
            if best.get("pr_auc") in (None,):
                criterion = "f1" if best.get("f1") is not None else "roc_auc"
            _upload_json({"by": criterion, "name": best_name}, f"{CFG.ARTIFACTS_GCS_PREFIX}/metrics/best.json")
    except Exception as e:
        print(f"best.json save skipped: {e}")

    #Deploy best
    if not SKIP_DEPLOY:
        try:
            from src.deploy import register_and_deploy_best
            register_and_deploy_best()
        except Exception as e:
            print(f"[Deploy] Deployment failed: {e}")
            raise
    else:
        print("Skipping Vertex AI deployment.")

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
