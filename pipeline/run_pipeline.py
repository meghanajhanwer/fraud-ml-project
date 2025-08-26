# pipeline/run_pipeline.py
import os
import warnings
import pandas as pd

# Import the config module once; always reference as CFG.<name>
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

# Optional LSTM (placeholder)
LSTM_AVAILABLE = False
try:
    from src.model_lstm import train_eval_lstm  # noqa: F401
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False


def main():
    warnings.filterwarnings("ignore")

    # Allow skipping Vertex AI deploy (useful for ULB / research runs)
    SKIP_DEPLOY = os.getenv("SKIP_DEPLOY", "0") == "1"

    print(
        f"=== Fraud pipeline start | DATASET={CFG.DATASET} | "
        f"SPLIT_STRATEGY={CFG.SPLIT_STRATEGY} | SKIP_DEPLOY={SKIP_DEPLOY} ==="
    )

    # 1) Load curated dataframe (dataset-aware under the hood)
    df = load_curated()
    if CFG.TARGET_COL not in df.columns:
        raise RuntimeError(
            f"Expected target column '{CFG.TARGET_COL}' not found in curated data."
        )
    print(
        f"Loaded rows: {len(df):,} | cols: {len(df.columns)} | "
        f"positives: {int(df[CFG.TARGET_COL].sum()):,}"
    )

    # 2) EDA artifacts
    try:
        profile_basic(df)
    except Exception as e:
        print(f"[EDA] profile_basic skipped: {e}")
    try:
        save_class_balance_plot(df, CFG.TARGET_COL)
    except Exception as e:
        print(f"[EDA] class balance plot skipped: {e}")

    # 3) Feature Engineering (order by account+time if available)
    if "AccountID" in df.columns and CFG.TIME_COL in df.columns:
        df = df.sort_values(["AccountID", CFG.TIME_COL]).reset_index(drop=True)
    elif CFG.TIME_COL in df.columns:
        df = df.sort_values([CFG.TIME_COL]).reset_index(drop=True)

    df_fe = add_simple_features(df)

    # 4) Splits
    ss = (CFG.SPLIT_STRATEGY or "random").lower()
    if ss == "group":
        df_train, df_val, df_test, y_train, y_val, y_test = split_data_group_by_account(
            df_fe
        )
        print("[Split] Group-by-AccountID split used.")
    elif ss == "time":
        df_train, df_val, df_test, y_train, y_val, y_test = split_data_time_based(
            df_fe
        )
        print("[Split] Time-based split used.")
    else:
        df_train, df_val, df_test, y_train, y_val, y_test = split_data(df_fe)
        print("[Split] Random stratified split used.")

    # 5) Preprocess (tabular)  ———  UPDATED: save preprocessor + feature names to GCS
    drop_cols = [CFG.TARGET_COL, CFG.ID_COL, CFG.TIME_COL, "PreviousTransactionDate", "nlp_text"]

    # Initialize to avoid UnboundLocalError on non-ULB datasets
    high_card = []

    # Avoid OHE blow-ups on ULB: drop high-cardinality categoricals (e.g., synthetic AccountID ~10k cats)
    if CFG.DATASET == "ulb":
        obj_cols = df_fe.select_dtypes(include=["object", "category"]).columns.tolist()
        high_card = [c for c in obj_cols if df_fe[c].nunique(dropna=True) > 100]
        if high_card:
            print(f"[Preproc] Dropping high-cardinality categoricals for ULB: {high_card}")
            drop_cols.extend(high_card)

    tabular_cols = [c for c in df_fe.columns if c not in drop_cols]
    preproc, _ = build_tabular_transformer(df_fe[tabular_cols])

    X_train = preproc.fit_transform(df_train[tabular_cols])
    X_val   = preproc.transform(df_val[tabular_cols])
    # X_test  = preproc.transform(df_test[tabular_cols])  # keep for extras if needed

    # --- NEW: save preprocessor + feature names to GCS for serving ---
    try:
        from google.cloud import storage
        import io, json, joblib

        # Bytes for the joblib object
        buf = io.BytesIO()
        joblib.dump(preproc, buf)
        buf.seek(0)

        # Helper to upload bytes/text to GCS
        def _upload_bytes(data: bytes, uri: str, content_type: str | None = None):
            assert uri.startswith("gs://"), f"Bad GCS URI: {uri}"
            bucket_name, blob_path = uri[5:].split("/", 1)
            client = storage.Client(project=CFG.PROJECT_ID)
            blob = client.bucket(bucket_name).blob(blob_path)
            blob.upload_from_file(io.BytesIO(data), rewind=True, content_type=content_type)

        def _upload_text(txt: str, uri: str, content_type: str = "application/json"):
            _upload_bytes(txt.encode("utf-8"), uri, content_type)

        preproc_joblib_uri = f"{CFG.ARTIFACTS_GCS_PREFIX}/preproc/preprocessor.joblib"
        featnames_uri      = f"{CFG.ARTIFACTS_GCS_PREFIX}/preproc/feature_names.json"
        columns_in_uri     = f"{CFG.ARTIFACTS_GCS_PREFIX}/preproc/columns_in.json"

        # Upload preprocessor
        _upload_bytes(buf.getvalue(), preproc_joblib_uri, "application/octet-stream")

        # Feature names in transformed space (may be empty if older sklearn)
        try:
            feature_names = preproc.get_feature_names_out().tolist()
        except Exception:
            feature_names = []
        _upload_text(json.dumps(feature_names), featnames_uri)

        # Columns expected on input to the preprocessor
        try:
            columns_in = preproc.feature_names_in_.tolist()
        except Exception:
            columns_in = tabular_cols
        _upload_text(json.dumps(columns_in), columns_in_uri)

        print(f"[Preproc] Saved preprocessor + names to {CFG.ARTIFACTS_GCS_PREFIX}/preproc/")
    except Exception as e:
        print(f"[Preproc] Save preprocessor skipped: {e}")

    # 6) SMOTE policy (dataset-aware)
    use_smote = (CFG.SMOTE_POLICY == "always") or (
        CFG.SMOTE_POLICY == "auto" and CFG.DATASET != "ulb"
    )
    if use_smote:
        print(f"[SMOTE] enabled (policy={CFG.SMOTE_POLICY}, dataset={CFG.DATASET})")
        X_train_bal, y_train_bal = smote_fit_resample(X_train, y_train)
    else:
        print(f"[SMOTE] disabled (policy={CFG.SMOTE_POLICY}, dataset={CFG.DATASET}); using class weighting in XGB")
        X_train_bal, y_train_bal = X_train, y_train

    metrics_all = []

    # 7) XGBoost
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
        print(f"[Viz] XGB plots skipped: {e}")

    metrics_all.append(xgb_metrics)

    # 8) NLP (TF-IDF + Logistic Regression) — safely skip if it fails
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
            print(f"[Viz] NLP plots skipped: {e}")

        metrics_all.append(nlp_metrics)
    except Exception as e:
        print(f"[NLP] Skipped due to error: {e}")

    # 9) Optional LSTM placeholder (zeros) if not available
    if not LSTM_AVAILABLE:
        metrics_all.append({
            "model_type": "lstm",
            "accuracy": 0, "precision": 0, "recall": 0,
            "f1": 0, "roc_auc": 0, "pr_auc": 0
        })

    # 10) Compare & (optionally) deploy best
    try:
        save_metrics_comparison(metrics_all)
    except Exception as e:
        print(f"[Compare] metrics comparison save skipped: {e}")

    if not SKIP_DEPLOY:
        try:
            from src.deploy import register_and_deploy_best  # lazy import
            register_and_deploy_best()
        except Exception as e:
            print(f"[Deploy] Deployment failed: {e}")
            raise
    else:
        print("⚠️  SKIP_DEPLOY=1 set: skipping Vertex AI registration/deployment.")

    print("=== Pipeline completed successfully. ===")


if __name__ == "__main__":
    main()
