import json
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from google.cloud import storage
from config.config import PROJECT_ID, ARTIFACTS_GCS_PREFIX, RANDOM_SEED,TARGET_COL, ID_COL, TIME_COL
from src.data_extraction import load_curated
from src.feature_engineering import add_simple_features
from src.preprocessing import split_data, build_tabular_transformer, smote_fit_resample
def _upload_json(obj, gcs_path):
    assert gcs_path.startswith("gs://")
    bucket_name = gcs_path.split("gs://",1)[1].split("/",1)[0]
    blob_name   = gcs_path.split(bucket_name + "/", 1)[1]
    storage.Client(project=PROJECT_ID).bucket(bucket_name).blob(blob_name)\
        .upload_from_string(json.dumps(obj, indent=2))

def main():
    df = load_curated().sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    df = add_simple_features(df)
    df_train, df_val, *_ = split_data(df)

    drop_cols   = [TARGET_COL, ID_COL, TIME_COL, "PreviousTransactionDate", "nlp_text"]
    tabular_cols = [c for c in df.columns if c not in drop_cols]

    preproc, _ = build_tabular_transformer(df[tabular_cols])
    X_train = preproc.fit_transform(df_train[tabular_cols])
    X_val   = preproc.transform(df_val[tabular_cols])
    y_train = df_train[TARGET_COL]
    y_val   = df_val[TARGET_COL]
    X_train_bal, y_train_bal = smote_fit_resample(X_train, y_train)
    grid = {
        "max_depth":        [3, 5, 6],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "n_estimators":     [300],
    }

    trials = []
    best = None
    for md in grid["max_depth"]:
        for eta in grid["learning_rate"]:
            for ss in grid["subsample"]:
                for cs in grid["colsample_bytree"]:
                    params = dict(
                        n_estimators=300,
                        max_depth=md,
                        learning_rate=eta,
                        subsample=ss,
                        colsample_bytree=cs,
                        reg_lambda=1.0,
                        random_state=RANDOM_SEED,
                        tree_method="hist",
                        n_jobs=2,
                        eval_metric="aucpr",
                    )
                    model = XGBClassifier(**params)
                    model.fit(X_train_bal, y_train_bal,
                              eval_set=[(X_val, y_val)],
                              verbose=False,
                              early_stopping_rounds=30)
                    y_pred = model.predict(X_val)
                    pr, rc, f1, _ = precision_recall_fscore_support(
                        y_val, y_pred, average="binary", zero_division=0
                    )
                    trial = {"params": params, "precision": float(pr), "recall": float(rc), "f1": float(f1)}
                    trials.append(trial)
                    if best is None or trial["f1"] > best["f1"]:
                        best = trial
    _upload_json({"best": best, "trials": trials}, f"{ARTIFACTS_GCS_PREFIX}/metrics/xgb_tuning_results.json")
    _upload_json(best["params"], f"{ARTIFACTS_GCS_PREFIX}/models/xgb/best_params.json")
    print("XGB tuning complete, Best params saved")
    print("Best:", best)

if __name__ == "__main__":
    main()
