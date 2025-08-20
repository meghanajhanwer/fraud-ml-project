import pandas as pd
from config.config import TARGET_COL, ID_COL, TIME_COL
from src.data_extraction import load_curated
from src.eda import save_class_balance_plot, profile_basic
from src.feature_engineering import add_simple_features
from src.preprocessing import split_data, build_tabular_transformer, smote_fit_resample
from src.model_xgb import train_eval_xgb
from src.model_nlp import train_eval_nlp
from src.evaluation import save_metrics_comparison
from src.visualization import save_confusion, save_roc, save_pr
from src.deploy import register_and_deploy_best
from config.config import SPLIT_STRATEGY
from src.preprocessing import split_data
from src.preprocessing import split_data_group_by_account, split_data_time_based

# Optional legacy LSTM (kept as placeholder; will be zeros if not implemented)
LSTM_AVAILABLE = False
try:
    from src.model_lstm import train_eval_lstm  # only if you later add a real sequence LSTM
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

def main():
    # 1) Load curated
    df = load_curated()
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Expected target column '{TARGET_COL}' not found in curated table.")

    # 2) EDA
    profile_basic(df)
    save_class_balance_plot(df, TARGET_COL)

    # 3) Feature Engineering
    df = df.sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    df_fe = add_simple_features(df)

    # 4) Splits
    
    if SPLIT_STRATEGY == "group":
        df_train, df_val, df_test, y_train, y_val, y_test = split_data_group_by_account(df_fe)
    elif SPLIT_STRATEGY == "time":
        df_train, df_val, df_test, y_train, y_val, y_test = split_data_time_based(df_fe)
    else:
        df_train, df_val, df_test, y_train, y_val, y_test = split_data(df_fe)


    # 5) Preprocess tabular
    drop_cols = [TARGET_COL, ID_COL, TIME_COL, "PreviousTransactionDate", "nlp_text"]
    tabular_cols = [c for c in df_fe.columns if c not in drop_cols]
    preproc, _ = build_tabular_transformer(df_fe[tabular_cols])
    X_train = preproc.fit_transform(df_train[tabular_cols])
    X_val   = preproc.transform(df_val[tabular_cols])

    # 6) SMOTE on train
    X_train_bal, y_train_bal = smote_fit_resample(X_train, y_train)

    metrics_all = []

    # 7) XGBoost
    xgb_metrics, xgb_model = train_eval_xgb(X_train_bal, y_train_bal, X_val, y_val)
    try:
        xgb_proba = xgb_model.predict_proba(X_val)[:,1]
    except Exception:
        xgb_proba = None
    xgb_pred = xgb_model.predict(X_val)
    save_confusion(y_val, xgb_pred, "xgb")
    save_roc(y_val, xgb_proba, "xgb")
    save_pr(y_val, xgb_proba, "xgb")
    metrics_all.append(xgb_metrics)

    # 8) NLP (TF-IDF + Logistic Regression on synthesized text)
    train_text = df_train["nlp_text"].fillna("")
    val_text   = df_val["nlp_text"].fillna("")
    nlp_metrics, nlp_model = train_eval_nlp(train_text, y_train, val_text, y_val)
    try:
        nlp_proba = nlp_model.predict_proba(val_text)[:,1]
    except Exception:
        nlp_proba = None
    nlp_pred = nlp_model.predict(val_text)
    save_confusion(y_val, nlp_pred, "nlp")
    save_roc(y_val, nlp_proba, "nlp")
    save_pr(y_val, nlp_proba, "nlp")
    metrics_all.append(nlp_metrics)

    # 9) Optional LSTM placeholder (zeros) to keep 3 rows if needed
    if not LSTM_AVAILABLE:
        metrics_all.append({"model_type":"lstm","precision":0,"recall":0,"f1":0,"roc_auc":0,"pr_auc":0})

    # 10) Compare & deploy best
    save_metrics_comparison(metrics_all)
    register_and_deploy_best()

if __name__ == "__main__":
    main()
