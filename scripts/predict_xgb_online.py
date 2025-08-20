import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from google.cloud import aiplatform
from config.config import PROJECT_ID, REGION, TARGET_COL, ID_COL, TIME_COL
from src.data_extraction import load_curated
from src.feature_engineering import add_simple_features
from src.preprocessing import split_data, build_tabular_transformer

def get_endpoint():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    eps = list(aiplatform.Endpoint.list(filter='display_name="fraud-endpoint"'))
    if not eps:
        raise RuntimeError("Endpoint 'fraud-endpoint' not found.")
    return eps[0]

def main():
    # 1) Rebuild features exactly like training
    df = load_curated()
    df = df.sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    df_fe = add_simple_features(df)

    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df_fe)

    drop_cols = [TARGET_COL, ID_COL, TIME_COL, "PreviousTransactionDate", "nlp_text"]
    tabular_cols = [c for c in df_fe.columns if c not in drop_cols]
    preproc, _ = build_tabular_transformer(df_fe[tabular_cols])

    X_train = preproc.fit_transform(df_train[tabular_cols])
    X_val   = preproc.transform(df_val[tabular_cols])

    # 2) Take one instance from validation
    import numpy as np
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    instance = X_val[0].tolist()

    # 3) Call endpoint
    ep = get_endpoint()
    preds = ep.predict(instances=[instance])
    print("Prediction response:")
    print(preds)

if __name__ == "__main__":
    main()
