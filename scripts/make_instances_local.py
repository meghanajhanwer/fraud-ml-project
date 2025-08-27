import json
import numpy as np
from config.config import TARGET_COL, ID_COL, TIME_COL
from src.data_extraction import load_curated
from src.feature_engineering import add_simple_features
from src.preprocessing import split_data, build_tabular_transformer

def main(n=1, out_path="instances.json"):
    df = load_curated().sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("Curated table/view is empty.")
    df = add_simple_features(df)
    drop_cols = [TARGET_COL, ID_COL, TIME_COL, "PreviousTransactionDate", "nlp_text"]
    tabular_cols = [c for c in df.columns if c not in drop_cols]
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    preproc, _ = build_tabular_transformer(df[tabular_cols])
    preproc.fit(df_train[tabular_cols])
    source_df = df_val if len(df_val) >= n else df_train
    X = preproc.transform(source_df.iloc[:max(1, n)][tabular_cols])
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass

    instances = np.asarray(X).tolist()
    with open(out_path, "w") as f:
        json.dump({"instances": instances}, f)
    print(f"Wrote {out_path} | instances: {len(instances)} | feature_len: {len(instances[0])}")

if __name__ == "__main__":
    main(n=1, out_path="instances.json")
