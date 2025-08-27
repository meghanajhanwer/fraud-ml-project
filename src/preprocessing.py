from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from config.config import TARGET_COL, RANDOM_SEED, TEST_SIZE, VAL_SIZE,TIME_COL, ID_COL

def split_data(df: pd.DataFrame):
    df_trainval, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_SEED
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_ratio, stratify=df_trainval[TARGET_COL], random_state=RANDOM_SEED
    )
    return df_train, df_val, df_test, df_train[TARGET_COL], df_val[TARGET_COL], df_test[TARGET_COL]

def build_tabular_transformer(df: pd.DataFrame, drop_cols=None) -> Tuple[ColumnTransformer, list]:
    drop_cols = drop_cols or []
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference([TARGET_COL]).difference(drop_cols).tolist()
    categorical_cols = df.select_dtypes(include=["object","string","category"]).columns.difference([TARGET_COL]).difference(drop_cols).tolist()

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop"
    )
    return preproc, numeric_cols + categorical_cols

def smote_fit_resample(X, y):
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass
    sm = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def split_data_group_by_account(df: pd.DataFrame, val_size=0.15, test_size=0.15, random_state=RANDOM_SEED):
    """Split so that no AccountID appears in more than one split."""
    groups = df["AccountID"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx = np.arange(len(df))
    train_val_idx, test_idx = next(gss.split(idx, groups=groups))
    df_train_val = df.iloc[train_val_idx].copy()
    df_test      = df.iloc[test_idx].copy()
    groups_tv = df_train_val["AccountID"].astype(str).values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1.0-test_size), random_state=random_state)
    tr_idx, va_idx = next(gss2.split(np.arange(len(df_train_val)), groups=groups_tv))
    df_train = df_train_val.iloc[tr_idx].copy()
    df_val   = df_train_val.iloc[va_idx].copy()

    y_train = df_train.pop("is_fraud")
    y_val   = df_val.pop("is_fraud")
    y_test  = df_test.pop("is_fraud")
    return df_train, df_val, df_test, y_train, y_val, y_test

def split_data_time_based(df: pd.DataFrame, val_frac=0.15, test_frac=0.15):
    """Per-account chronological split: earliest→train, then val, then latest→test."""
    df = df.sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    parts = []
    for acc, grp in df.groupby("AccountID", sort=False):
        n = len(grp)
        n_test = max(1, int(round(n * test_frac)))
        n_val  = max(1, int(round(n * val_frac)))
        n_train = max(0, n - n_val - n_test)
        g_train = grp.iloc[:n_train]
        g_val   = grp.iloc[n_train:n_train+n_val]
        g_test  = grp.iloc[n_train+n_val:]
        parts.append((g_train, g_val, g_test))
    tr = pd.concat([p[0] for p in parts], ignore_index=True)
    va = pd.concat([p[1] for p in parts], ignore_index=True)
    te = pd.concat([p[2] for p in parts], ignore_index=True)
    y_train = tr.pop("is_fraud")
    y_val   = va.pop("is_fraud")
    y_test  = te.pop("is_fraud")
    return tr, va, te, y_train, y_val, y_test
