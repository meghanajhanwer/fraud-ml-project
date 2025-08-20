# src/model_lstm.py
from typing import Dict, Any, Tuple
import os, tempfile, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from google.cloud import storage

from config.config import (
    RANDOM_SEED, ARTIFACTS_GCS_PREFIX, PROJECT_ID,
    TARGET_COL, SEQUENCE_LENGTH
)

import tensorflow as tf
from tensorflow import keras

rng = np.random.default_rng(RANDOM_SEED)

def _upload_dir_to_gcs(local_dir: str, gcs_uri: str):
    assert gcs_uri.startswith("gs://")
    _, path = gcs_uri.split("gs://", 1)
    bucket_name, prefix = path.split("/", 1)
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for f in files:
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, local_dir).replace("\\", "/")
            blob = bucket.blob(f"{prefix}/{rel}")
            blob.upload_from_filename(lp)

def _build_sequences(df: pd.DataFrame, feature_cols):
    # df must have AccountID, TransactionDate, TARGET_COL and feature_cols
    df = df.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
    Xs, ys = [], []
    for _, grp in df.groupby("AccountID"):
        if len(grp) < SEQUENCE_LENGTH:
            continue
        feats = grp[feature_cols].to_numpy(dtype=np.float32)
        labels = grp[TARGET_COL].to_numpy()
        for i in range(SEQUENCE_LENGTH - 1, len(grp)):
            Xs.append(feats[i-SEQUENCE_LENGTH+1:i+1, :])
            ys.append(labels[i])
    if not Xs:
        return None, None
    X = np.stack(Xs, axis=0)  # [N, L, D]
    y = np.array(ys).astype(np.int32)
    return X, y

def _build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Masking(mask_value=0.0),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["AUC", "Precision", "Recall"])
    return model

def train_eval_lstm(
    X_train_num: pd.DataFrame, y_train: pd.Series,
    X_val_num: pd.DataFrame,   y_val: pd.Series,
    acc_train: pd.Series,      t_train: pd.Series,
    acc_val: pd.Series,        t_val: pd.Series
) -> Tuple[Dict[str, Any], Any]:

    # Build DataFrames with required columns
    df_tr = X_train_num.copy()
    df_tr[TARGET_COL] = y_train.values
    df_tr["AccountID"] = acc_train.values
    df_tr["TransactionDate"] = pd.to_datetime(t_train.values)

    df_va = X_val_num.copy()
    df_va[TARGET_COL] = y_val.values
    df_va["AccountID"] = acc_val.values
    df_va["TransactionDate"] = pd.to_datetime(t_val.values)

    # Scale numeric features
    feature_cols = X_train_num.columns.tolist()
    scaler = StandardScaler()
    Xtr_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=feature_cols, index=X_train_num.index)
    Xva_scaled = pd.DataFrame(scaler.transform(X_val_num), columns=feature_cols, index=X_val_num.index)

    df_tr[feature_cols] = Xtr_scaled
    df_va[feature_cols] = Xva_scaled

    # Build sequences
    Xtr_seq, ytr_seq = _build_sequences(df_tr[["AccountID","TransactionDate",TARGET_COL]+feature_cols], feature_cols)
    Xva_seq, yva_seq = _build_sequences(df_va[["AccountID","TransactionDate",TARGET_COL]+feature_cols], feature_cols)

    if Xtr_seq is None or Xva_seq is None:
        # Not enough sequences – return zeros and don't export
        return {"model_type":"lstm","precision":0,"recall":0,"f1":0,"roc_auc":0,"pr_auc":0}, None

    input_shape = (Xtr_seq.shape[1], Xtr_seq.shape[2])
    model = _build_model(input_shape)

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq), epochs=8, batch_size=64, callbacks=[es], verbose=0)

    # Evaluate
    y_proba = model.predict(Xva_seq, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    pr, rc, f1, _ = precision_recall_fscore_support(yva_seq, y_pred, average="binary", zero_division=0)
    metrics = {
        "model_type": "lstm",
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(yva_seq, y_proba)) if len(np.unique(yva_seq)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(yva_seq, y_proba)) if len(np.unique(yva_seq)) > 1 else 0.0,
    }

    # Export SavedModel and upload to GCS
    export_uri = f"{ARTIFACTS_GCS_PREFIX}/models/lstm_savedmodel"
    with tempfile.TemporaryDirectory() as td:
        local_dir = os.path.join(td, "saved_model")
        # Keras SavedModel
        model.save(local_dir)  # creates saved_model.pb + variables/*
        # Upload directory recursively
        _upload_dir_to_gcs(local_dir, export_uri)

    return metrics, model
