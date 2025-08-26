# app.py
import json, io, math
from typing import Optional, List, Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from google.cloud import storage
from google.cloud import aiplatform

from config.config import (
    PROJECT_ID, REGION, ENDPOINT_DISPLAY_NAME, ARTIFACTS_GCS_BASE
)

# Build dashboard HTML (existing)
from dashboard.build_static_dashboard import build_html

app = FastAPI(title="Fraud ML Dashboard (FastAPI)")

# ---------- Constants ----------
PRIMARY_PREFIX = f"{ARTIFACTS_GCS_BASE}/datasets/primary"
PREPROC_JOBLIB_URI   = f"{PRIMARY_PREFIX}/preproc/preprocessor.joblib"
FEATURE_NAMES_URI    = f"{PRIMARY_PREFIX}/preproc/feature_names.json"
COLUMNS_IN_URI       = f"{PRIMARY_PREFIX}/preproc/columns_in.json"
XGB_BST_URI          = f"{PRIMARY_PREFIX}/models/xgb_model.bst"
THRESHOLDS_URI       = f"{PRIMARY_PREFIX}/metrics/thresholds.json"
SCHEMA_VERSION       = "primary-v1"

# ---------- Storage helpers ----------
def _storage():
    return storage.Client(project=PROJECT_ID)

def _gcs_read_bytes(uri: str) -> Optional[bytes]:
    try:
        assert uri.startswith("gs://")
        bucket, path = uri[5:].split("/", 1)
        b = _storage().bucket(bucket).blob(path)
        if not b.exists():
            return None
        return b.download_as_bytes()
    except Exception:
        return None

def _gcs_read_text(uri: str) -> Optional[str]:
    b = _gcs_read_bytes(uri)
    return b.decode() if b else None

# ---------- Lazy artifact cache ----------
_cache: Dict[str, Any] = {}

def _ensure_artifacts_loaded():
    # Preprocessor
    if "preproc" not in _cache:
        raw = _gcs_read_bytes(PREPROC_JOBLIB_URI)
        if not raw:
            raise HTTPException(status_code=503, detail="preprocessor.joblib not found; re-run training.")
        import joblib
        _cache["preproc"] = joblib.load(io.BytesIO(raw))

    # Feature names (transformed space)
    if "feature_names" not in _cache:
        txt = _gcs_read_text(FEATURE_NAMES_URI)
        _cache["feature_names"] = json.loads(txt) if txt else []

    # Input columns that preproc expects
    if "columns_in" not in _cache:
        txt = _gcs_read_text(COLUMNS_IN_URI)
        _cache["columns_in"] = json.loads(txt) if txt else None

    # Booster for reasons
    if "booster" not in _cache:
        raw = _gcs_read_bytes(XGB_BST_URI)
        if not raw:
            raise HTTPException(status_code=503, detail="xgb_model.bst not found; re-run training.")
        bst = xgb.Booster()
        bst.load_model(io.BytesIO(raw))
        _cache["booster"] = bst

    # Thresholds
    if "threshold" not in _cache:
        txt = _gcs_read_text(THRESHOLDS_URI)
        thresh = 0.5
        if txt:
            try:
                js = json.loads(txt)
                # expect something like {"xgb": {"best_threshold": 0.61, ...}, "nlp": {...}}
                if "xgb" in js and "best_threshold" in js["xgb"]:
                    thresh = float(js["xgb"]["best_threshold"])
            except Exception:
                pass
        _cache["threshold"] = thresh

    # Endpoint
    if "endpoint" not in _cache:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        eps = list(aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'))
        if not eps:
            raise HTTPException(status_code=503, detail=f"Endpoint {ENDPOINT_DISPLAY_NAME} not found.")
        _cache["endpoint"] = eps[0]

# ---------- Health & home ----------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    ds = request.query_params.get("ds")
    try:
        html = build_html(PROJECT_ID, ds=ds)
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        msg = f"Dashboard render failed: {e}"
        return HTMLResponse(f"<h1>Fraud ML Dashboard</h1><p>{msg}</p>", status_code=200)

# ---------- Raw schema (canonical) ----------
class RawTxn(BaseModel):
    TransactionID: Optional[str] = None
    AccountID: Optional[str] = None
    TransactionAmount: Optional[float] = 0.0
    TransactionDate: Optional[str] = None   # ISO8601 or 'YYYY-MM-DD HH:MM:SS'
    TransactionType: Optional[str] = None
    Location: Optional[str] = None
    DeviceID: Optional[str] = None
    IP_Address: Optional[str] = None
    MerchantID: Optional[str] = None
    Channel: Optional[str] = None
    CustomerAge: Optional[int] = None
    CustomerOccupation: Optional[str] = None
    TransactionDuration: Optional[int] = None
    LoginAttempts: Optional[int] = 0
    AccountBalance: Optional[float] = None
    PreviousTransactionDate: Optional[str] = None
    # Optional NLP text your pipeline synthesizes (ok if absent)
    nlp_text: Optional[str] = ""

class PredictRawResponse(BaseModel):
    score: float
    threshold_used: float
    decision: int
    reasons: List[Dict[str, Any]] = Field(default_factory=list)
    schema_version: str

# ---------- Transform single raw record into model vector ----------
def _df_from_raw(raw: RawTxn) -> pd.DataFrame:
    data = raw.dict()
    # Parse datetimes
    for col in ("TransactionDate", "PreviousTransactionDate"):
        if data.get(col):
            try:
                data[col] = pd.to_datetime(data[col])
            except Exception:
                data[col] = pd.NaT
        else:
            data[col] = pd.NaT
    # Minimal defaults
    if data.get("TransactionAmount") is None:
        data["TransactionAmount"] = 0.0
    if data.get("LoginAttempts") is None:
        data["LoginAttempts"] = 0
    return pd.DataFrame([data])

# Use your existing FE
from src.feature_engineering import add_simple_features

def _transform_to_vector(df_raw: pd.DataFrame):
    _ensure_artifacts_loaded()
    preproc = _cache["preproc"]
    columns_in = _cache["columns_in"]  # columns preproc expects

    # 1) Apply your feature engineering
    df_fe = add_simple_features(df_raw.copy())

    # 2) Ensure missing columns exist
    if columns_in is None:
        # Fallback: use columns present during fit (sklearn >=1.0)
        try:
            columns_in = preproc.feature_names_in_.tolist()
        except Exception:
            raise HTTPException(status_code=500, detail="Preprocessor missing feature_names_in_.")

    for c in columns_in:
        if c not in df_fe.columns:
            # Create missing columns as NA/None; imputers in preproc will handle
            df_fe[c] = np.nan

    # 3) Transform
    X = preproc.transform(df_fe[columns_in])
    return X

# ---------- Reasons from XGB booster ----------
def _topk_reasons(X, k=6):
    _ensure_artifacts_loaded()
    bst = _cache["booster"]
    featnames = _cache["feature_names"]

    # Convert X to DMatrix
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = np.asarray(X)
    dmx = xgb.DMatrix(X_arr, feature_names=featnames if featnames else None)

    contrib = bst.predict(dmx, pred_contribs=True)[0]  # includes bias as last term
    bias = float(contrib[-1])
    contrib = contrib[:-1]  # drop bias for ranking

    names = featnames if featnames else [f"f{i}" for i in range(len(contrib))]
    pairs = list(zip(names, contrib))

    # Sort by absolute contribution
    pairs_sorted = sorted(pairs, key=lambda t: abs(float(t[1])), reverse=True)[:k]
    reasons = []
    for name, val in pairs_sorted:
        val = float(val)
        reasons.append({
            "feature": name,
            "contribution": round(val, 6),
            "sign": "+" if val >= 0 else "-"
        })
    return reasons

# ---------- Endpoint getter ----------
def _get_endpoint():
    _ensure_artifacts_loaded()
    return _cache["endpoint"]

# ---------- POST /predict/raw ----------
@app.post("/predict/raw", response_model=PredictRawResponse)
def predict_raw(raw: RawTxn):
    _ensure_artifacts_loaded()
    # 1) build dataframe
    df = _df_from_raw(raw)
    # 2) transform to model vector
    X = _transform_to_vector(df)
    # 3) score via Vertex AI
    ep = _get_endpoint()
    # Vertex AI expects plain arrays
    if hasattr(X, "toarray"):
        vec = X.toarray()[0].tolist()
    else:
        vec = np.asarray(X)[0].tolist()
    pred = ep.predict(instances=[vec])
    if not pred or not pred.predictions:
        raise HTTPException(status_code=502, detail="No prediction returned from endpoint.")
    score = float(pred.predictions[0])

    # 4) decision using saved threshold
    threshold = float(_cache["threshold"])
    decision = int(score >= threshold)

    # 5) reasons (local booster)
    reasons = _topk_reasons(X, k=6)

    return PredictRawResponse(
        score=score,
        threshold_used=threshold,
        decision=decision,
        reasons=reasons,
        schema_version=SCHEMA_VERSION,
    )
