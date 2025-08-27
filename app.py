import json, io, tempfile
from typing import Optional, List, Any, Dict
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google.cloud import storage
from google.cloud import aiplatform
from config.config import PROJECT_ID, REGION, ENDPOINT_DISPLAY_NAME, ARTIFACTS_GCS_BASE
from dashboard.build_static_dashboard import build_html
from src.feature_engineering import add_simple_features

app = FastAPI(title="Fraud Detection Dashboard")
app.mount("/static", StaticFiles(directory="static"), name="static")

PRIMARY_PREFIX     = f"{ARTIFACTS_GCS_BASE}/datasets/primary"
PREPROC_JOBLIB_URI = f"{PRIMARY_PREFIX}/preproc/preprocessor.joblib"
FEATURE_NAMES_URI  = f"{PRIMARY_PREFIX}/preproc/feature_names.json"
COLUMNS_IN_URI     = f"{PRIMARY_PREFIX}/preproc/columns_in.json"
XGB_BST_URI        = f"{PRIMARY_PREFIX}/models/xgb_model.bst"
THRESHOLDS_URI     = f"{PRIMARY_PREFIX}/metrics/thresholds.json"
SCHEMA_VERSION     = "primary-v1"

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
    data = _gcs_read_bytes(uri)
    return data.decode() if data else None

def _load_joblib_from_gcs(uri: str):
    raw = _gcs_read_bytes(uri)
    if not raw:
        raise HTTPException(status_code=503, detail=f"Artifact not found at {uri}. Re-run training.")
    import joblib
    try:
        return joblib.load(io.BytesIO(raw))
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tf:
            tf.write(raw); tf.flush()
            return joblib.load(tf.name)

def _load_booster_from_gcs(uri: str) -> xgb.Booster:
    data = _gcs_read_bytes(uri)
    if not data:
        raise HTTPException(status_code=503, detail=f"Model file not found at {uri}. Re-run training.")
    bst = xgb.Booster()
    try:
        bst.load_model(bytearray(data))
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".bst") as tf:
            tf.write(data); tf.flush()
            bst.load_model(tf.name)
    return bst

_cache: Dict[str, Any] = {}

def _ensure_artifacts_loaded():
    if "preproc" not in _cache:
        _cache["preproc"] = _load_joblib_from_gcs(PREPROC_JOBLIB_URI)
    if "feature_names" not in _cache:
        txt = _gcs_read_text(FEATURE_NAMES_URI); _cache["feature_names"] = json.loads(txt) if txt else []
    if "columns_in" not in _cache:
        txt = _gcs_read_text(COLUMNS_IN_URI); _cache["columns_in"] = json.loads(txt) if txt else None
    if "booster" not in _cache:
        _cache["booster"] = _load_booster_from_gcs(XGB_BST_URI)
    if "threshold" not in _cache:
        txt = _gcs_read_text(THRESHOLDS_URI); thr = 0.5
        if txt:
            try: thr = float(json.loads(txt).get("xgb", {}).get("best_threshold", 0.5))
            except Exception: pass
        _cache["threshold"] = thr
    if "endpoint" not in _cache:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        eps = list(aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'))
        if not eps:
            raise HTTPException(status_code=503, detail=f"Endpoint {ENDPOINT_DISPLAY_NAME} not found.")
        _cache["endpoint"] = eps[0]

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    ds = request.query_params.get("ds")
    html = build_html(PROJECT_ID, ds=ds)
    return HTMLResponse(content=html, status_code=200)

class RawTxn(BaseModel):
    TransactionID: Optional[str] = None
    AccountID: Optional[str] = None
    TransactionAmount: Optional[float] = 0.0
    TransactionDate: Optional[str] = None
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
    nlp_text: Optional[str] = ""

class PredictRawResponse(BaseModel):
    score: float
    threshold_used: float
    decision: int
    reasons: List[Dict[str, Any]] = Field(default_factory=list)
    schema_version: str

def _df_from_raw(raw: RawTxn) -> pd.DataFrame:
    data = raw.dict()
    for col in ("TransactionDate", "PreviousTransactionDate"):
        if data.get(col):
            try: data[col] = pd.to_datetime(data[col])
            except Exception: data[col] = pd.NaT
        else:
            data[col] = pd.NaT
    data["TransactionAmount"] = float(data.get("TransactionAmount") or 0.0)
    data["LoginAttempts"] = int(data.get("LoginAttempts") or 0)
    return pd.DataFrame([data])

def _transform_to_vector(df_raw: pd.DataFrame):
    _ensure_artifacts_loaded()
    preproc = _cache["preproc"]
    columns_in = _cache["columns_in"]
    df_fe = add_simple_features(df_raw.copy())
    if columns_in is None:
        try:
            columns_in = preproc.feature_names_in_.tolist()
        except Exception:
            raise HTTPException(status_code=500, detail="Preprocessor missing feature_names_in_.")
    for c in columns_in:
        if c not in df_fe.columns:
            df_fe[c] = np.nan
    X = preproc.transform(df_fe[columns_in])
    try:
        from scipy import sparse as sp
        if sp.issparse(X):
            return X.astype(np.float32, copy=False).tocsr()
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32, order="C")

def _topk_reasons(X, k=6):
    """Compute per-feature contributions using the local XGBoost booster.
       Always cast to numeric float32 to avoid unicode/object issues."""
    _ensure_artifacts_loaded()
    bst = _cache["booster"]
    featnames = _cache.get("feature_names") or []
    try:
        from scipy import sparse as sp
    except Exception:
        sp = None
    if sp and sp.issparse(X):
        Xn = X.astype(np.float32, copy=False).tocsr()
        dmx = xgb.DMatrix(Xn)
        nfeat = Xn.shape[1]
    else:
        arr = np.array(X, dtype=np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        dmx = xgb.DMatrix(arr)
        nfeat = arr.shape[1]

    contrib = bst.predict(dmx, pred_contribs=True, validate_features=False)[0]
    if len(contrib) == nfeat + 1:
        contrib = contrib[:-1]
    contrib = np.asarray(contrib, dtype=float)
    names = featnames if len(featnames) == nfeat else [f"f{i}" for i in range(nfeat)]
    order = np.argsort(-np.abs(contrib))[:k]
    return [
        {"feature": names[i], "contribution": float(contrib[i]), "sign": ("+" if contrib[i] >= 0 else "-")}
        for i in order
    ]


def _get_endpoint():
    _ensure_artifacts_loaded()
    return _cache["endpoint"]

@app.post("/predict/raw", response_model=PredictRawResponse)
def predict_raw(raw: RawTxn):
    _ensure_artifacts_loaded()
    df = _df_from_raw(raw)
    X = _transform_to_vector(df)
    ep = _get_endpoint()
    vec = (X.toarray()[0] if hasattr(X, "toarray") else np.asarray(X)[0]).tolist()
    vec = [float(v) for v in vec]
    pred = ep.predict(instances=[vec])
    if not pred or not pred.predictions:
        raise HTTPException(status_code=502, detail="No prediction returned from endpoint.")
    score = float(pred.predictions[0])
    threshold = float(_cache["threshold"])
    decision = int(score >= threshold)
    reasons = _topk_reasons(X, k=6)
    return PredictRawResponse(score=score, threshold_used=threshold, decision=decision,
                              reasons=reasons, schema_version=SCHEMA_VERSION)

def _render_form_page(result: Dict[str, Any] | None = None, err: str | None = None) -> HTMLResponse:
    preset = {
        "TransactionAmount": "249.99",
        "TransactionDate": "2024-05-03T14:12:00Z",
        "PreviousTransactionDate": "2024-05-02T10:00:00Z",
        "TransactionType": "purchase",
        "Location": "NY", "DeviceID": "mobile", "IP_Address": "203.0.113.15",
        "MerchantID": "M-778", "Channel": "mobile_app", "CustomerAge": "34",
        "CustomerOccupation": "Engineer", "TransactionDuration": "45",
        "LoginAttempts": "1", "AccountBalance": "1200.50", "nlp_text": ""
    }
    def val(k): return (result.get("input", {}).get(k) if result else None) or preset.get(k, "")
    html = f"""<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="/static/style.css"><title>Live Prediction • Fraud</title>
</head><body><div class="container">
  <h1>Live Prediction</h1>
  <div class="card"><h2>Input</h2><div class="body">
    <form method="post" action="/predict/form">
      <div class="grid">
        <div><label>TransactionAmount</label><input name="TransactionAmount" type="number" step="0.01" value="{val('TransactionAmount')}" required></div>
        <div><label>TransactionDate</label><input name="TransactionDate" type="text" placeholder="YYYY-MM-DDTHH:MM:SSZ" value="{val('TransactionDate')}"></div>
        <div><label>PreviousTransactionDate</label><input name="PreviousTransactionDate" type="text" placeholder="YYYY-MM-DDTHH:MM:SSZ" value="{val('PreviousTransactionDate')}"></div>
        <div><label>TransactionType</label><input name="TransactionType" type="text" value="{val('TransactionType')}"></div>
        <div><label>Location</label><input name="Location" type="text" value="{val('Location')}"></div>
        <div><label>DeviceID</label><input name="DeviceID" type="text" value="{val('DeviceID')}"></div>
        <div><label>IP_Address</label><input name="IP_Address" type="text" value="{val('IP_Address')}"></div>
        <div><label>MerchantID</label><input name="MerchantID" type="text" value="{val('MerchantID')}"></div>
        <div><label>Channel</label><input name="Channel" type="text" value="{val('Channel')}"></div>
        <div><label>CustomerAge</label><input name="CustomerAge" type="number" value="{val('CustomerAge')}"></div>
        <div><label>CustomerOccupation</label><input name="CustomerOccupation" type="text" value="{val('CustomerOccupation')}"></div>
        <div><label>TransactionDuration</label><input name="TransactionDuration" type="number" value="{val('TransactionDuration')}"></div>
        <div><label>LoginAttempts</label><input name="LoginAttempts" type="number" value="{val('LoginAttempts')}"></div>
        <div><label>AccountBalance</label><input name="AccountBalance" type="number" step="0.01" value="{val('AccountBalance')}"></div>
        <div><label>nlp_text</label><input name="nlp_text" type="text" value="{val('nlp_text')}"></div>
      </div>
      <div style="margin-top:12px" class="row">
        <button class="btn" type="submit">Predict</button>
        <a class="btn" href="/">Back to Dashboard</a>
      </div>
    </form>
  </div></div>
  {"<div class='card'><h2>Result</h2><div class='body'>" + (
    f"<p>Score: <span class='badge {'ok' if (result or {}).get('decision')==0 else 'bad'}'>{(result or {}).get('score'):.4f}</span> — Threshold: {(result or {}).get('threshold'):.4f} — Decision: <strong>{'Fraud' if (result or {}).get('decision') else 'Legit'}</strong></p>"
    f"<p class='small'>Schema: {(result or {}).get('schema')}</p>"
    "<h3>Top reasons</h3><ul>" +
    "".join([f"<li><code>{r['feature']}</code> — {r['contribution']:+.4f}</li>" for r in (result or {}).get('reasons', [])]) +
    "</ul><h3>Raw JSON</h3><pre>" + json.dumps(result, indent=2) + "</pre>"
  ) + "</div></div>" if result else ""}
  {f"<div class='card'><h2>Error</h2><div class='body'><pre>{err}</pre></div></div>" if err else ""}
</div></body></html>"""
    return HTMLResponse(content=html, status_code=200)

@app.get("/predict/form", response_class=HTMLResponse)
def form_page():
    return _render_form_page()

@app.post("/predict/form", response_class=HTMLResponse)
async def form_submit(request: Request):
    try:
        form = await request.form()
        payload = {k: (v if str(v).strip() != "" else None) for k, v in form.items()}
        for k in ("TransactionAmount","CustomerAge","TransactionDuration","LoginAttempts","AccountBalance"):
            if payload.get(k) is not None:
                try:
                    payload[k] = float(payload[k]) if k in ("TransactionAmount","AccountBalance") else int(payload[k])
                except Exception:
                    pass
        raw = RawTxn(**payload)
        df = _df_from_raw(raw)
        X = _transform_to_vector(df)
        ep = _get_endpoint()
        vec = (X.toarray()[0] if hasattr(X, "toarray") else np.asarray(X)[0]).tolist()
        vec = [float(v) for v in vec]
        pred = ep.predict(instances=[vec])
        if not pred or not pred.predictions:
            raise RuntimeError("No prediction returned from endpoint.")
        score = float(pred.predictions[0])
        threshold = float(_cache["threshold"])
        decision = int(score >= threshold)
        reasons = _topk_reasons(X, k=6)
        result = {"score": score, "threshold": threshold, "decision": decision,
                  "reasons": reasons, "schema": SCHEMA_VERSION, "input": payload}
        return _render_form_page(result=result)
    except Exception as e:
        return _render_form_page(err=str(e))
