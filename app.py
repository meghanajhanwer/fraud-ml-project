import json, io, tempfile
from typing import Optional, List, Any, Dict, Tuple, Iterable
import numpy as np
import pandas as pd
import pandas.api.types as pdt
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google.cloud import storage
from google.cloud import aiplatform
from config.config import PROJECT_ID, REGION, ENDPOINT_DISPLAY_NAME, ARTIFACTS_GCS_BASE
from dashboard.build_static_dashboard import build_html
from src.feature_engineering import add_simple_features
try:
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
except Exception:
    Pipeline = SimpleImputer = OneHotEncoder = ColumnTransformer = object

app = FastAPI(title="Fraud Detection Dashboard")
app.mount("/static", StaticFiles(directory="static"), name="static")

PRIMARY_PREFIX     = f"{ARTIFACTS_GCS_BASE}/datasets/primary"
PREPROC_JOBLIB_URI = f"{PRIMARY_PREFIX}/preproc/preprocessor.joblib"
FEATURE_NAMES_URI  = f"{PRIMARY_PREFIX}/preproc/feature_names.json" 
COLUMNS_IN_URI     = f"{PRIMARY_PREFIX}/preproc/columns_in.json"
THRESHOLDS_URI     = f"{PRIMARY_PREFIX}/metrics/thresholds.json"
SCHEMA_VERSION     = "primary-v1"
NUMERIC_COLS_HINT  = ["TransactionAmount","CustomerAge","TransactionDuration","LoginAttempts","AccountBalance"]
DATETIME_COLS_HINT = ["TransactionDate","PreviousTransactionDate"]
TEXT_COLS_HINT     = ["TransactionType","Location","DeviceID","IP_Address","MerchantID","Channel",
                      "CustomerOccupation","nlp_text","TransactionID","AccountID"]

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

_cache: Dict[str, Any] = {}

def _ensure_artifacts_loaded():
    if "preproc" not in _cache:
        _cache["preproc"] = _load_joblib_from_gcs(PREPROC_JOBLIB_URI)
    if "columns_in" not in _cache:
        txt = _gcs_read_text(COLUMNS_IN_URI); _cache["columns_in"] = json.loads(txt) if txt else None
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
def _safe_list(cols: Any) -> List[str]:
    if cols is None:
        return []
    if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
        return [str(c) for c in list(cols)]
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []

def _expected_schema_from_preproc(preproc) -> Tuple[List[str], List[str], List[str]]:
    num_cols: List[str] = []
    cat_cols: List[str] = []
    other_cols: List[str] = []
    try:
        if hasattr(preproc, "transformers_"):
            for name, trans, cols in preproc.transformers_:
                cols_list = _safe_list(cols)
                if trans in (None, "drop"):
                    continue
                steps = getattr(trans, "steps", None)
                def has_instance(t, cls):
                    if isinstance(t, cls): return True
                    if steps: return any(isinstance(est, cls) for _, est in steps)
                    return False
                if has_instance(trans, OneHotEncoder):
                    cat_cols += cols_list; continue
                simp = None
                if isinstance(trans, SimpleImputer): simp = trans
                elif steps:
                    for _, est in steps:
                        if isinstance(est, SimpleImputer): simp = est; break
                if simp is not None:
                    strat = getattr(simp, "strategy", None)
                    if strat in ("mean", "median"): num_cols += cols_list; continue
                    elif strat in ("most_frequent", "constant"): cat_cols += cols_list; continue
                lname = (name or "").lower()
                if   lname.startswith("num"): num_cols += cols_list
                elif lname.startswith("cat"): cat_cols += cols_list
                else:                         other_cols += cols_list
        else:
            num_cols, cat_cols = NUMERIC_COLS_HINT[:], TEXT_COLS_HINT[:]
    except Exception:
        num_cols, cat_cols = NUMERIC_COLS_HINT[:], TEXT_COLS_HINT[:]
    seen=set(); num_cols=[c for c in num_cols if not (c in seen or seen.add(c))]
    seen=set(); cat_cols=[c for c in cat_cols if not (c in seen or seen.add(c))]
    seen=set(); other_cols=[c for c in other_cols if not (c in seen or seen.add(c))]
    return num_cols, cat_cols, other_cols

def _df_from_raw(raw: RawTxn) -> pd.DataFrame:
    df = pd.DataFrame([raw.dict()])
    for c in DATETIME_COLS_HINT:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    for c in NUMERIC_COLS_HINT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TEXT_COLS_HINT:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def _coerce_input_frame(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    for c in df.columns:
        if c in num_cols and (pdt.is_datetime64_any_dtype(df[c]) or "date" in c.lower() or "time" in c.lower()):
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            s_ns = s.view("int64").astype("float64")
            s_ns = s_ns.where(s.notna(), np.nan)
            df[c] = s_ns / 1e9
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").astype(object)
            df[c] = df[c].astype(str).str.strip()
    for c in df.columns:
        if c not in num_cols and c not in cat_cols and df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any(): df[c] = coerced
            else: df[c] = df[c].astype(str).str.strip()
    return df
def _transform_to_vector(df_raw: pd.DataFrame):
    _ensure_artifacts_loaded()
    preproc = _cache["preproc"]
    columns_in = _cache["columns_in"]

    df_fe = add_simple_features(df_raw.copy())

    num_expected, cat_expected, other_expected = _expected_schema_from_preproc(preproc)
    expected_all = (columns_in or [])
    if not expected_all:
        expected_all = list({*num_expected, *cat_expected, *other_expected})
    for c in expected_all:
        if c not in df_fe.columns:
            df_fe[c] = np.nan
    df_in = _coerce_input_frame(df_fe.copy(), num_expected, cat_expected)
    if expected_all:
        df_in = df_in[expected_all]
    X = preproc.transform(df_in)
    try:
        from scipy import sparse as sp
        if sp.issparse(X):
            return X.astype(np.float32, copy=False).tocsr()
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32, order="C")
def _to_vertex_instance(x_row) -> List[float]:
    arr = np.asarray(x_row, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0,
                        posinf=np.finfo(np.float64).max / 2,
                        neginf=-np.finfo(np.float64).max / 2)
    return [float(v) for v in arr.tolist()]
def _get_endpoint():
    _ensure_artifacts_loaded()
    return _cache["endpoint"]

@app.post("/predict/raw", response_model=PredictRawResponse)
def predict_raw(raw: RawTxn):
    _ensure_artifacts_loaded()
    df = _df_from_raw(raw)
    X = _transform_to_vector(df)
    vec_raw = (X.toarray()[0] if hasattr(X, "toarray") else np.asarray(X)[0])
    vec = _to_vertex_instance(vec_raw)
    ep = _get_endpoint()
    pred = ep.predict(instances=[vec])
    if not pred or not getattr(pred, "predictions", None):
        raise HTTPException(status_code=502, detail="No prediction returned from endpoint.")
    score = float(pred.predictions[0])
    threshold = float(_cache["threshold"])
    decision = int(score >= threshold)

    return PredictRawResponse(score=score, threshold_used=threshold, decision=decision,
                              reasons=[], schema_version=SCHEMA_VERSION)
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
        vec_raw = (X.toarray()[0] if hasattr(X, "toarray") else np.asarray(X)[0])
        vec = _to_vertex_instance(vec_raw)
        ep = _get_endpoint()
        pred = ep.predict(instances=[vec])
        if not pred or not getattr(pred, "predictions", None):
            raise RuntimeError("No prediction returned from endpoint.")
        score = float(pred.predictions[0])
        threshold = float(_cache["threshold"])
        decision = int(score >= threshold)

        result = {"score": score, "threshold": threshold, "decision": decision,
                  "reasons": [], "schema": SCHEMA_VERSION, "input": payload}
        return _render_form_page(result=result)
    except Exception as e:
        return _render_form_page(err=str(e))