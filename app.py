# app.py
import base64
import io
import json
from typing import Optional, List, Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from google.cloud import storage
from google.cloud import aiplatform

from config.config import (
    PROJECT_ID, REGION, ENDPOINT_DISPLAY_NAME,
    ARTIFACTS_GCS_PREFIX
)
# Reuse your dynamic HTML builder
from dashboard.build_static_dashboard import build_html  # updated in step 3

app = FastAPI(title="Fraud ML Dashboard (FastAPI)")

# ---- GCS helpers ----
def _storage():
    return storage.Client(project=PROJECT_ID)

def gcs_text(gcs_path: str) -> Optional[str]:
    assert gcs_path.startswith("gs://")
    bucket = gcs_path.split("gs://",1)[1].split("/",1)[0]
    blob   = gcs_path.split(bucket + "/", 1)[1]
    b = _storage().bucket(bucket).blob(blob)
    return b.download_as_text() if b.exists() else None

# ---- Home: dynamic dashboard ----
@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = build_html(PROJECT_ID)  # renders using GCS artifacts
    return HTMLResponse(content=html, status_code=200)

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# ---- Vertex AI predict proxy ----
# NOTE: this expects *already preprocessed* feature vectors as your XGB
# endpoint does. (Same format as your earlier smoke test.)
class PredictRequest(BaseModel):
    instances: List[Any]

class PredictResponse(BaseModel):
    predictions: Optional[List[Any]] = None
    error: Optional[str] = None
    model: Optional[str] = None
    deployed_model_id: Optional[str] = None
    model_version_id: Optional[str] = None

_endpoint_cache = None

def _get_endpoint():
    global _endpoint_cache
    if _endpoint_cache is not None:
        return _endpoint_cache
    aiplatform.init(project=PROJECT_ID, location=REGION)
    eps = list(aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'))
    if not eps:
        raise RuntimeError(f"No endpoint with display_name={ENDPOINT_DISPLAY_NAME}")
    _endpoint_cache = eps[0]
    return _endpoint_cache

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        ep = _get_endpoint()
        pred = ep.predict(instances=req.instances)
        return PredictResponse(
            predictions=pred.predictions,
            model=pred.model_resource_name,
            deployed_model_id=pred.deployed_model_id,
            model_version_id=getattr(pred, "model_version_id", None),
        )
    except Exception as e:
        return PredictResponse(error=str(e))
