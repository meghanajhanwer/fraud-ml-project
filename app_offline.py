from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import os

from dashboard.build_html_offline import build_html_local

ART_DIR = os.path.join(os.path.dirname(__file__), "offline_artifacts")
MODEL_BST = os.path.join(ART_DIR, "models", "xgb", "model.bst")

app = FastAPI(title="Fraud ML Dashboard (Offline)")

# Load XGB booster once at startup
if not os.path.exists(MODEL_BST):
    raise RuntimeError(f"Model file not found: {MODEL_BST}. Copy artifacts first.")
booster = xgb.Booster()
booster.load_model(MODEL_BST)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(build_html_local("OFFLINE"))

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

class PredictRequest(BaseModel):
    instances: list

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = np.array(req.instances, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dm = xgb.DMatrix(X)
        preds = booster.predict(dm).tolist()
        return JSONResponse({"predictions": preds, "model": "xgb-offline-bst"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
