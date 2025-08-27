import io, json, base64, csv
from typing import Optional
from google.cloud import storage
from config.config import ARTIFACTS_GCS_BASE, ARTIFACTS_GCS_PREFIX, PROJECT_ID

def _storage(): 
  return storage.Client(project=PROJECT_ID)

def _bucket_blob_from_gcs_uri(gcs_uri: str, rel_path: str):
    assert gcs_uri.startswith("gs://")
    _, rest = gcs_uri.split("gs://", 1)
    bucket, prefix = rest.split("/", 1)
    prefix = prefix.rstrip("/")
    blob_path = f"{prefix}/{rel_path}".lstrip("/")
    return bucket, blob_path

def dl_bytes(prefix_uri: str, rel_path: str) -> Optional[bytes]:
    bucket, blob_path = _bucket_blob_from_gcs_uri(prefix_uri, rel_path)
    blob = _storage().bucket(bucket).blob(blob_path)
    return blob.download_as_bytes() if blob.exists() else None

def dl_text(prefix_uri: str, rel_path: str) -> Optional[str]:
    b = dl_bytes(prefix_uri, rel_path)
    return b.decode("utf-8") if b else None

def embed_png_b64(prefix_uri: str, rel_path: str) -> Optional[str]:
    b = dl_bytes(prefix_uri, rel_path)
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii") if b else None

def _read_metrics(prefix_uri: str) -> dict:
    j = dl_text(prefix_uri, "metrics/metrics.json")
    try: return json.loads(j) if j else {}
    except Exception: return {}

def _kpi(value) -> str:
    try: return f"{float(value):.3f}"
    except Exception: return "-"

def _img_or_note(b64: Optional[str], title: str) -> str:
    return f"<h3>{title}</h3><img src='{b64}' />" if b64 else f"<h3>{title}</h3><p class='small'>Not found.</p>"

def build_html(project_id: str, ds: Optional[str] = None) -> str:
    ds = (ds or ARTIFACTS_GCS_PREFIX.split('/datasets/')[-1]).lower()
    ds_prefix = f"{ARTIFACTS_GCS_BASE}/datasets/{ds}"

    metrics = _read_metrics(ds_prefix)
    xgb = metrics.get("xgb", {}) if isinstance(metrics, dict) else {}
    acc = _kpi(xgb.get("accuracy")); prec = _kpi(xgb.get("precision"))
    rec = _kpi(xgb.get("recall"));   f1   = _kpi(xgb.get("f1"))
    roc = _kpi(xgb.get("roc_auc"));  pr   = _kpi(xgb.get("pr_auc"))

    cm_b64  = embed_png_b64(ds_prefix, "plots/confusion_xgb.png")
    pr_b64  = embed_png_b64(ds_prefix, "plots/pr_xgb.png")
    roc_b64 = embed_png_b64(ds_prefix, "plots/roc_xgb.png")

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud ML Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body{{padding:20px;background:#f7f9fc}}
  img{{max-width:100%;height:auto}}
  .card{{margin-bottom:18px;border:1px solid #e6eef5;border-radius:14px;box-shadow:0 1px 2px rgba(0,0,0,.03)}}
  .kpi{{display:flex;gap:16px;flex-wrap:wrap}}
  .kpi .item{{background:#fff;border:1px solid #e6eef5;border-radius:12px;padding:12px 16px;min-width:120px}}
  .kpi .value{{font-size:22px;font-weight:700}}
  .kpi .label{{color:#6b7280;font-size:12px}}
  .dataset a.btn{{margin-right:8px}}
</style>
</head><body>
<div class="container-fluid">
  <h1 class="mb-2">Fraud Detection Dashboard</h1>
  <p class="text-muted">Project: <code>{project_id}</code> — Artifacts: <code>{ds_prefix}</code></p>

  <div class="dataset mb-3">
    <span class="me-2">Dataset:</span>
    <a href="?ds=primary" class="btn btn-sm btn-outline-primary">primary</a>
    <a href="?ds=ulb" class="btn btn-sm btn-outline-primary">ulb</a>
    <a href="/predict/form" class="btn btn-sm btn-primary">Live Predict</a>
  </div>

  <div class="card"><div class="card-body">
    <h2 class="mb-3">Best Model: XGBoost</h2>
    <div class="kpi">
      <div class="item"><div class="value">{f1}</div><div class="label">F1</div></div>
      <div class="item"><div class="value">{pr}</div><div class="label">PR AUC</div></div>
      <div class="item"><div class="value">{roc}</div><div class="label">ROC AUC</div></div>
      <div class="item"><div class="value">{acc}</div><div class="label">Accuracy</div></div>
      <div class="item"><div class="value">{prec}</div><div class="label">Precision</div></div>
      <div class="item"><div class="value">{rec}</div><div class="label">Recall</div></div>
    </div>
  </div></div>

  <div class="row g-3">
    <div class="col-lg-6">
      <div class="card"><div class="card-body">
        {_img_or_note(cm_b64, "Confusion Matrix (XGB)")}
      </div></div>
    </div>
    <div class="col-lg-6">
      <div class="card"><div class="card-body">
        {_img_or_note(pr_b64, "Precision–Recall (XGB)")}
        {_img_or_note(roc_b64, "ROC (XGB)")}
      </div></div>
    </div>
  </div>

  <footer class="text-muted mt-3"><small>Rendered live from GCS artifacts. Live Predict uses your deployed Vertex AI endpoint.</small></footer>
</div>
</body></html>"""
    return html
