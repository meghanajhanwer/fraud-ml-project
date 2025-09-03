import io, json, base64
from typing import Optional, Dict, Any, List, Tuple
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

CANDIDATE_MODELS = ["xgb", "rf", "logreg", "lgbm", "catboost", "svm", "dt", "nb"]

def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

def _read_metrics_legacy(prefix_uri: str) -> Dict[str, Any]:
    """
    Legacy format: metrics/metrics.json containing a dict like {"xgb": {...}, "rf": {...}}
    or a single {"xgb": {...}, ... "best_threshold": ...}.
    """
    j = dl_text(prefix_uri, "metrics/metrics.json")
    try:
        return json.loads(j) if j else {}
    except Exception:
        return {}

def _read_metrics_per_model(prefix_uri: str) -> Dict[str, Any]:
    """
    Newer format: one JSON per model in metrics/<model>.json
    """
    found: Dict[str, Any] = {}
    for m in CANDIDATE_MODELS:
        txt = dl_text(prefix_uri, f"metrics/{m}.json")
        if not txt:
            continue
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                found[m] = obj
        except Exception:
            pass
    return found

def _collect_models(prefix_uri: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict: model_name -> metrics dict (must include at least one of pr_auc/f1/roc_auc)
    Will merge legacy and per-model sources; per-model overrides legacy if both present.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    legacy = _read_metrics_legacy(prefix_uri)
    if isinstance(legacy, dict):
        for k, v in legacy.items():
            if isinstance(v, dict) and (("pr_auc" in v) or ("f1" in v) or ("roc_auc" in v)):
                merged[k] = v

    per_model = _read_metrics_per_model(prefix_uri)
    for k, v in per_model.items():
        merged[k] = v
    if not merged and isinstance(legacy, dict) and ("pr_auc" in legacy or "f1" in legacy or "roc_auc" in legacy):
        merged["xgb"] = legacy

    return merged

def _find_best_model(models: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """
    Best by PR-AUC, then F1, then ROC-AUC.
    Returns (best_model_name, criterion_used)
    """
    if not models:
        return None, "none"
    scored = []
    for name, m in models.items():
        pr = _safe_float(m.get("pr_auc"), -1.0)
        f1 = _safe_float(m.get("f1"), -1.0)
        roc = _safe_float(m.get("roc_auc"), -1.0)
        scored.append((name, pr, f1, roc))
    best = max(scored, key=lambda t: (t[1], t[2], t[3]))
    if all(t[1] is None or t[1] < 0 for t in scored):
        best = max(scored, key=lambda t: (t[2], t[3]))
        return best[0], "f1"
    return best[0], "pr_auc"

def _kpi(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"

def _img_or_note(b64: Optional[str], title: str) -> str:
    return f"<h3 class='h6 mb-2'>{title}</h3>" + (f"<img src='{b64}' />" if b64 else "<p class='small text-muted mb-0'>Not found.</p>")

def _try_plot(prefix_uri: str, model: str, kind: str) -> Optional[str]:
    """
    Try two naming schemes:
      plots/{kind}_{model}.png   e.g., plots/confusion_xgb.png
      plots/{model}/{kind}.png   e.g., plots/xgb/confusion.png
    """
    p1 = embed_png_b64(prefix_uri, f"plots/{kind}_{model}.png")
    if p1:
        return p1
    return embed_png_b64(prefix_uri, f"plots/{model}/{kind}.png")

def build_html(project_id: str, ds: Optional[str] = None) -> str:
    ds = "primary"
    ds_prefix = f"{ARTIFACTS_GCS_BASE}/datasets/{ds}"

    models = _collect_models(ds_prefix)
    best_model, criterion = _find_best_model(models)
    chosen_key = best_model or ("xgb" if "xgb" in models else (next(iter(models.keys())) if models else None))
    best_metrics = models.get(chosen_key, {}) if chosen_key else {}

    f1   = _kpi(best_metrics.get("f1"))
    pr   = _kpi(best_metrics.get("pr_auc"))
    roc  = _kpi(best_metrics.get("roc_auc"))
    acc  = _kpi(best_metrics.get("accuracy"))
    prec = _kpi(best_metrics.get("precision"))
    rec  = _kpi(best_metrics.get("recall"))
    thr  = _kpi(best_metrics.get("threshold"))
    cm_legacy  = embed_png_b64(ds_prefix, "plots/confusion_xgb.png")
    pr_legacy  = embed_png_b64(ds_prefix, "plots/pr_xgb.png")
    roc_legacy = embed_png_b64(ds_prefix, "plots/roc_xgb.png")
    def comparison_rows() -> str:
        if not models:
            return "<tr><td colspan='8' class='text-muted small'>No model metrics found.</td></tr>"
        items = []
        for name, m in models.items():
            items.append({
                "name": name,
                "pr_auc": _safe_float(m.get("pr_auc")),
                "roc_auc": _safe_float(m.get("roc_auc")),
                "f1": _safe_float(m.get("f1")),
                "precision": _safe_float(m.get("precision")),
                "recall": _safe_float(m.get("recall")),
                "accuracy": _safe_float(m.get("accuracy")),
                "threshold": _safe_float(m.get("threshold")),
            })
        items.sort(key=lambda x: (x["pr_auc"] if x["pr_auc"] is not None else -1,
                                  x["f1"] if x["f1"] is not None else -1,
                                  x["roc_auc"] if x["roc_auc"] is not None else -1), reverse=True)
        max_pr  = max([x["pr_auc"] for x in items if x["pr_auc"] is not None] or [1])
        max_f1  = max([x["f1"]     for x in items if x["f1"]     is not None] or [1])

        rows = []
        for x in items:
            star = "" if x["name"] == chosen_key else ""
            pr_val = _kpi(x["pr_auc"]); f1_val = _kpi(x["f1"])
            pr_w = 0 if x["pr_auc"] is None else int(100 * (x["pr_auc"] / max_pr))
            f1_w = 0 if x["f1"]     is None else int(100 * (x["f1"]     / max_f1))
            rows.append(f"""
            <tr>
              <td><code>{x['name']}</code> {star}</td>
              <td>
                <div class="bar"><div class="fill" style="width:{pr_w}%"></div></div>
                <div class="small">{pr_val}</div>
              </td>
              <td>
                <div class="bar"><div class="fill f1" style="width:{f1_w}%"></div></div>
                <div class="small">{f1_val}</div>
              </td>
              <td>{_kpi(x['roc_auc'])}</td>
              <td>{_kpi(x['precision'])}</td>
              <td>{_kpi(x['recall'])}</td>
              <td>{_kpi(x['accuracy'])}</td>
              <td>{_kpi(x['threshold'])}</td>
            </tr>
            """)
        return "\n".join(rows)

    def confusion_grid() -> str:
        if not models:
            return f"""
            <div class="col-lg-4"><div class="card"><div class="card-body">
              {_img_or_note(cm_legacy, "Confusion (XGB)")}
            </div></div></div>
            <div class="col-lg-4"><div class="card"><div class="card-body">
              {_img_or_note(pr_legacy, "Precision - Recall (XGB)")}
            </div></div></div>
            <div class="col-lg-4"><div class="card"><div class="card-body">
              {_img_or_note(roc_legacy, "ROC (XGB)")}
            </div></div></div>
            """
        blocks = []
        show_models = list(models.keys())[:3]
        for m in show_models:
            cm   = _try_plot(ds_prefix, m, "confusion")
            prp  = _try_plot(ds_prefix, m, "pr")
            rocp = _try_plot(ds_prefix, m, "roc")
            blocks.append(f"""
            <div class="col-lg-4">
              <div class="card"><div class="card-body">
                <h5 class="mb-2">{m.upper()}</h5>
                {_img_or_note(cm, "Confusion")}
                {_img_or_note(prp, "Precision–Recall")}
                {_img_or_note(rocp, "ROC")}
              </div></div>
            </div>""")
        return "\n".join(blocks)
    best_name_label = (chosen_key or "—").upper()
    crit_label = {"pr_auc": "PR-AUC", "f1": "F1"}.get(criterion, criterion)

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
  .dataset .badge{{font-size:12px}}
  .dataset a.btn{{margin-left:8px}}
  .bar{{height:8px;background:#eef2f7;border-radius:999px;overflow:hidden;margin-bottom:4px}}
  .fill{{height:8px;background:#4f46e5}} .fill.f1{{background:#0ea5e9}}
  table.table td, table.table th{{vertical-align:middle}}
</style>
</head><body>
<div class="container-fluid">
  <h1 class="mb-2">Fraud Detection Dashboard</h1>
  <p class="text-muted">Project: <code>{project_id}</code> — Artifacts: <code>{ds_prefix}</code></p>

  <div class="dataset mb-3">
    <span class="me-2">Dataset:</span>
    <span class="badge bg-secondary">primary</span>
    <a href="/predict/form" class="btn btn-sm btn-primary">Live Predict</a>
  </div>

  <div class="card"><div class="card-body">
    <h2 class="mb-1">Best Model: {best_name_label}</h2>
    <p class="text-muted mb-3 small">Selected by <strong>{crit_label or '—'}</strong></p>
    <div class="kpi">
      <div class="item"><div class="value">{f1}</div><div class="label">F1 (at threshold)</div></div>
      <div class="item"><div class="value">{pr}</div><div class="label">PR AUC</div></div>
      <div class="item"><div class="value">{roc}</div><div class="label">ROC AUC</div></div>
      <div class="item"><div class="value">{acc}</div><div class="label">Accuracy</div></div>
      <div class="item"><div class="value">{prec}</div><div class="label">Precision</div></div>
      <div class="item"><div class="value">{rec}</div><div class="label">Recall</div></div>
      <div class="item"><div class="value">{thr}</div><div class="label">Threshold</div></div>
    </div>
  </div></div>

  <div class="card"><div class="card-body">
    <h2 class="mb-3">Model Comparison</h2>
    <div class="table-responsive">
      <table class="table table-sm align-middle">
        <thead>
          <tr>
            <th>Model</th>
            <th>PR AUC</th>
            <th>F1</th>
            <th>ROC AUC</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Accuracy</th>
            <th>Threshold</th>
          </tr>
        </thead>
        <tbody>
          {comparison_rows()}
        </tbody>
      </table>
    </div>
  </div></div>

  <div class="row g-3">
    {confusion_grid()}
  </div>

  <footer class="text-muted mt-3"><small>Rendered from latest artifacts in GCS. Live Predict uses your deployed Vertex AI endpoint.</small></footer>
</div>
</body></html>"""
    return html
