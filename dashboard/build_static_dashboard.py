# dashboard/build_static_dashboard.py
import io, json, csv, base64, os
from typing import Optional
from google.cloud import storage

from config.config import ARTIFACTS_GCS_BASE, ARTIFACTS_GCS_PREFIX, PROJECT_ID

def _storage():
    from config.config import PROJECT_ID
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
    b = dl_bytes(prefix_uri, rel_path);  return b.decode("utf-8") if b else None

def embed_png_b64(prefix_uri: str, rel_path: str) -> Optional[str]:
    b = dl_bytes(prefix_uri, rel_path)
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii") if b else None

def json_table(jtxt: Optional[str], title: str, cols=None):
    if not jtxt: return f"<h3>{title}</h3><p><em>Not found.</em></p>"
    obj = json.loads(jtxt)
    if isinstance(obj, dict) and cols:
        models = sorted(obj.keys())
        head = "<tr><th>model_type</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        body = ""
        for m in models:
            row = obj[m]
            body += "<tr><td>{}</td>{}</tr>".format(m, "".join(f"<td>{row.get(c,'')}</td>" for c in cols))
        table = f"<table class='table table-sm table-bordered mb-2'><thead>{head}</thead><tbody>{body}</tbody></table>"
        return f"<h3>{title}</h3><div class='table-responsive'>{table}</div>"
    return f"<h3>{title}</h3><pre>{json.dumps(obj, indent=2)}</pre>"

def csv_tbl(txt, title):
    if not txt: return f"<h3>{title}</h3><p><em>Not found.</em></p>"
    rows = list(csv.reader(io.StringIO(txt)))
    if not rows: return f"<h3>{title}</h3><p><em>Empty.</em></p>"
    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in rows[0]) + "</tr>"
    body  = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
    table = f"<table class='table table-sm table-striped table-bordered mb-2'><thead>{thead}</thead><tbody>{body}</tbody></table>"
    return f"<h3>{title}</h3><div class='table-responsive'>{table}</div>"

def build_html(project_id: str, ds: Optional[str] = None) -> str:
    # Which dataset path to display
    ds = (ds or ARTIFACTS_GCS_PREFIX.split("/datasets/")[-1]).lower()
    ds_prefix = f"{ARTIFACTS_GCS_BASE}/datasets/{ds}"

    # Also a global cross-domain area (not dataset-scoped)
    cross_prefix = ARTIFACTS_GCS_BASE

    # Dataset-specific
    m_val = dl_text(ds_prefix, "metrics/metrics.json")
    m_tst = dl_text(ds_prefix, "metrics/metrics_test.json")
    cmp_b64 = embed_png_b64(ds_prefix, "metrics/metrics_compare.png")

    thr_json = dl_text(ds_prefix, "metrics/thresholds.json")
    conf_xgb_tuned = embed_png_b64(ds_prefix, "plots/confusion_xgb_tuned.png")
    conf_nlp_tuned = embed_png_b64(ds_prefix, "plots/confusion_nlp_tuned.png")
    cal_xgb = embed_png_b64(ds_prefix, "plots/calibration_xgb.png")
    cal_nlp = embed_png_b64(ds_prefix, "plots/calibration_nlp.png")

    abl = dl_text(ds_prefix, "ablation/ablation_results.json")
    abl_fig = embed_png_b64(ds_prefix, "ablation/ablation_compare_f1.png")

    xgb_imp_fig = embed_png_b64(ds_prefix, "interpretability/xgb_feature_importance_top20.png")
    nlp_pos = embed_png_b64(ds_prefix, "interpretability/nlp_top_terms_positive.png")
    nlp_neg = embed_png_b64(ds_prefix, "interpretability/nlp_top_terms_negative.png")

    head_csv     = dl_text(ds_prefix, "eda/head.csv")
    desc_num_csv = dl_text(ds_prefix, "eda/describe_numeric.csv")
    desc_cat_csv = dl_text(ds_prefix, "eda/describe_categorical.csv")

    # Cross-domain summary
    cd_summary = dl_text(cross_prefix, "cross_domain/summary.json")

    def img_if(b64, title): return (f"<h4>{title}</h4><img src='{b64}'/>") if b64 else f"<p><em>{title} not found.</em></p>"

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud ML Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>body{{padding:20px}} img{{max-width:100%;height:auto}} .card{{margin-bottom:20px}} .table{{font-size:12px}} .card-body{{overflow:auto}}</style>
</head><body><div class="container-fluid">
  <h1 class="mb-3">📊 Fraud Detection Dashboard</h1>
  <p class="text-muted">Project: <code>{project_id}</code> — Artifacts: <code>{ds_prefix}</code></p>
  <p>Dataset selector:
    <a href="?ds=primary" class="btn btn-sm btn-outline-primary">primary</a>
    <a href="?ds=ulb" class="btn btn-sm btn-outline-primary">ulb</a>
  </p>

  <div class="card"><div class="card-body">
    <h2>Model Metrics ({ds})</h2>
    {json_table(m_val, "Validation metrics", cols=["accuracy","precision","recall","f1","roc_auc","pr_auc"])}
    <h3>Comparison (Accuracy / Precision / Recall / F1 / ROC AUC)</h3>
    {("<img src='"+cmp_b64+"'/>" if cmp_b64 else "<p><em>Comparison chart not found.</em></p>")}
    {json_table(m_tst, "Held-out TEST metrics", cols=["accuracy","precision","recall","f1","roc_auc","pr_auc"])}
  </div></div>

  <div class="card"><div class="card-body">
    <h2>Threshold Tuning & Calibration</h2>
    {json_table(thr_json, "F1-optimal thresholds")}
    <div class='row'>
      <div class='col-md-6'>{img_if(conf_xgb_tuned, "Tuned Confusion (XGB)")}</div>
      <div class='col-md-6'>{img_if(conf_nlp_tuned, "Tuned Confusion (NLP)")}</div>
    </div>
    <div class='row'>
      <div class='col-md-6'>{img_if(cal_xgb, "Calibration (XGB)")}</div>
      <div class='col-md-6'>{img_if(cal_nlp, "Calibration (NLP)")}</div>
    </div>
  </div></div>

  <div class="card"><div class="card-body">
    <h2>Ablation (drop ID-like features)</h2>
    {json_table(abl, "Ablation results (val/test)")}
    {img_if(abl_fig, "F1 — Baseline vs Ablated")}
  </div></div>

  <div class="card"><div class="card-body">
    <h2>Interpretability</h2>
    {img_if(xgb_imp_fig, "XGB Feature Importance (top 20)")}
    <div class='row'>
      <div class='col-md-6'>{img_if(nlp_pos, "NLP top positive terms")}</div>
      <div class='col-md-6'>{img_if(nlp_neg, "NLP top negative terms")}</div>
    </div>
  </div></div>

  <div class="row g-3">
    <div class="col-lg-6"><div class="card"><div class="card-body">
      {csv_tbl(head_csv, "Head (first rows)")}
    </div></div></div>
    <div class="col-lg-6"><div class="card"><div class="card-body">
      {csv_tbl(desc_num_csv, "Describe (numeric)")}
      {csv_tbl(desc_cat_csv, "Describe (categorical)")}
    </div></div></div>
  </div>

  <div class="card"><div class="card-body">
    <h2>Cross-Domain Results (global)</h2>
    {json_table(cd_summary, "Train→Test matrix (F1/PR-AUC/ROC-AUC)", cols=None)}
  </div></div>

  <footer class="text-muted mt-4"><small>Rendered live from GCS artifacts.</small></footer>
</div></body></html>"""
    return html
