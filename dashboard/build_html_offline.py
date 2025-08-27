import os, io, csv, json, base64
from typing import Optional

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "offline_artifacts")

def _read_bytes(rel_path: str) -> Optional[bytes]:
    p = os.path.join(BASE_DIR, rel_path)
    return open(p, "rb").read() if os.path.exists(p) else None

def _read_text(rel_path: str) -> Optional[str]:
    b = _read_bytes(rel_path);  return b.decode("utf-8") if b else None

def _img_b64(rel_path: str) -> Optional[str]:
    b = _read_bytes(rel_path)
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii") if b else None

def _json_table(jtxt: Optional[str], title: str, cols=None):
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

def _csv_tbl(txt, title):
    if not txt: return f"<h3>{title}</h3><p><em>Not found.</em></p>"
    rows = list(csv.reader(io.StringIO(txt)))
    if not rows: return f"<h3>{title}</h3><p><em>Empty.</em></p>"
    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in rows[0]) + "</tr>"
    body  = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
    table = f"<table class='table table-sm table-striped table-bordered mb-2'><thead>{thead}</thead><tbody>{body}</tbody></table>"
    return f"<h3>{title}</h3><div class='table-responsive'>{table}</div>"

def build_html_local(project_label="OFFLINE"):
    m_val = _read_text("metrics/metrics.json")
    m_tst = _read_text("metrics/metrics_test.json")
    cmp_b64 = _img_b64("metrics/metrics_compare.png")

    thr_json = _read_text("metrics/thresholds.json")
    conf_xgb_tuned = _img_b64("plots/confusion_xgb_tuned.png")
    conf_nlp_tuned = _img_b64("plots/confusion_nlp_tuned.png")
    cal_xgb = _img_b64("plots/calibration_xgb.png")
    cal_nlp = _img_b64("plots/calibration_nlp.png")

    abl = _read_text("ablation/ablation_results.json")
    abl_fig = _img_b64("ablation/ablation_compare_f1.png")

    xgb_imp = _img_b64("interpretability/xgb_feature_importance_top20.png")
    nlp_pos = _img_b64("interpretability/nlp_top_terms_positive.png")
    nlp_neg = _img_b64("interpretability/nlp_top_terms_negative.png")

    head_csv     = _read_text("eda/head.csv")
    desc_num_csv = _read_text("eda/describe_numeric.csv")
    desc_cat_csv = _read_text("eda/describe_categorical.csv")

    def img_if(b64, title): return (f"<h4>{title}</h4><img src='{b64}'/>") if b64 else f"<p><em>{title} not found.</em></p>"

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud ML Dashboard (Offline)</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>body{{padding:20px}} img{{max-width:100%;height:auto}} .card{{margin-bottom:20px}} .table{{font-size:12px}} .card-body{{overflow:auto}}</style>
</head><body><div class="container-fluid">
  <h1 class="mb-3">Fraud Detection Dashboard (Offline)</h1>
  <p class="text-muted">Artifacts: <code>offline_artifacts/</code> — Mode: <b>{project_label}</b></p>

  <div class="card"><div class="card-body">
    <h2>Model Metrics</h2>
    {_json_table(m_val, "Validation metrics", cols=["accuracy","precision","recall","f1","roc_auc","pr_auc"])}
    <h3>Comparison (Accuracy / Precision / Recall / F1 / ROC AUC)</h3>
    {("<img src='"+cmp_b64+"'/>" if cmp_b64 else "<p><em>Comparison chart not found.</em></p>")}
    {_json_table(m_tst, "Held-out TEST metrics", cols=["accuracy","precision","recall","f1","roc_auc","pr_auc"])}
  </div></div>

  <div class="card"><div class="card-body">
    <h2>Threshold Tuning & Calibration</h2>
    {_json_table(thr_json, "F1-optimal thresholds")}
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
    <h2>Interpretability</h2>
    {img_if(xgb_imp, "XGB Feature Importance (top 20)")}
    <div class='row'>
      <div class='col-md-6'>{img_if(nlp_pos, "NLP top positive terms")}</div>
      <div class='col-md-6'>{img_if(nlp_neg, "NLP top negative terms")}</div>
    </div>
  </div></div>

  <div class="row g-3">
    <div class="col-lg-6"><div class="card"><div class="card-body">
      {_csv_tbl(head_csv, "Head (first rows)")}
    </div></div></div>
    <div class="col-lg-6"><div class="card"><div class="card-body">
      {_csv_tbl(desc_num_csv, "Describe (numeric)")}
      {_csv_tbl(desc_cat_csv, "Describe (categorical)")}
    </div></div></div>
  </div>

  <footer class="text-muted mt-4"><small>Rendered from local artifacts.</small></footer>
</div></body></html>"""
    return html
