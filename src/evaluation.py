import io, json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from google.cloud import storage
from config.config import ARTIFACTS_GCS_PREFIX, PROJECT_ID

def _upload_bytes(b: bytes, gcs_uri: str):
    assert gcs_uri.startswith("gs://")
    _, path = gcs_uri.split("gs://",1)
    bucket_name, blob_name = path.split("/",1)
    storage.Client(project=PROJECT_ID).bucket(bucket_name).blob(blob_name).upload_from_string(b)

def _upload_fig(fig, gcs_uri: str):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    _upload_bytes(buf.getvalue(), gcs_uri)

def save_metrics_comparison(metrics_list: List[Dict[str, Any]]):
    """Save metrics.json and comparison bar chart (XGB vs others)."""
    by_model = {}
    for m in metrics_list:
        if not m or "model_type" not in m: 
            continue
        by_model[m["model_type"]] = m

    _upload_bytes(
        json.dumps(by_model, indent=2).encode("utf-8"),
        f"{ARTIFACTS_GCS_PREFIX}/metrics/metrics.json"
    )

    models = list(by_model.keys())
    if not models:
        return

    order = ["accuracy","precision","recall","f1","roc_auc"]
    labels = ["Accuracy","Precision","Recall","F1","ROC AUC"]
    data = np.array([[float(by_model[m].get(k, 0) or 0) for k in order] for m in models], dtype=float)

    x = np.arange(len(labels))
    width = 0.16 if len(models) >= 4 else 0.22
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models)-1)/2) * width, data[i], width, label=m.upper())
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _upload_fig(fig, f"{ARTIFACTS_GCS_PREFIX}/metrics/metrics_compare.png")
