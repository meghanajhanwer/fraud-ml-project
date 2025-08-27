import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from google.cloud import storage
from config.config import ARTIFACTS_GCS_PREFIX, PROJECT_ID

def _upload_png(fig, gcs_path: str):
    assert gcs_path.startswith("gs://")
    _, rest = gcs_path.split("gs://", 1)
    bucket_name, blob_name = rest.split("/", 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    storage.Client(project=PROJECT_ID).bucket(bucket_name).blob(blob_name).upload_from_string(
        buf.getvalue(), content_type="image/png"
    )

def save_confusion(y_true, y_pred, tag: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="#1f2937", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_yticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({tag.upper()})", pad=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)
    _upload_png(fig, f"{ARTIFACTS_GCS_PREFIX}/plots/confusion_{tag}.png")

def save_roc(y_true, y_proba, tag: str):
    if y_proba is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({tag.upper()})"); ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    _upload_png(fig, f"{ARTIFACTS_GCS_PREFIX}/plots/roc_{tag}.png")

def save_pr(y_true, y_proba, tag: str):
    if y_proba is None:
        return
    p, r, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(r, p, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall ({tag.upper()})"); ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    _upload_png(fig, f"{ARTIFACTS_GCS_PREFIX}/plots/pr_{tag}.png")
