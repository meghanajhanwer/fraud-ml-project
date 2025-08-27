from typing import Dict, Any, Tuple
import io, joblib, tempfile, os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, accuracy_score
from google.cloud import storage
from config.config import PROJECT_ID, ARTIFACTS_GCS_PREFIX, RANDOM_SEED

def train_eval_nlp(train_text, y_train, val_text, y_val) -> Tuple[Dict[str, any], any]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))
    ])
    pipe.fit(train_text, y_train)

    y_pred = pipe.predict(val_text)
    try:
        y_proba = pipe.predict_proba(val_text)[:,1]
    except Exception:
        y_proba = None

    pr, rc, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_val, y_pred)
    metrics = {
        "model_type": "nlp",
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_val, y_proba)) if y_proba is not None else 0.0,
        "pr_auc": float(average_precision_score(y_val, y_proba)) if y_proba is not None else 0.0,
    }
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "model.joblib")
        joblib.dump(pipe, lp)
        bucket = ARTIFACTS_GCS_PREFIX.split("gs://",1)[1].split("/",1)[0]
        prefix = ARTIFACTS_GCS_PREFIX.split(bucket+"/",1)[1]
        storage.Client(project=PROJECT_ID).bucket(bucket).blob(f"{prefix}/models/nlp/model.joblib").upload_from_filename(lp)

    return metrics, pipe
