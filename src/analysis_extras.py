import io, os, json, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from google.cloud import storage
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,roc_auc_score, average_precision_score, brier_score_loss,precision_recall_curve,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from config.config import PROJECT_ID, ARTIFACTS_GCS_PREFIX, RANDOM_SEED,TARGET_COL, ID_COL, TIME_COL
from src.data_extraction import load_curated
from src.feature_engineering import add_simple_features
from src.preprocessing import split_data, build_tabular_transformer, smote_fit_resample
from src.model_xgb import train_eval_xgb
from src.model_nlp import train_eval_nlp 

def _client():
    return storage.Client(project=PROJECT_ID)

def _upload_bytes(b: bytes, gcs_uri: str):
    assert gcs_uri.startswith("gs://")
    _, path = gcs_uri.split("gs://",1)
    bucket_name, blob_name = path.split("/",1)
    _client().bucket(bucket_name).blob(blob_name).upload_from_string(b)

def _upload_fig(fig, gcs_uri: str, dpi=160):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    _upload_bytes(buf.getvalue(), gcs_uri)

def _metrics_dict(y_true, y_pred, y_proba):
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None and len(np.unique(y_true))>1 else 0.0
    pra = float(average_precision_score(y_true, y_proba)) if y_proba is not None and len(np.unique(y_true))>1 else 0.0
    return {"accuracy":float(acc),"precision":float(pr),"recall":float(rc),"f1":float(f1),"roc_auc":roc,"pr_auc":pra}

def prepare_data(drop_id_like=False):
    """Return splits and featurizer (tabular)."""
    df = load_curated().sort_values(["AccountID", TIME_COL]).reset_index(drop=True)
    df = add_simple_features(df)
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)

    drop_cols = [TARGET_COL, ID_COL, TIME_COL, "PreviousTransactionDate", "nlp_text"]
    if drop_id_like:
        for c in ["DeviceID","IP_Address","MerchantID","Location"]:
            if c in df.columns: drop_cols.append(c)

    tabular_cols = [c for c in df.columns if c not in drop_cols]
    preproc, _ = build_tabular_transformer(df[tabular_cols])

    X_train = preproc.fit_transform(df_train[tabular_cols])
    X_val   = preproc.transform(df_val[tabular_cols])
    X_test  = preproc.transform(df_test[tabular_cols])
    X_train_bal, y_train_bal = smote_fit_resample(X_train, y_train)
    return (df_train, df_val, df_test, y_train, y_val, y_test,
            X_train_bal, y_train_bal, X_val, X_test, preproc, tabular_cols)

def train_xgb_and_eval(X_train_bal, y_train, X_val, y_val, X_test, y_test):
    m_val, model = train_eval_xgb(X_train_bal, y_train, X_val, y_val)
    try:
        pro_val = model.predict_proba(X_val)[:,1]
        pro_tst = model.predict_proba(X_test)[:,1]
    except Exception:
        pro_val = pro_tst = None
    pred_val = model.predict(X_val)
    pred_tst = model.predict(X_test)
    val = _metrics_dict(y_val, pred_val, pro_val)
    tst = _metrics_dict(y_test, pred_tst, pro_tst)
    return model, val, tst, pro_val

def train_nlp_and_eval(train_text, y_train, val_text, y_val, test_text, y_test):
    m_val, pipe = train_eval_nlp(train_text, y_train, val_text, y_val)
    try:
        pro_val = pipe.predict_proba(val_text)[:,1]
        pro_tst = pipe.predict_proba(test_text)[:,1]
    except Exception:
        pro_val = pro_tst = None
    pred_val = pipe.predict(val_text)
    pred_tst = pipe.predict(test_text)
    val = _metrics_dict(y_val, pred_val, pro_val)
    tst = _metrics_dict(y_test, pred_tst, pro_tst)
    return pipe, val, tst, pro_val

def best_f1_threshold(y_true, y_proba):
    if y_proba is None: return 0.5, None
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = np.where((p+r)>0, 2*p*r/(p+r), 0)
    idx = int(np.nanargmax(f1))
    thr = float(t[idx-1]) if idx>0 and idx-1 < len(t) else 0.5
    return thr, float(f1[idx])

def save_tuned_confusion(y_true, y_proba, tag: str):
    if y_proba is None: return None
    thr, best_f1 = best_f1_threshold(y_true, y_proba)
    y_pred = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha="center",va="center",color="black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
    ax.set_xlabel(f"Predicted (thr={thr:.3f})"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion @ F1-opt ({tag.upper()})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _upload_fig(fig, f"{ARTIFACTS_GCS_PREFIX}/plots/confusion_{tag}_tuned.png")
    return {"threshold":thr,"best_f1":best_f1}

def save_calibration_plot(y_true, y_proba, tag: str):
    if y_proba is None: return None
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8,4.0))
    ax.plot(mean_pred, frac_pos, marker="o", label="Raw")
    ax.plot([0,1],[0,1], linestyle="--", label="Perfect")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability Diagram ({tag.upper()})"); ax.legend()
    _upload_fig(fig, f"{ARTIFACTS_GCS_PREFIX}/plots/calibration_{tag}.png")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(y_proba.reshape(-1,1), y_true)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_proba, y_true)
    brier_raw = brier_score_loss(y_true, y_proba)
    brier_platt = brier_score_loss(y_true, lr.predict_proba(y_proba.reshape(-1,1))[:,1])
    brier_iso = brier_score_loss(y_true, iso.predict(y_proba))
    return {"brier_raw":float(brier_raw),"brier_platt":float(brier_platt),"brier_isotonic":float(brier_iso)}

def save_xgb_feature_importance(model, preproc, top_k=20):
    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(model.n_features_in_)])
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    pairs = []
    for k,v in gain.items():
        try:
            idx = int(k.replace("f",""))
            name = feat_names[idx]
        except Exception:
            name = k
        pairs.append((name, v))
    if not pairs:
        return
    imp = pd.DataFrame(pairs, columns=["feature","gain"]).sort_values("gain", ascending=False)
    csv_b = imp.to_csv(index=False).encode()
    _upload_bytes(csv_b, f"{ARTIFACTS_GCS_PREFIX}/interpretability/xgb_feature_importance.csv")
    top = imp.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, top_k*0.28)))
    ax.barh(top["feature"], top["gain"])
    ax.set_title("XGB Feature Importance (gain, top 20)")
    ax.set_xlabel("Gain")
    _upload_fig(fig, f"{ARTIFACTS_GCS_PREFIX}/interpretability/xgb_feature_importance_top20.png")

def save_nlp_top_terms(pipe, top_k=20):
    try:
        vec: TfidfVectorizer = pipe.named_steps["tfidf"]
        lr:  LogisticRegression = pipe.named_steps["lr"]
    except Exception:
        return
    feats = np.array(vec.get_feature_names_out())
    coefs = lr.coef_.ravel()
    idx_pos = np.argsort(coefs)[-top_k:][::-1]
    idx_neg = np.argsort(coefs)[:top_k]
    pos = pd.DataFrame({"term": feats[idx_pos], "weight": coefs[idx_pos]})
    neg = pd.DataFrame({"term": feats[idx_neg], "weight": coefs[idx_neg]})
    _upload_bytes(pos.to_csv(index=False).encode(), f"{ARTIFACTS_GCS_PREFIX}/interpretability/nlp_top_terms_positive.csv")
    _upload_bytes(neg.to_csv(index=False).encode(), f"{ARTIFACTS_GCS_PREFIX}/interpretability/nlp_top_terms_negative.csv")
    for name, df in [("positive", pos), ("negative", neg.iloc[::-1])]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(df["term"], df["weight"])
        ax.set_title(f"NLP top {top_k} {name} terms")
        ax.set_xlabel("Coefficient (LR)")
        _upload_fig(fig, f"{ARTIFACTS_GCS_PREFIX}/interpretability/nlp_top_terms_{name}.png")

def run_ablation_id_drop():
    (df_tr, df_va, df_te, y_tr, y_va, y_te,
     Xtr_b, ytr_b, Xva, Xte, pre, tab) = prepare_data(drop_id_like=False)
    mdl_base, val_base, test_base, pro_base = train_xgb_and_eval(Xtr_b, ytr_b, Xva, y_va, Xte, y_te)
    (df_tr2, df_va2, df_te2, y_tr2, y_va2, y_te2,
     Xtr_b2, ytr_b2, Xva2, Xte2, pre2, tab2) = prepare_data(drop_id_like=True)
    mdl_ab, val_ab, test_ab, pro_ab = train_xgb_and_eval(Xtr_b2, ytr_b2, Xva2, y_va2, Xte2, y_te2)

    result = {"baseline_val":val_base, "ablated_val":val_ab,
              "baseline_test":test_base, "ablated_test":test_ab}
    _upload_bytes(json.dumps(result, indent=2).encode(), f"{ARTIFACTS_GCS_PREFIX}/ablation/ablation_results.json".replace(" ",""))
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(2); width=0.35
    ax.bar(x - width/2, [val_base["f1"], test_base["f1"]], width, label="Baseline")
    ax.bar(x + width/2, [val_ab["f1"],   test_ab["f1"]],   width, label="Ablated")
    ax.set_xticks(x, ["Validation F1","Test F1"]); ax.set_ylim(0,1.05)
    ax.set_title("XGB F1 — Baseline vs ID-drop")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _upload_fig(fig, f"{ARTIFACTS_GCS_PREFIX}/ablation/ablation_compare_f1.png")
    return (mdl_base, pre), (mdl_ab, pre2), result

def run_all_and_save():
    (df_tr, df_va, df_te, y_tr, y_va, y_te,
     Xtr_b, ytr_b, Xva, Xte, pre, tab) = prepare_data(drop_id_like=False)
    xgb_model, val_xgb, tst_xgb, pro_val_xgb = train_xgb_and_eval(Xtr_b, ytr_b, Xva, y_va, Xte, y_te)
    xgb_tuned = save_tuned_confusion(y_va, pro_val_xgb, "xgb")
    xgb_cal = save_calibration_plot(y_va, pro_val_xgb, "xgb")
    save_xgb_feature_importance(xgb_model, pre)
    train_text = df_tr["nlp_text"].fillna(""); val_text = df_va["nlp_text"].fillna(""); test_text = df_te["nlp_text"].fillna("")
    nlp_model, val_nlp, tst_nlp, pro_val_nlp = train_nlp_and_eval(train_text, y_tr, val_text, y_va, test_text, y_te)
    nlp_tuned = save_tuned_confusion(y_va, pro_val_nlp, "nlp")
    nlp_cal = save_calibration_plot(y_va, pro_val_nlp, "nlp")
    save_nlp_top_terms(nlp_model)
    test_metrics = {"xgb":tst_xgb, "nlp":tst_nlp}
    _upload_bytes(json.dumps(test_metrics, indent=2).encode(), f"{ARTIFACTS_GCS_PREFIX}/metrics/metrics_test.json")
    thresholds = {"xgb": xgb_tuned or {}, "nlp": nlp_tuned or {}}
    _upload_bytes(json.dumps(thresholds, indent=2).encode(), f"{ARTIFACTS_GCS_PREFIX}/metrics/thresholds.json")
    if xgb_cal or nlp_cal:
        _upload_bytes(json.dumps({"xgb":xgb_cal, "nlp":nlp_cal}, indent=2).encode(),
                      f"{ARTIFACTS_GCS_PREFIX}/metrics/calibration_brier.json")
    run_ablation_id_drop()
