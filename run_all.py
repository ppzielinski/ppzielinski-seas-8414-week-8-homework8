# run_all.py
import json
import os
import re
import subprocess
import sys
from typing import Callable, List

import pandas as pd
from pycaret.classification import load_model as load_clf, predict_model as predict_clf
from pycaret.clustering import load_model as load_clu, predict_model as predict_clu

# ---------- Config (adjust names if you changed them in train_model.py) ----------
MODELS_DIR = "models"
CLASSIFIER_NAME = "phishing_url_detector"
CLUSTERER_NAME = "threat_actor_profiler"
MAPPING_JSON = os.path.join(MODELS_DIR, "cluster_profile_mapping.json")

# Optional files (nice-to-have if you export them from train_model.py)
EXPECTED_CLS_PATH = os.path.join(MODELS_DIR, "classifier_expected_columns.json")
EXPECTED_CLU_PATH = os.path.join(MODELS_DIR, "cluster_expected_columns.json")

# Features we explicitly set in our crafted tests (others will be back-filled with 0)
BASE_FEATURES = [
    "URL_Length",            #  1=long, 0=normal, -1=short
    "SSLfinal_State",        #  1=trusted, 0=suspicious, -1=none
    "having_Sub_Domain",     #  1=many, 0=one, -1=none
    "Prefix_Suffix",         #  1=yes, -1=no
    "having_IP_Address",     #  1=yes, -1=no
    "Shortining_Service",    #  1=yes, -1=no (intentional misspelling, match training)
    "having_At_Symbol",      #  1=yes, -1=no
    "Abnormal_URL",          #  1=yes, -1=no
    "has_political_keyword", #  1=yes,  0=no
]
# -------------------------------------------------------------------------------


def ensure_trained():
    """Train by running train_model.py if any artifacts are missing."""
    need = []
    if not os.path.isdir(MODELS_DIR):
        need.append("models dir")
    else:
        if not (
            os.path.exists(os.path.join(MODELS_DIR, CLASSIFIER_NAME))
            or os.path.exists(os.path.join(MODELS_DIR, CLASSIFIER_NAME + ".pkl"))
        ):
            need.append("classifier")
        if not (
            os.path.exists(os.path.join(MODELS_DIR, CLUSTERER_NAME))
            or os.path.exists(os.path.join(MODELS_DIR, CLUSTERER_NAME + ".pkl"))
        ):
            need.append("clusterer")
        if not os.path.exists(MAPPING_JSON):
            need.append("mapping json")

    if need:
        print(f"Artifacts missing ({', '.join(need)}). Training by running: {sys.executable} train_model.py")
        res = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            raise SystemExit("Training failed; see logs above.")


def load_artifacts():
    clf = load_clf(os.path.join(MODELS_DIR, CLASSIFIER_NAME))
    clu = load_clu(os.path.join(MODELS_DIR, CLUSTERER_NAME))
    with open(MAPPING_JSON, "r") as f:
        mapping = {int(k): v for k, v in json.load(f).items()}
    return clf, clu, mapping


def row(d: dict) -> pd.DataFrame:
    """Create one-row DF with our base features; missing fields default to 0."""
    return pd.DataFrame([{k: d.get(k, 0) for k in BASE_FEATURES}])


def samples() -> List[tuple]:
    """Three clear exemplars aligned with your generator & UI."""
    org = row(dict(
        URL_Length=1, SSLfinal_State=0, having_Sub_Domain=1, Prefix_Suffix=1,
        having_IP_Address=1, Shortining_Service=1, having_At_Symbol=1,
        Abnormal_URL=1, has_political_keyword=0
    ))
    state = row(dict(
        URL_Length=0, SSLfinal_State=1, having_Sub_Domain=0, Prefix_Suffix=1,
        having_IP_Address=-1, Shortining_Service=-1, having_At_Symbol=-1,
        Abnormal_URL=0, has_political_keyword=0
    ))
    hack = row(dict(
        URL_Length=0, SSLfinal_State=0, having_Sub_Domain=0, Prefix_Suffix=0,
        having_IP_Address=-1, Shortining_Service=0, having_At_Symbol=0,
        Abnormal_URL=0, has_political_keyword=1
    ))
    return [
        ("Organized Cybercrime", org),
        ("State-Sponsored", state),
        ("Hacktivist", hack),
    ]


def is_malicious_label(val) -> bool:
    s = str(val).upper()
    return (val == 1) or ("MAL" in s) or (s == "MALICIOUS")


def predict_with_backfill(predict_fn: Callable, model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calls PyCaret predict_model(model, df). If KeyError lists missing columns,
    add those columns (filled with 0) and retry.
    """
    cur = df.copy()
    for _ in range(3):  # avoid infinite loops
        try:
            return predict_fn(model, cur)
        except KeyError as e:
            msg = str(e)
            # Pattern: "['A', 'B'] not in index"
            m = re.search(r"\[(.*?)\]\s+not in index", msg)
            if not m:
                raise
            raw = m.group(1)
            missing = [c.strip().strip("'").strip('"') for c in raw.split(",")]
            for col in missing:
                if col not in cur.columns:
                    cur[col] = 0
    # final attempt without catching
    return predict_fn(model, cur)


def add_missing(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """Ensure all expected columns exist; fill missing with 0; align order."""
    cur = df.copy()
    for c in expected_cols:
        if c not in cur.columns:
            cur[c] = 0
    # Keep expected order first, then any extras (PyCaret ignores extras anyway)
    return cur[expected_cols + [c for c in cur.columns if c not in expected_cols]]


def predict_classifier(clf, df: pd.DataFrame) -> pd.DataFrame:
    """Predict using classifier, preferring expected-columns JSON when available."""
    if os.path.exists(EXPECTED_CLS_PATH):
        with open(EXPECTED_CLS_PATH, "r") as f:
            expected = json.load(f)
        df_adj = add_missing(df, expected)
        return predict_clf(clf, df_adj)
    return predict_with_backfill(predict_clf, clf, df)


def predict_clusterer(clu, df: pd.DataFrame) -> pd.DataFrame:
    """Predict using clusterer, preferring expected-columns JSON when available."""
    if os.path.exists(EXPECTED_CLU_PATH):
        with open(EXPECTED_CLU_PATH, "r") as f:
            expected = json.load(f)
        df_adj = add_missing(df, expected)
        return predict_clu(clu, df_adj)
    return predict_with_backfill(predict_clu, clu, df)


def to_cluster_id(val):
    """
    Accepts 0, '0', 'Cluster 0', 'cluster_0', etc. -> returns int cluster id.
    """
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val)
    m = re.search(r'(\d+)$', s.strip())
    if not m:
        raise ValueError(f"Cannot parse cluster id from: {val!r}")
    return int(m.group(1))


def main():
    ensure_trained()
    clf, clu, mapping = load_artifacts()

    all_ok = True
    for expected, df in samples():
        print(f"\n=== Testing expected: {expected} ===")
        # 1) classification
        clf_pred = predict_classifier(clf, df.copy())
        # PyCaret may expose 'prediction_label' or 'Label'
        if "prediction_label" in clf_pred.columns:
            label = clf_pred["prediction_label"].iloc[0]
        elif "Label" in clf_pred.columns:
            label = clf_pred["Label"].iloc[0]
        else:
            raise SystemExit(f"Cannot find prediction column in classifier output: {clf_pred.columns.tolist()}")
        print("Classifier label:", label)

        if not is_malicious_label(label):
            print("Classifier says BENIGN → skipping attribution (matches app logic).")
            continue

        # 2) clustering + mapping
        clu_pred = predict_clusterer(clu, df.copy())
        if "Cluster" not in clu_pred.columns:
            raise SystemExit(f"Clustering output missing 'Cluster' column: {clu_pred.columns.tolist()}")
        raw_cluster = clu_pred["Cluster"].iloc[0]
        cluster_id = to_cluster_id(raw_cluster)
        actor = mapping.get(cluster_id, f"Unknown({cluster_id})")
        print("Cluster:", raw_cluster, "→", cluster_id, "→", actor)

        if actor != expected:
            print(f"❌ Attribution mismatch. Expected {expected}, got {actor}")
            all_ok = False
        else:
            print("✅ Attribution OK")

    # Optional centroid preview (helps interpret drift)
    if hasattr(clu, "cluster_centers_"):
        centers = clu.cluster_centers_
        # Use whichever features the clusterer was actually trained on, if available
        trained_cols = None
        if os.path.exists(EXPECTED_CLU_PATH):
            with open(EXPECTED_CLU_PATH, "r") as f:
                trained_cols = json.load(f)
        else:
            # Fallback: BASE_FEATURES (we may not know all training columns, but this is a peek)
            trained_cols = BASE_FEATURES

        idx = {k: i for i, k in enumerate(trained_cols) if i < centers.shape[1]}
        def g(cvec, name):
            j = idx.get(name, None)
            return round(float(cvec[j]), 3) if j is not None else None

        print("\nCentroid preview (key features):")
        for i, cvec in enumerate(centers):
            print(i, {
                'political': g(cvec, "has_political_keyword"),
                'shortener': g(cvec, "Shortining_Service"),
                'ip':        g(cvec, "having_IP_Address"),
                'abnormal':  g(cvec, "Abnormal_URL"),
                'ssl':       g(cvec, "SSLfinal_State"),
            })

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()