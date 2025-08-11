import os
import json
import numpy as np
import pandas as pd

from pycaret.classification import (
    setup as clf_setup,
    compare_models as clf_compare,
    finalize_model as clf_finalize,
    save_model as clf_save,
    plot_model as clf_plot,
)

from pycaret.clustering import (
    setup as clu_setup,
    create_model as clu_create,
    save_model as clu_save,
)

# =========================
# 1) Synthetic data
# =========================

_rng = np.random.default_rng(42)

def _bern(p, n, pos=1, neg=-1):
    return np.where(_rng.random(n) < p, pos, neg)

def _tri(p_neg, p_zero, p_pos, n):
    return _rng.choice([-1, 0, 1], size=n, p=[p_neg, p_zero, p_pos])

def generate_synthetic_data(num_samples: int = 600) -> pd.DataFrame:
    """
    Extend the original generator with one NEW feature:
    - has_political_keyword (0/1) to make Hacktivist separable
    Slightly nudge distributions so Organized (noisy), State (clean), Hacktivist (political) are distinct.
    """
    profiles = ["State-Sponsored", "Organized Cybercrime", "Hacktivist"]
    per_profile = num_samples // len(profiles)
    parts = []

    for profile in profiles:
        d = pd.DataFrame(index=range(per_profile))

        if profile == "State-Sponsored":
            d["having_IP_Address"] = _bern(0.10, per_profile, 1, -1)
            d["URL_Length"]        = _tri(0.30, 0.40, 0.30, per_profile)
            d["Shortining_Service"]= _bern(0.10, per_profile, 1, -1)
            d["having_At_Symbol"]  = _bern(0.10, per_profile, 1, -1)
            d["Prefix_Suffix"]     = _bern(0.60, per_profile, 1, -1)
            d["having_Sub_Domain"] = _tri(0.20, 0.50, 0.30, per_profile)
            d["SSLfinal_State"]    = _tri(0.05, 0.15, 0.80, per_profile)   # trusted mostly
            d["URL_of_Anchor"]     = _tri(0.30, 0.30, 0.40, per_profile)
            d["Links_in_tags"]     = _tri(0.30, 0.30, 0.40, per_profile)
            d["SFH"]               = _tri(0.30, 0.30, 0.40, per_profile)
            d["Abnormal_URL"]      = _bern(0.30, per_profile, 1, -1)
            d["has_political_keyword"] = np.zeros(per_profile, dtype=int)

        elif profile == "Organized Cybercrime":
            d["having_IP_Address"] = _bern(0.85, per_profile, 1, -1)
            d["URL_Length"]        = _tri(0.05, 0.25, 0.70, per_profile)   # mostly long
            d["Shortining_Service"]= _bern(0.90, per_profile, 1, -1)
            d["having_At_Symbol"]  = _bern(0.50, per_profile, 1, -1)
            d["Prefix_Suffix"]     = _bern(0.80, per_profile, 1, -1)
            d["having_Sub_Domain"] = _tri(0.15, 0.25, 0.60, per_profile)   # many
            d["SSLfinal_State"]    = _tri(0.75, 0.20, 0.05, per_profile)   # not trusted
            d["URL_of_Anchor"]     = _tri(0.60, 0.20, 0.20, per_profile)
            d["Links_in_tags"]     = _tri(0.60, 0.20, 0.20, per_profile)
            d["SFH"]               = _tri(0.60, 0.20, 0.20, per_profile)
            d["Abnormal_URL"]      = _bern(0.90, per_profile, 1, -1)
            d["has_political_keyword"] = np.zeros(per_profile, dtype=int)

        else:  # Hacktivist
            d["having_IP_Address"] = _bern(0.20, per_profile, 1, -1)
            d["URL_Length"]        = _tri(0.30, 0.40, 0.30, per_profile)
            d["Shortining_Service"]= _bern(0.30, per_profile, 1, -1)
            d["having_At_Symbol"]  = _bern(0.40, per_profile, 1, -1)
            d["Prefix_Suffix"]     = _bern(0.40, per_profile, 1, -1)
            d["having_Sub_Domain"] = _tri(0.40, 0.45, 0.15, per_profile)   # none/one common
            d["SSLfinal_State"]    = _tri(0.45, 0.45, 0.10, per_profile)   # rarely trusted
            d["URL_of_Anchor"]     = _tri(0.40, 0.30, 0.30, per_profile)
            d["Links_in_tags"]     = _tri(0.40, 0.30, 0.30, per_profile)
            d["SFH"]               = _tri(0.40, 0.30, 0.30, per_profile)
            d["Abnormal_URL"]      = _bern(0.35, per_profile, 1, -1)
            d["has_political_keyword"] = np.ones(per_profile, dtype=int)   # key separator

        d["label"] = 1
        parts.append(d)

    # benign background
    benign_n = num_samples // 2
    benign = pd.DataFrame({
        "having_IP_Address":  _bern(0.05, benign_n, 1, -1),
        "URL_Length":         _tri(0.30, 0.60, 0.10, benign_n),
        "Shortining_Service": _bern(0.10, benign_n, 1, -1),
        "having_At_Symbol":   _bern(0.05, benign_n, 1, -1),
        "Prefix_Suffix":      _bern(0.10, benign_n, 1, -1),
        "having_Sub_Domain":  _tri(0.50, 0.40, 0.10, benign_n),
        "SSLfinal_State":     _tri(0.05, 0.15, 0.80, benign_n),
        "URL_of_Anchor":      _tri(0.10, 0.20, 0.70, benign_n),
        "Links_in_tags":      _tri(0.10, 0.20, 0.70, benign_n),
        "SFH":                _tri(0.10, 0.10, 0.80, benign_n),
        "Abnormal_URL":       _bern(0.10, benign_n, 1, -1),
        "has_political_keyword": np.zeros(benign_n, dtype=int),
        "label": 0
    })

    df = pd.concat(parts + [benign], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# =========================
# 2) Mapping derivation
# =========================

def _derive_mapping_from_centroids(centroids: np.ndarray, cols: list[str]) -> dict[str, str]:
    idx = {c: i for i, c in enumerate(cols)}

    def s_h(c):   # Hacktivist → highest political
        j = idx.get("has_political_keyword")
        return centroids[c, j] if j is not None else -1e9

    def s_state(c):  # State → trusted SSL, low noise
        s = 0.0
        j = idx.get("SSLfinal_State");        s += centroids[c, j] if j is not None else 0.0
        for k in ("Shortining_Service","having_IP_Address","Abnormal_URL","having_At_Symbol"):
            j = idx.get(k);                   s -= centroids[c, j] if j is not None else 0.0
        return s

    clusters = list(range(centroids.shape[0]))
    hk = max(clusters, key=s_h)
    rem = [c for c in clusters if c != hk]
    st = max(rem, key=s_state) if rem else hk
    rem = [c for c in rem if c != st]
    org = rem[0] if rem else hk

    return {str(org): "Organized Cybercrime", str(st): "State-Sponsored", str(hk): "Hacktivist"}

# =========================
# 3) Train & save
# =========================

def train():
    MODELS_DIR = "models"
    DATA_DIR   = "data"
    CLF_NAME   = os.path.join(MODELS_DIR, "phishing_url_detector")
    CLU_NAME   = os.path.join(MODELS_DIR, "threat_actor_profiler")
    MAP_JSON   = os.path.join(MODELS_DIR, "cluster_profile_mapping.json")
    FEAT_PNG   = os.path.join(MODELS_DIR, "feature_importance.png")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Data
    data = generate_synthetic_data()
    data.to_csv(os.path.join(DATA_DIR, "phishing_synthetic.csv"), index=False)

    # Classification (drop 'profile' if present — we didn't add it here, but stay safe)
    clf_setup(data=data, target="label", session_id=42, verbose=False)
    best = clf_compare(n_select=1, include=["rf", "et", "lightgbm"])
    final = clf_finalize(best)
    try:
        clf_plot(final, plot="feature", save=True)
        if os.path.exists("Feature Importance.png"):
            os.replace("Feature Importance.png", FEAT_PNG)
    except Exception:
        pass
    clf_save(final, CLF_NAME)

    # Clustering on MALICIOUS ONLY (drop label afterwards)
    X = data.query("label == 1").drop(columns=["label"])
    clu_setup(data=X, session_id=42, verbose=False, normalize=True)
    clu_model = clu_create("kmeans", num_clusters=3)
    clu_save(clu_model, CLU_NAME)

    # Mapping
    centroids = getattr(clu_model, "cluster_centers_", None)
    if centroids is None:
        mapping = {"0": "Organized Cybercrime", "1": "State-Sponsored", "2": "Hacktivist"}
    else:
        mapping = _derive_mapping_from_centroids(centroids, list(X.columns))

    with open(MAP_JSON, "w") as f:
        json.dump(mapping, f)

    print(f"Saved models to '{MODELS_DIR}' and mapping: {mapping}")

if __name__ == "__main__":
    train()