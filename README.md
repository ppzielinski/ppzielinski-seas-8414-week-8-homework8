# Cognitive SOAR: From Prediction to Attribution

## Overview
This project extends the original Mini-SOAR application from a basic phishing URL detector into a **two-stage intelligent system**.  
In addition to predicting whether a URL is malicious, the application now performs **threat attribution** for malicious URLs, 
grouping them into likely **threat actor profiles** using unsupervised clustering.

This enhancement transforms the tool from answering *"What will happen?"* to *"Who might be behind it?"*, 
providing crucial context for analysts in a modern Security Operations Center (SOC).

---

## New Features
- **Synthetic Feature Engineering** to simulate realistic threat actor patterns.
- Added `has_political_keyword` feature to identify Hacktivist activity.
- **K-Means clustering model** trained on malicious-only data to identify 3 distinct actor profiles:
  - State-Sponsored
  - Organized Cybercrime
  - Hacktivist
- JSON mapping from cluster IDs to human-readable actor profiles.
- New **"Threat Attribution" tab** in the Streamlit UI with actor descriptions.
- Optional **feature importance visualization** for analyst training and onboarding.

---

## Technology Stack
- Python 3.x
- PyCaret (Classification & Clustering)
- Streamlit (Web UI)
- Docker (Deployment)
- JSON (Cluster mapping storage)

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
```bash
python train_model.py
```
This will generate:
- `models/phishing_url_detector.pkl` — classification model
- `models/threat_actor_profiler.pkl` — clustering model
- `models/cluster_profile_mapping.json` — mapping from cluster IDs to actor profiles

### 4. Run the Application
```bash
streamlit run app.py
```
Then open the displayed URL in your browser to interact with the UI.

> **Note:** The Threat Attribution step runs **only** if the classification model predicts the URL is malicious.

---

## Example Output

| Example URL Features             | Verdict     | Threat Actor Profile   |
|----------------------------------|-------------|------------------------|
| SSLfinal_State=1, Prefix_Suffix=1| Malicious   | State-Sponsored        |
| Shortining_Service=1, IP=1       | Malicious   | Organized Cybercrime   |
| has_political_keyword=1          | Malicious   | Hacktivist             |
| SSLfinal_State=1, Prefix_Suffix=0| Benign      | N/A                    |

---

## Project Workflow

```plaintext
User Input → Classification Model (Malicious/Benign) 
    ├── Benign → Display verdict only
    └── Malicious → Clustering Model (K-Means) → Map to Threat Actor Profile → Display in Threat Attribution tab
```

---

---

## License
This project is released under the MIT License.
