# Installation Guide

## Prerequisites
- Python 3.8+
- pip (Python package manager)
- (Optional) Docker and docker-compose for containerized deployment

---

## 1. Clone the Repository
```bash
git clone <repo-url>
cd <repo-folder>
```

---

## 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```

---

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Train Models
```bash
python train_model.py
```
This will generate:
- `models/phishing_url_detector.pkl` — Classification model
- `models/threat_actor_profiler.pkl` — Clustering model
- `models/cluster_profile_mapping.json` — Mapping of clusters to profiles

---

## 5. Run the Application
```bash
streamlit run app.py
```
Then open the provided localhost URL in your browser.

---

## 6. (Optional) Run via Docker
```bash
docker-compose up --build
```

---

## Notes
- The Threat Attribution step only runs for URLs predicted as **malicious**.
- For production deployment, ensure the `models/` directory is included.
