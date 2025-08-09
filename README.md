# Mini-SOAR Development Roadmap

This project is a **Mini Security Orchestration, Automation, and Response (SOAR)** platform for phishing detection and automated incident response.  
It uses:
- **Predictive Analytics** (PyCaret) to classify phishing URLs  
- **Prescriptive Analytics** (Generative AI) to generate structured incident response plans  
- **Streamlit** for the interactive UI

---

## Project Roadmap

### Completed
- **Environment Setup**
  - Defined `Dockerfile` for containerized environment
  - Created `docker-compose.yml` for orchestration
  - Added `Makefile` for workflow automation

### In Progress / To Do
- **Predictive Engine** (`train_model.py`)
  - [ ] Create synthetic data generator
  - [ ] Integrate PyCaret to train and save the classification model

- **Prescriptive Engine** (`genai_prescriptions.py`)
  - [ ] Define a robust prompt to produce JSON-formatted output
  - [ ] Implement API calls to Generative AI services (OpenAI, Google GenAI)

- **UI & Orchestration** (`app.py`)
  - [ ] Build input sidebar for URL features
  - [ ] Orchestrate predictive (PyCaret) and prescriptive (GenAI) workflows
  - [ ] Design output tabs for results display (analysis summary, visual insights, prescriptive plan)

---

## Goal
Deliver a **fully functional, documented, and automated web application** that:
1. Accepts suspicious URL feature inputs
2. Predicts if the URL is malicious
3. Generates an AI-driven response plan
4. Presents results in a clean, interactive UI
