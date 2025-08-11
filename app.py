import os
import json
import time
import pandas as pd
import streamlit as st

from pycaret.classification import load_model as load_clf_model, predict_model as predict_clf_model
from pycaret.clustering import load_model as load_clu_model, predict_model as predict_clu_model
from genai_prescriptions import generate_prescription

st.set_page_config(page_title="GenAI-Powered Phishing SOAR", layout="wide")

@st.cache_resource
def load_assets():
    model = load_clf_model("models/phishing_url_detector") if os.path.exists("models/phishing_url_detector.pkl") else None
    cluster_model = load_clu_model("models/threat_actor_profiler") if os.path.exists("models/threat_actor_profiler.pkl") else None

    mapping_path = "models/cluster_profile_mapping.json"
    cluster_mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            cluster_mapping = json.load(f)

    feature_plot = "models/feature_importance.png" if os.path.exists("models/feature_importance.png") else None
    return model, cluster_model, cluster_mapping, feature_plot

model, cluster_model, cluster_mapping, feature_plot = load_assets()

if not model:
    st.error("Classification model not found. Please retrain (run train_model.py).")
    st.stop()

with st.sidebar:
    st.title("URL Feature Input")
    form_values = {
        "url_length":        st.select_slider("URL Length", ["Short", "Normal", "Long"], "Long"),
        "ssl_state":         st.select_slider("SSL Status", ["Trusted", "Suspicious", "None"], "Suspicious"),
        "sub_domain":        st.select_slider("Sub-domain", ["None", "One", "Many"], "One"),
        "prefix_suffix":     st.checkbox("Has Prefix/Suffix", True),
        "has_ip":            st.checkbox("Uses IP Address", False),
        "short_service":     st.checkbox("Is Shortened", False),
        "at_symbol":         st.checkbox("Has '@'", False),
        "abnormal_url":      st.checkbox("Is Abnormal", True),
        "political_keyword": st.checkbox("Has Political Keyword", False),
    }
    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI"])
    submitted = st.button("Analyze & Respond", use_container_width=True, type="primary")

st.title("GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Provide URL features in the sidebar and click **Analyze & Respond**.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot, caption="Global feature importance from the classifier.")
else:
    # Map sidebar inputs to model-friendly feature encodings
    input_dict = {
        "having_IP_Address":   1 if form_values["has_ip"] else -1,
        "URL_Length":          -1 if form_values["url_length"] == "Short" else (0 if form_values["url_length"] == "Normal" else 1),
        "Shortining_Service":  1 if form_values["short_service"] else -1,
        "having_At_Symbol":    1 if form_values["at_symbol"] else -1,
        "Prefix_Suffix":       1 if form_values["prefix_suffix"] else -1,
        "having_Sub_Domain":   -1 if form_values["sub_domain"] == "None" else (0 if form_values["sub_domain"] == "One" else 1),
        "SSLfinal_State":      -1 if form_values["ssl_state"] == "None" else (0 if form_values["ssl_state"] == "Suspicious" else 1),
        "Abnormal_URL":        1 if form_values["abnormal_url"] else -1,
        "URL_of_Anchor":       0,
        "Links_in_tags":       0,
        "SFH":                 0,
        "has_political_keyword": 1 if form_values["political_keyword"] else 0,
    }
    input_df = pd.DataFrame([input_dict])

    # Simple risk view
    risk_scores = {
        "Bad SSL":            25 if input_dict["SSLfinal_State"] < 1 else 0,
        "Abnormal URL":       20 if input_dict["Abnormal_URL"] == 1 else 0,
        "Prefix/Suffix":      15 if input_dict["Prefix_Suffix"] == 1 else 0,
        "Shortened URL":      15 if input_dict["Shortining_Service"] == 1 else 0,
        "Complex Sub-domain": 10 if input_dict["having_Sub_Domain"] == 1 else 0,
        "Long URL":           10 if input_dict["URL_Length"] == 1 else 0,
        "Uses IP Address":     5 if input_dict["having_IP_Address"] == 1 else 0,
        "Political Keyword":   5 if input_dict["has_political_keyword"] == 1 else 0,
    }
    risk_df = (
        pd.DataFrame(list(risk_scores.items()), columns=["Feature", "Risk Contribution"])
        .sort_values("Risk Contribution", ascending=False)
    )

    with st.status("Executing SOAR playbook...", expanded=True) as status:
        st.write("Step 1: Predictive Analysis")
        prediction = predict_clf_model(model, data=input_df)
        is_malicious = prediction["prediction_label"].iloc[0] == 1

        st.write(f"Step 2: Verdict → {'MALICIOUS' if is_malicious else 'BENIGN'}")

        actor_profile = None
        prescription = None

        if is_malicious and cluster_model is not None and cluster_mapping:
            st.write("Step 3: Threat Attribution")
            clu_result = predict_clu_model(cluster_model, data=input_df)
            cid_raw = clu_result["Cluster"].iloc[0]
            try:
                cluster_id = int(str(cid_raw).replace("Cluster ", ""))
            except Exception:
                cluster_id = int(cid_raw)
            actor_profile = cluster_mapping.get(str(cluster_id), f"Unknown Profile (Cluster {cluster_id})")

        if is_malicious:
            st.write(f"Step 4: Generating response with {genai_provider}")
            prescription = generate_prescription(genai_provider, input_dict)

        status.update(label="Playbook complete.", state="complete", expanded=False)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Analysis Summary", "Visual Insights", "Prescriptive Plan", "Threat Attribution"])

    with tab1:
        st.subheader("Verdict and Confidence")
        if is_malicious:
            st.error("**Prediction: Malicious Phishing URL**")
        else:
            st.success("**Prediction: Benign URL**")
        score = prediction["prediction_score"].iloc[0]
        st.metric("Malicious Confidence Score", f"{score:.2%}" if is_malicious else f"{1 - score:.2%}")

    with tab2:
        st.subheader("Visual Analysis")
        st.write("#### Risk Contribution by Feature")
        st.bar_chart(risk_df.set_index("Feature"))
        if feature_plot:
            st.write("#### Classifier Feature Importance")
            st.image(feature_plot)

    with tab3:
        st.subheader("Actionable Response Plan")
        if prescription:
            st.success("A prescriptive response plan has been generated.")
            actions = prescription.get("recommended_actions", [])
            if actions:
                for i, action in enumerate(actions, 1):
                    if isinstance(action, dict):
                        st.markdown(f"**{i}. {action.get('action','N/A')}:** {action.get('details','N/A')}")
                    else:
                        st.markdown(f"**{i}.** {action}")
            st.write("#### Communication Draft")
            st.text_area("Draft", prescription.get("communication_draft", ""), height=150)
        else:
            st.info("URL was benign. No plan needed.")

    with tab4:
        st.subheader("Threat Actor Attribution")
        if is_malicious and actor_profile:
            st.write(f"**Predicted Threat Actor Profile:** {actor_profile}")
            if actor_profile == "State-Sponsored":
                st.info("Highly sophisticated attacks with stealthy techniques, often backed by nation-state resources.")
            elif actor_profile == "Organized Cybercrime":
                st.info("Profit-driven, large-scale attacks, often noisy and indiscriminate.")
            elif actor_profile == "Hacktivist":
                st.info("Motivated by political or social causes; often opportunistic.")
        elif is_malicious:
            st.warning("Unable to determine actor profile.")
        else:
            st.info("Threat attribution is not applicable for benign URLs.")