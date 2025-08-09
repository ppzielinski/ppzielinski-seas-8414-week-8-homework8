import google.generativeai as genai
import openai
import streamlit as st
import json

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    pass # Fail silently, the app will handle it

def get_base_prompt(alert_details):
    return f"""You are an expert SOAR system. A URL has been flagged as a phishing attack with these characteristics: {json.dumps(alert_details, indent=2)}. Generate a prescriptive incident response plan. Provide a JSON response with keys: "summary", "risk_level", "recommended_actions": a list of STRINGS, and "communication_draft". Return ONLY the raw JSON object."""

def get_gemini_prescription(alert_details):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = get_base_prompt(alert_details)
    response_text = model.generate_content(prompt).text.strip().lstrip("'''json\n").rstrip("'''")
    return json.loads(response_text)

def get_openai_prescription(alert_details):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = get_base_prompt(alert_details)
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    return json.loads(response.choices[0].message.content)

def generate_prescription(provider, alert_details):
    if provider == "Gemini":
        return get_gemini_prescription(alert_details)
    elif provider == "OpenAI":
        return get_openai_prescription(alert_details)
    else:
        raise ValueError("Invalid provider selected")