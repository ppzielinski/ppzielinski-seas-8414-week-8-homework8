import json, re
import streamlit as st
import google.generativeai as genai
import openai

# Configure keys (already fine)
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    if "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

def get_base_prompt(alert_details):
    return (
        "You are an expert SOAR system. "
        "A URL has been flagged as a phishing attack with these characteristics:\n"
        f"{json.dumps(alert_details, indent=2)}\n\n"
        "Return ONLY a JSON object with EXACTLY these keys:\n"
        '  \"summary\" (string),\n'
        '  \"risk_level\" (\"Critical\"|\"High\"|\"Medium\"|\"Low\"),\n'
        '  \"recommended_actions\" (array of strings),\n'
        '  \"communication_draft\" (string)\n'
        "No markdown, no code fences, no commentary."
    )

def _coerce_json(text: str):
    s = (text or "").strip()
    # strip code fences if any
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I|re.S).strip()
    s = re.sub(r"^'''(?:json)?\s*|\s*'''$", "", s, flags=re.I|re.S).strip()
    # keep only the outermost {...}
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1:
        s = s[a:b+1]
    return json.loads(s)

def get_gemini_prescription(alert_details):
    if "GEMINI_API_KEY" not in st.secrets or not st.secrets["GEMINI_API_KEY"]:
        raise RuntimeError("Gemini key not configured.")

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = get_base_prompt(alert_details)
    resp = model.generate_content(prompt)

    # Debug/diagnostics: if blocked or empty, raise a readable error
    # (You will see this in `make logs` output)
    if hasattr(resp, "prompt_feedback") and resp.prompt_feedback:
        fb = resp.prompt_feedback
        if getattr(fb, "block_reason", None):
            raise RuntimeError(f"Gemini blocked the prompt: {fb.block_reason}")

    text = getattr(resp, "text", "") or ""
    if not text.strip():
        # Sometimes text is empty but candidates exist; try to extract manually
        try:
            cand = resp.candidates[0]
            parts = getattr(cand, "content", {}).parts if hasattr(cand, "content") else []
            text = "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            pass

    if not text.strip():
        raise RuntimeError("Gemini returned empty response text (possible safety block or quota issue).")

    return _coerce_json(text)

def get_openai_prescription(alert_details):
    if "OPENAI_API_KEY" not in st.secrets or not st.secrets["OPENAI_API_KEY"]:
        raise RuntimeError("OpenAI key not configured.")
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = get_base_prompt(alert_details)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return _coerce_json(resp.choices[0].message.content)

def generate_prescription(provider, alert_details):
    if provider == "Gemini":
        return get_gemini_prescription(alert_details)
    elif provider == "OpenAI":
        return get_openai_prescription(alert_details)
    else:
        raise ValueError("Invalid provider selected")