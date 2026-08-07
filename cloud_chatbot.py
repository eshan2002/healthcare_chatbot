# cloud_chatbot.py
import streamlit as st
import requests
import re
import os

st.set_page_config(page_title="Healthcare Assistant", page_icon="❤️", layout="wide")

st.markdown("""
<div style="background-color:#ff1744; color:white; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; margin-bottom:1rem;">
    🚨 CALL EMERGENCY SERVICES (911/999) IF THIS IS A MEDICAL EMERGENCY
</div>
""", unsafe_allow_html=True)

st.title("❤️ Healthcare Assistant")
st.caption("Powered by Hugging Face - 100% FREE")

try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    HF_API_KEY = os.getenv("HF_API_KEY")
    if not HF_API_KEY:
        st.warning("⚠️ Add HF_API_KEY to Streamlit Secrets")
        st.info("Get one free at https://huggingface.co/settings/tokens")

def ask_ai(question):
    if not HF_API_KEY:
        return "Please add your Hugging Face API key."
    
    url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": f"Question: {question}\nAnswer:", "parameters": {"max_length": 150}}
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        if r.status_code == 200:
            result = r.json()
            if result and isinstance(result, list):
                text = result[0].get('generated_text', '')
                if 'Answer:' in text:
                    text = text.split('Answer:')[-1].strip()
                return text if len(text) > 3 else "I'm not sure. Please try again."
        return "Error. Please try again."
    except:
        return "Service unavailable. Please try later."

def calc_bmi(w, h_cm):
    h_m = h_cm / 100
    bmi = round(w / (h_m * h_m), 1)
    if bmi < 18.5:
        cat, adv = "Underweight", "Consider consulting a nutritionist."
    elif bmi < 25:
        cat, adv = "Normal weight", "Great! Healthy range."
    elif bmi < 30:
        cat, adv = "Overweight", "Consider balanced diet and exercise."
    else:
        cat, adv = "Obese", "Please consult a doctor."
    return f"📊 BMI: {bmi} - {cat}\n\n{adv}\n\n⚠️ BMI is a screening tool only."

with st.sidebar:
    st.header("Quick Examples")
    for q in ["Calculate my BMI with weight 70kg height 175cm", "What's the nutrition in chicken breast?", "I have a headache"]:
        if st.button(q, use_container_width=True):
            st.session_state.example = q
            st.rerun()
    st.divider()
    st.success("✅ Hugging Face API Ready" if HF_API_KEY else "❌ No API Key")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hello! Ask me anything about health. ⚠️ I'm an AI, not a doctor."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if "example" in st.session_state:
    p = st.session_state.example
    del st.session_state.example
    st.session_state.messages.append({"role": "user", "content": p})
    with st.chat_message("user"):
        st.markdown(p)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if "bmi" in p.lower():
                wm = re.search(r'(\d+)\s*kg', p, re.I)
                hm = re.search(r'(\d+)\s*cm', p, re.I)
                if wm and hm:
                    r = calc_bmi(float(wm.group(1)), float(hm.group(1)))
                else:
                    r = "Please provide weight in kg and height in cm."
            else:
                r = ask_ai(p)
            st.markdown(r)
            st.session_state.messages.append({"role": "assistant", "content": r})

if p := st.chat_input("Ask about your health..."):
    st.session_state.messages.append({"role": "user", "content": p})
    with st.chat_message("user"):
        st.markdown(p)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if "bmi" in p.lower():
                wm = re.search(r'(\d+)\s*kg', p, re.I)
                hm = re.search(r'(\d+)\s*cm', p, re.I)
                if wm and hm:
                    r = calc_bmi(float(wm.group(1)), float(hm.group(1)))
                else:
                    r = "Please provide weight in kg and height in cm."
            else:
                r = ask_ai(p)
            st.markdown(r)
            st.session_state.messages.append({"role": "assistant", "content": r})