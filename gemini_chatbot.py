# gemini_chatbot.py - Healthcare Assistant with Google Gemini
import streamlit as st
import google.generativeai as genai
import re
import os

# Page setup
st.set_page_config(page_title="Healthcare Assistant", page_icon="❤️", layout="wide")

# Emergency banner
st.markdown("""
<div style="background-color:#ff1744; color:white; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; margin-bottom:1rem;">
    🚨 CALL EMERGENCY SERVICES (911/999) IF THIS IS A MEDICAL EMERGENCY
</div>
""", unsafe_allow_html=True)

st.title("❤️ Healthcare Assistant")
st.caption("Powered by Google Gemini - 100% FREE")

# Get API key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash')  # Fast, free model[citation:6]
except:
    st.warning("⚠️ Please add GEMINI_API_KEY to Streamlit Secrets")
    st.info("Get a free key from: https://ai.dev")
    st.stop()

# Functions
def ask_gemini(question):
    """Query Google Gemini API"""
    try:
        response = model.generate_content(f"Question: {question}\nAnswer:")
        return response.text if response.text else "I'm not sure. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    bmi = round(weight / (height_m * height_m), 1)
    if bmi < 18.5:
        cat, advice = "Underweight", "Consider consulting a nutritionist."
    elif bmi < 25:
        cat, advice = "Normal weight", "Great! You're in a healthy range."
    elif bmi < 30:
        cat, advice = "Overweight", "Consider balanced diet and exercise."
    else:
        cat, advice = "Obese", "Please consult a healthcare professional."
    return f"📊 **BMI Result:** {bmi} - {cat}\n\n{advice}"

def process_query(prompt):
    if "bmi" in prompt.lower():
        wm = re.search(r'(\d+\.?\d*)\s*(?:kg|kgs?)', prompt, re.I)
        hm = re.search(r'(\d+\.?\d*)\s*(?:cm|centimeters?)', prompt, re.I)
        if wm and hm:
            return calculate_bmi(float(wm.group(1)), float(hm.group(1)))
        return "📊 Please provide weight and height. Example: 'Calculate BMI 70kg 175cm'"
    return ask_gemini(prompt)

# Sidebar
with st.sidebar:
    st.header("💡 Quick Actions")
    for ex in ["Calculate my BMI with weight 70kg height 175cm", "What's the nutrition in chicken breast?", "I have a headache"]:
        if st.button(ex, use_container_width=True):
            st.session_state.example = ex
            st.rerun()
    st.divider()
    st.success("✅ Gemini Connected")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hello! I'm your Healthcare Assistant. ⚠️ I'm an AI, not a doctor."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Example handler
if "example" in st.session_state:
    prompt = st.session_state.example
    del st.session_state.example
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input
if prompt := st.chat_input("Ask about your health..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})