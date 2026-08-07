# cloud_chatbot.py - Healthcare Assistant with Hugging Face
import streamlit as st
import requests
import re
import os

st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="❤️",
    layout="wide"
)

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
        st.warning("⚠️ Please add HF_API_KEY to Streamlit Secrets")

def ask_huggingface(question):
    if not HF_API_KEY:
        return "Please add your Hugging Face API key."
    
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": f"Question: {question}\nAnswer:",
        "parameters": {"max_length": 150, "temperature": 0.7}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '')
                if 'Answer:' in answer:
                    answer = answer.split('Answer:')[-1].strip()
                return answer if len(answer) > 5 else "I'm not sure. Please try again."
            return "No response generated."
        elif response.status_code == 503:
            return "Model is loading. Please wait 10 seconds."
        else:
            return f"Error: {response.status_code}"
    except:
        return "Service unavailable. Please try later."

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    bmi = round(weight / (height_m * height_m), 1)
    
    if bmi < 18.5:
        category, advice = "Underweight", "Consider consulting a nutritionist."
    elif bmi < 25:
        category, advice = "Normal weight", "Great! You're in a healthy range."
    elif bmi < 30:
        category, advice = "Overweight", "Consider balanced diet and exercise."
    else:
        category, advice = "Obese", "Please consult a healthcare professional."
    
    return f"📊 **BMI Result:** {bmi} - {category}\n\n{advice}\n\n⚠️ BMI is a screening tool only."

def process_query(prompt):
    if "bmi" in prompt.lower():
        weight_match = re.search(r'(\d+\.?\d*)\s*(?:kg|kgs?)', prompt, re.I)
        height_match = re.search(r'(\d+\.?\d*)\s*(?:cm|centimeters?)', prompt, re.I)
        if weight_match and height_match:
            return calculate_bmi(float(weight_match.group(1)), float(height_match.group(1)))
        else:
            return "📊 Please provide weight and height.\n\nExample: *'Calculate BMI 70kg 175cm'*"
    return ask_huggingface(prompt)

with st.sidebar:
    st.header("💡 Quick Actions")
    examples = [
        "Calculate my BMI with weight 70kg height 175cm",
        "What's the nutrition in chicken breast?",
        "Give me exercises for weight loss",
        "I have a headache"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.example_query = ex
            st.rerun()
    
    st.divider()
    st.success("✅ Hugging Face Connected" if HF_API_KEY else "❌ No API Key")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """
👋 Hello! I'm your Healthcare Assistant.

**Try asking:**
- "Calculate my BMI with weight 70kg height 175cm"
- "What's the nutrition in chicken breast?"
- "I have a headache"

⚠️ I'm an AI, not a doctor.
"""}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if prompt := st.chat_input("Ask about your health..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})