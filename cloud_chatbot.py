# cloud_chatbot.py - FREE Hugging Face Chatbot for Streamlit Cloud
import streamlit as st
import requests
import os
import re

# Page Configuration
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="❤️",
    layout="wide"
)

# Emergency Banner
st.markdown("""
<div style="background-color:#ff1744; color:white; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; margin-bottom:1rem;">
    🚨 CALL EMERGENCY SERVICES (911/999) IF THIS IS A MEDICAL EMERGENCY
</div>
""", unsafe_allow_html=True)

st.title("❤️ Healthcare Assistant")
st.caption("Powered by Hugging Face - 100% Free")

# Get API Key from Streamlit Secrets
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    HF_API_KEY = os.getenv("HF_API_KEY")
    if not HF_API_KEY:
        st.warning("⚠️ Please add HF_API_KEY to Streamlit Secrets")
        st.info("""
        1. Get a free token from: https://huggingface.co/settings/tokens
        2. In your Streamlit app, go to Settings → Secrets
        3. Add: HF_API_KEY = "your-token-here"
        4. Redeploy
        """)

# Hugging Face API Function
def query_huggingface(prompt):
    if not HF_API_KEY:
        return "Please add your Hugging Face API key to use this feature."
    
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": f"Question: {prompt}\nAnswer:",
        "parameters": {
            "max_length": 150,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '')
                if 'Answer:' in answer:
                    answer = answer.split('Answer:')[-1].strip()
                if prompt in answer:
                    answer = answer.replace(prompt, '').strip()
                return answer if len(answer) > 5 else "I'm not sure. Please try rephrasing."
            return "No response generated. Please try again."
        elif response.status_code == 503:
            return "The model is loading. Please wait 10 seconds and try again."
        else:
            return f"Error: {response.status_code}. Please try again later."
            
    except requests.exceptions.Timeout:
        return "Request timed out. The model may be busy. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

# BMI Calculator
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    bmi = round(weight / (height_m * height_m), 1)
    
    if bmi < 18.5:
        category = "Underweight"
        advice = "Consider consulting a nutritionist for healthy weight gain."
    elif bmi < 25:
        category = "Normal weight"
        advice = "Great! You're in a healthy weight range. Keep it up!"
    elif bmi < 30:
        category = "Overweight"
        advice = "Consider a balanced diet and regular exercise."
    else:
        category = "Obese"
        advice = "Please consult a healthcare professional for personalized advice."
    
    return f"""
📊 **BMI Result:** {bmi} - {category}

{advice}

⚠️ BMI is a screening tool, not a diagnostic test.
"""

# Process Query
def process_query(prompt):
    if "bmi" in prompt.lower() or "calculate my bmi" in prompt.lower():
        weight_match = re.search(r'(\d+\.?\d*)\s*(?:kg|kgs?)', prompt, re.I)
        height_match = re.search(r'(\d+\.?\d*)\s*(?:cm|centimeters?)', prompt, re.I)
        
        if weight_match and height_match:
            weight = float(weight_match.group(1))
            height = float(height_match.group(1))
            return calculate_bmi(weight, height)
        else:
            return "📊 Please provide weight and height.\n\nExample: *'Calculate BMI 70kg 175cm'*"
    
    return query_huggingface(prompt)

# Sidebar
with st.sidebar:
    st.header("💡 Quick Actions")
    examples = [
        "Calculate my BMI with weight 70kg height 175cm",
        "What's the nutrition in chicken breast?",
        "Give me exercises for weight loss",
        "I have a headache",
        "What's my diabetes risk?"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.example_query = ex
            st.rerun()
    
    st.divider()
    
    if HF_API_KEY:
        st.success("✅ Hugging Face Connected")
    else:
        st.error("❌ No API Key Found")
        st.info("Add HF_API_KEY to Secrets")
    
    st.divider()
    st.caption("⚡ Free tier may have rate limits")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """
👋 Hello! I'm your Healthcare Assistant, available 24/7!

**I can help with:**
- 📊 BMI calculation
- 🍎 Nutrition advice
- 🏋️ Exercise recommendations
- 🩺 Symptom analysis
- 📈 Diabetes risk assessment

⚠️ I'm an AI, not a doctor. Always consult healthcare professionals.

**Try asking:**
- "Calculate my BMI with weight 70kg height 175cm"
- "What's the nutrition in chicken breast?"
- "I have a headache"
"""}
    ]

# Chat Interface
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
