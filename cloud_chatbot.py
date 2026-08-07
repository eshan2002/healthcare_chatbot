# cloud_chatbot.py - For Streamlit Cloud with Hugging Face (FREE)
import streamlit as st
import requests
import os

# ============================================
# Get API Key from Streamlit Secrets or .env
# ============================================

try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    # For local testing, use environment variable
    HF_API_KEY = os.getenv("HF_API_KEY")
    if not HF_API_KEY:
        st.error("⚠️ Please add HF_API_KEY to Streamlit secrets or .env file")
        st.info("""
        1. Go to https://huggingface.co/settings/tokens
        2. Create a free token
        3. Add it to Streamlit Cloud Secrets
        """)
        st.stop()

# ============================================
# Hugging Face API Function
# ============================================

def query_huggingface(prompt):
    """Query Hugging Face Inference API"""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
# BMI Calculator
# ============================================

def calculate_bmi(weight, height_cm):
    """Calculate BMI and return result"""
    height_m = height_cm / 100
    bmi = weight / (height_m * height_m)
    bmi = round(bmi, 1)
    
    if bmi < 18.5:
        category = "Underweight"
        advice = "Consider consulting a nutritionist."
    elif bmi < 25:
        category = "Normal weight"
        advice = "Great! You're in a healthy range."
    elif bmi < 30:
        category = "Overweight"
        advice = "Consider balanced diet and exercise."
    else:
        category = "Obese"
        advice = "Please consult a healthcare professional."
    
    return f"""
📊 **BMI Result:** {bmi} - {category}

{advice}

⚠️ BMI is a screening tool, not a diagnostic test.
"""

# ============================================
# Main Streamlit App
# ============================================

st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="❤️",
    layout="wide"
)

# Emergency banner
st.markdown("""
<div style="background-color:#ff1744; color:white; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; margin-bottom:1rem;">
    🚨 CALL EMERGENCY SERVICES (911/999) IF THIS IS A MEDICAL EMERGENCY
</div>
""", unsafe_allow_html=True)

st.title("❤️ Healthcare Assistant")
st.caption("Powered by Hugging Face - 100% Free Cloud")

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
    st.header("📊 About")
    st.info("""
    This chatbot uses:
    - 🤗 Hugging Face (free)
    - 🌐 Hosted in the cloud
    - 📱 Accessible anywhere
    - 💰 100% FREE
    """)

# Initialize chat history
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

# Process example query
if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if it's a BMI query
            import re
            if "bmi" in prompt.lower():
                weight_match = re.search(r'(\d+\.?\d*)\s*(?:kg|kgs?)', prompt, re.I)
                height_match = re.search(r'(\d+\.?\d*)\s*(?:cm|centimeters?)', prompt, re.I)
                if weight_match and height_match:
                    weight = float(weight_match.group(1))
                    height = float(height_match.group(1))
                    response = calculate_bmi(weight, height)
                else:
                    response = "Please provide weight and height. Example: 'Calculate BMI 70kg 175cm'"
            else:
                response = query_huggingface(prompt)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your health..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            import re
            if "bmi" in prompt.lower():
                weight_match = re.search(r'(\d+\.?\d*)\s*(?:kg|kgs?)', prompt, re.I)
                height_match = re.search(r'(\d+\.?\d*)\s*(?:cm|centimeters?)', prompt, re.I)
                if weight_match and height_match:
                    weight = float(weight_match.group(1))
                    height = float(height_match.group(1))
                    response = calculate_bmi(weight, height)
                else:
                    response = "📊 Please provide weight and height.\n\nExample: *'Calculate BMI 70kg 175cm'*"
            else:
                response = query_huggingface(prompt)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})