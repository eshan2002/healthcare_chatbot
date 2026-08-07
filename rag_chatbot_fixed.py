# rag_chatbot_fixed.py - CORRECTED IMPORTS
import streamlit as st
import pandas as pd
import re
import os

# ============================================
# CORRECT IMPORTS FOR NEWER LANGCHAIN
# ============================================

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ============================================
# App Configuration
# ============================================

st.set_page_config(
    page_title="RAG Healthcare Assistant",
    page_icon="📊",
    layout="wide"
)

# Emergency banner
st.markdown("""
<div style="background-color:#ff1744; color:white; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; margin-bottom:1rem;">
    🚨 CALL EMERGENCY SERVICES (911/999) IF THIS IS A MEDICAL EMERGENCY
</div>
""", unsafe_allow_html=True)

st.title("📊 RAG Healthcare Assistant")
st.caption("Powered by Ollama + Your CSV Data - 100% Free and Private")

# ============================================
# Load CSV Data
# ============================================

@st.cache_data
def load_csv_data():
    """Load all CSV files"""
    data = {}
    csv_files = {
        'nutrition': 'nutrition.csv',
        'exercises': 'exercises.csv', 
        'symptoms': 'symptoms.csv',
        'diabetes': 'diabetes.csv'
    }
    
    for name, file in csv_files.items():
        try:
            data[name] = pd.read_csv(file)
            st.sidebar.success(f"✅ Loaded {name}.csv ({len(data[name])} records)")
        except FileNotFoundError:
            data[name] = None
            st.sidebar.error(f"❌ {file} not found")
    
    return data

# ============================================
# Create Documents from CSV Data
# ============================================

def create_documents_from_csv(data):
    """Convert CSV data to LangChain Documents"""
    documents = []
    
    # 1. Nutrition Data
    if data.get('nutrition') is not None:
        for _, row in data['nutrition'].iterrows():
            content = f"""
FOOD: {row['food_item'].replace('_', ' ')}
CALORIES: {row['calories']} kcal
PROTEIN: {row['protein_g']}g
CARBS: {row['carbs_g']}g
FIBER: {row['fiber_g']}g
SUGAR: {row['sugar_g']}g
FAT: {row['fat_g']}g
BEST FOR: {row['recommended_for']}
AVOID IF: {row['avoid_if'] if pd.notna(row['avoid_if']) else 'None'}
"""
            documents.append(Document(
                page_content=content,
                metadata={"type": "nutrition", "food": row['food_item']}
            ))
    
    # 2. Exercise Data
    if data.get('exercises') is not None:
        for _, row in data['exercises'].iterrows():
            content = f"""
EXERCISE: {row['exercise_name'].replace('_', ' ')}
DURATION: {row['duration_min']} minutes
CALORIES BURNED: {row['calories_burned']} kcal
DIFFICULTY: {row['difficulty']}
BEST FOR: {row['best_for']}
EQUIPMENT: {row['equipment']}
"""
            documents.append(Document(
                page_content=content,
                metadata={"type": "exercise", "exercise": row['exercise_name']}
            ))
    
    # 3. Symptoms Data
    if data.get('symptoms') is not None:
        for _, row in data['symptoms'].iterrows():
            content = f"""
SYMPTOM: {row['symptom'].replace('_', ' ')}
CONDITION: {row['condition']}
SEVERITY: {row['severity']}
RECOMMENDATION: {row['recommendation']}
CONSULT DOCTOR: {row['should_consult_doctor']}
"""
            documents.append(Document(
                page_content=content,
                metadata={"type": "symptom", "symptom": row['symptom']}
            ))
    
    # 4. Diabetes Data
    if data.get('diabetes') is not None:
        df = data['diabetes']
        content = f"""
DIABETES DATASET ANALYSIS:
Total patients: {len(df)}
Patients with diabetes: {len(df[df['Outcome'] == 1])}
Diabetes rate: {(len(df[df['Outcome'] == 1])/len(df)*100):.1f}%

KEY RISK FACTORS:
Average Glucose: {df['Glucose'].mean():.1f} mg/dL
Average BMI: {df['BMI'].mean():.1f}
Average Age: {df['Age'].mean():.1f} years
Average Pregnancies: {df['Pregnancies'].mean():.1f}

RISK THRESHOLDS:
- Glucose > 140 mg/dL: High risk
- BMI > 30: High risk
- Age > 40: Moderate risk
"""
        documents.append(Document(
            page_content=content,
            metadata={"type": "diabetes"}
        ))
    
    return documents

# ============================================
# Setup Vector Store
# ============================================

@st.cache_resource
def setup_vector_store():
    """Create vector store from CSV data"""
    
    with st.spinner("Loading your healthcare data into the knowledge base..."):
        # Load data
        data = load_csv_data()
        documents = create_documents_from_csv(data)
        
        if not documents:
            st.error("❌ No documents created! Check your CSV files.")
            return None, None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return None, None
        
        # Create vector store
        persist_directory = "./chroma_db"
        
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        except:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vectorstore.persist()
        
        return vectorstore, data

# ============================================
# Create RAG Chain
# ============================================

@st.cache_resource
def create_rag_chain(vectorstore):
    """Create RAG chain with Ollama"""
    
    # Check if Ollama is running
    try:
        llm = Ollama(model="phi3:mini", temperature=0.3)
        llm.invoke("Hello")
    except Exception as e:
        st.warning(f"⚠️ Ollama not available: {e}")
        st.info("""
        **Start Ollama:**
        1. Open a new Command Prompt
        2. Run: `ollama serve`
        3. Keep it running
        4. Refresh this page
        """)
        return None
    
    # Custom prompt template using the CSV data
    prompt_template = """
    You are a helpful healthcare assistant with access to specific health data.
    Use the following information from the user's health database to answer the question.
    If the information doesn't contain the answer, say so politely.
    
    HEALTH DATABASE INFORMATION:
    {context}
    
    USER QUESTION: {question}
    
    HELPFUL RESPONSE (based on the health database):
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

# ============================================
# BMI Calculator
# ============================================

def calculate_bmi(weight: float, height_cm: float) -> str:
    """Calculate BMI and return result"""
    height_m = height_cm / 100
    bmi = weight / (height_m * height_m)
    bmi = round(bmi, 1)
    
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

# ============================================
# Fallback Response
# ============================================

def get_fallback_response(query: str, vectorstore) -> str:
    """Response when Ollama is not available"""
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "I don't have information about that in your health database. Please consult a doctor."
    
    response = "📚 **From your health database:**\n\n"
    for i, doc in enumerate(docs, 1):
        response += f"**Source {i}:**\n{doc.page_content.strip()}\n\n"
    
    response += "⚠️ I'm an AI, not a doctor. Please consult a healthcare professional."
    return response

# ============================================
# Process Query Function
# ============================================

def process_query(prompt):
    """Process user query with RAG"""
    lower_prompt = prompt.lower()
    
    # BMI detection
    if "bmi" in lower_prompt or "calculate my bmi" in lower_prompt:
        weight_match = re.search(r'(\d+\.?\d*)\s*(?:kg|kgs?)', prompt, re.I)
        height_match = re.search(r'(\d+\.?\d*)\s*(?:cm|centimeters?)', prompt, re.I)
        
        if weight_match and height_match:
            weight = float(weight_match.group(1))
            height = float(height_match.group(1))
            return calculate_bmi(weight, height)
        else:
            return "📊 Please provide weight and height.\n\nExample: *'Calculate BMI 70kg 175cm'*"
    
    # Use RAG for everything else
    if "rag_chain" in st.session_state and st.session_state.rag_chain:
        try:
            result = st.session_state.rag_chain.run(prompt)
            return result
        except Exception as e:
            return f"Error: {str(e)}\n\nUsing fallback..."
    
    # Fallback
    if "vectorstore" in st.session_state:
        return get_fallback_response(prompt, st.session_state.vectorstore)
    
    return "Please wait for the system to initialize..."

# ============================================
# Main App
# ============================================

# Initialize vector store and RAG chain
if "vectorstore" not in st.session_state:
    vectorstore, data = setup_vector_store()
    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.session_state.data = data

if "rag_chain" not in st.session_state:
    if "vectorstore" in st.session_state:
        chain = create_rag_chain(st.session_state.vectorstore)
        if chain:
            st.session_state.rag_chain = chain

# Sidebar
with st.sidebar:
    st.header("📊 Your Data")
    
    if "data" in st.session_state:
        data = st.session_state.data
        for name, df in data.items():
            if df is not None:
                st.success(f"✅ {name}: {len(df)} records")
            else:
                st.error(f"❌ {name}: missing")
    
    st.divider()
    st.header("💡 Example Questions")
    examples = [
        "What's the nutrition in chicken breast?",
        "What exercises are good for weight loss?",
        "I have a headache, what should I do?",
        "What's my diabetes risk?",
        "Calculate my BMI with weight 70kg height 175cm"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.example_query = ex
            st.rerun()
    
    st.divider()
    st.header("ℹ️ About")
    st.info("""
    This chatbot uses:
    - 🦙 Ollama (local LLM)
    - 📊 Your CSV data (RAG)
    - 🔒 100% private
    - 💰 100% free
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """
👋 Hello! I'm your **RAG Healthcare Assistant**!

I can answer questions using **YOUR healthcare data** from:
- 🍎 **nutrition.csv** (food nutrition facts)
- 🏋️ **exercises.csv** (exercise recommendations)
- 🩺 **symptoms.csv** (symptom analysis)
- 📈 **diabetes.csv** (diabetes risk assessment)

**Try asking:**
- "What's the nutrition in chicken breast?"
- "What exercises are good for weight loss?"
- "I have a headache, what should I do?"
- "What's my diabetes risk?"
- "Calculate my BMI with weight 70kg height 175cm"

⚠️ I'm an AI, not a doctor. Always consult healthcare professionals.
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
        with st.spinner("Searching your health database..."):
            response = process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your health data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching your health database..."):
            response = process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})