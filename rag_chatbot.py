# rag_chatbot.py - RAG Healthcare Chatbot with Ollama
import os
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import re
import time

# ============================================
# LANGCHAIN IMPORTS FOR OLLAMA
# ============================================

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError as e:
    st.error(f"Missing package: {e}")
    st.info("Install with: pip install langchain chromadb sentence-transformers")
    st.stop()

# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_csv_data():
    """Load all CSV files"""
    data = {}
    csv_files = ['nutrition.csv', 'exercises.csv', 'symptoms.csv', 'diabetes.csv']
    
    for file in csv_files:
        try:
            data[file.replace('.csv', '')] = pd.read_csv(file)
        except FileNotFoundError:
            data[file.replace('.csv', '')] = None
    
    return data

# ============================================
# DOCUMENT CREATION
# ============================================

def create_documents(data: Dict) -> List[Document]:
    """Convert CSV data to LangChain Documents"""
    documents = []
    
    # Nutrition Data
    if data.get('nutrition') is not None:
        for _, row in data['nutrition'].iterrows():
            content = f"""
FOOD: {row['food_item'].replace('_', ' ')}
Calories: {row['calories']} kcal
Protein: {row['protein_g']}g
Carbs: {row['carbs_g']}g
Fiber: {row['fiber_g']}g
Sugar: {row['sugar_g']}g
Fat: {row['fat_g']}g
Best for: {row['recommended_for']}
Avoid if: {row['avoid_if'] if pd.notna(row['avoid_if']) else 'None'}
            """
            documents.append(Document(
                page_content=content,
                metadata={"type": "nutrition", "food": row['food_item']}
            ))
    
    # Exercise Data
    if data.get('exercises') is not None:
        for _, row in data['exercises'].iterrows():
            content = f"""
EXERCISE: {row['exercise_name'].replace('_', ' ')}
Duration: {row['duration_min']} minutes
Calories burned: {row['calories_burned']} kcal
Difficulty: {row['difficulty']}
Best for: {row['best_for']}
Equipment: {row['equipment']}
            """
            documents.append(Document(
                page_content=content,
                metadata={"type": "exercise", "exercise": row['exercise_name']}
            ))
    
    # Symptoms Data
    if data.get('symptoms') is not None:
        for _, row in data['symptoms'].iterrows():
            content = f"""
SYMPTOM: {row['symptom'].replace('_', ' ')}
Condition: {row['condition']}
Severity: {row['severity']}
Recommendation: {row['recommendation']}
Consult doctor: {row['should_consult_doctor']}
            """
            documents.append(Document(
                page_content=content,
                metadata={"type": "symptom", "symptom": row['symptom']}
            ))
    
    # Diabetes Data
    if data.get('diabetes') is not None:
        df = data['diabetes']
        content = f"""
DIABETES ANALYSIS:
Total patients: {len(df)}
Diabetic: {len(df[df['Outcome'] == 1])}
Rate: {(len(df[df['Outcome'] == 1])/len(df)*100):.1f}%
Avg Glucose: {df['Glucose'].mean():.1f}
Avg BMI: {df['BMI'].mean():.1f}
Avg Age: {df['Age'].mean():.1f}
        """
        documents.append(Document(
            page_content=content,
            metadata={"type": "diabetes"}
        ))
    
    return documents

# ============================================
# VECTOR STORE SETUP
# ============================================

@st.cache_resource
def setup_vector_store():
    """Create or load vector store"""
    
    with st.spinner("Loading your healthcare data..."):
        # Load data
        data = load_csv_data()
        documents = create_documents(data)
        
        if not documents:
            st.error("❌ No documents created! Check your CSV files.")
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings (free, local)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return None
        
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
        
        return vectorstore

# ============================================
# RAG CHAIN CREATION WITH OLLAMA
# ============================================

@st.cache_resource
def create_rag_chain(vectorstore):
    """Create RAG chain with Ollama"""
    
    # Check if Ollama is running
    try:
        llm = Ollama(model="phi3:mini", temperature=0.3)
        # Test connection
        llm.invoke("Hello")
    except Exception as e:
        st.warning(f"⚠️ Ollama not available: {str(e)}")
        st.info("""
        **To use Ollama:**
        1. Make sure Ollama is installed
        2. Open Command Prompt and run: `ollama pull phi3:mini`
        3. Keep Ollama running in the background
        4. Refresh this page
        """)
        return None
    
    # Healthcare prompt template
    prompt_template = """
    You are a helpful healthcare assistant. Use the following context to answer the user's question.
    If you don't know the answer, say so and suggest consulting a doctor.
    Be empathetic and provide clear, helpful information.
    
    CONTEXT:
    {context}
    
    USER QUESTION: {question}
    
    HELPFUL RESPONSE:
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
# FALLBACK RESPONSE (Without LLM)
# ============================================

def get_fallback_response(query: str, vectorstore) -> str:
    """Response when Ollama is not available"""
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "I don't have information about that. Please consult a doctor."
    
    response = "📚 **From your health data:**\n\n"
    for i, doc in enumerate(docs, 1):
        response += f"**Source {i}:**\n{doc.page_content.strip()}\n\n"
    
    response += "⚠️ I'm an AI, not a doctor. Please consult a healthcare professional."
    return response

# ============================================
# BMI CALCULATOR
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

⚠️ BMI is a screening tool, not a diagnostic test. It doesn't account for muscle mass, bone density, or overall body composition.
"""

# ============================================
# PROCESS QUERY
# ============================================

def process_query(prompt, vectorstore=None):
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
    
    # Fallback if no RAG chain
    if vectorstore:
        return get_fallback_response(prompt, vectorstore)
    
    return "Please wait for the system to initialize..."

# ============================================
# MAIN STREAMLIT APP
# ============================================

def main():
    st.set_page_config(
        page_title="RAG Healthcare Assistant",
        page_icon="❤️",
        layout="wide"
    )
    
    # Emergency banner
    st.markdown("""
    <div style="background-color:#ff1744; color:white; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; margin-bottom:1rem;">
        🚨 CALL EMERGENCY SERVICES (911/999) IF THIS IS A MEDICAL EMERGENCY
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🦙 RAG Healthcare Assistant")
        st.caption("Powered by Ollama + RAG - 100% Free and Private")
    with col2:
        st.image("https://ollama.ai/public/ollama.png", width=80)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check Ollama
        try:
            from langchain_community.llms import Ollama
            llm = Ollama(model="phi3:mini")
            llm.invoke("test")
            st.success("✅ Ollama connected")
            st.info(f"Model: phi3:mini")
        except:
            st.error("❌ Ollama not running")
            st.info("""
            **Start Ollama:**
            1. Open Command Prompt
            2. Run: `ollama pull phi3:mini`
            3. Keep it running
            4. Refresh this page
            """)
        
        st.divider()
        st.header("📊 Your Data")
        data = load_csv_data()
        for name, df in data.items():
            if df is not None:
                st.success(f"✅ {name}: {len(df)} records")
            else:
                st.error(f"❌ {name}: missing")
        
        st.divider()
        st.header("💡 Quick Examples")
        examples = [
            "Nutrition for chicken",
            "Exercises for weight loss",
            "I have a headache",
            "Diabetes risk assessment",
            "Calculate BMI 70kg 175cm"
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
        - 🧠 RAG (Retrieval-Augmented Generation)
        - 📊 Your healthcare data
        - 🔒 100% private (runs locally)
        """)
    
    # Initialize vector store
    if "vectorstore" not in st.session_state:
        vectorstore = setup_vector_store()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
    
    # Initialize RAG chain
    if "rag_chain" not in st.session_state:
        if "vectorstore" in st.session_state:
            chain = create_rag_chain(st.session_state.vectorstore)
            if chain:
                st.session_state.rag_chain = chain
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """
👋 Hello! I'm your RAG Healthcare Assistant powered by Ollama.

**I can help with:**
- 🍎 Nutrition information
- 🏋️ Exercise recommendations
- 🩺 Symptom analysis
- 📈 Diabetes risk assessment
- 📊 BMI calculation

**Try asking:**
- "Nutrition for chicken"
- "Exercises for weight loss"
- "I have back pain"
- "What's my diabetes risk?"

⚠️ I'm an AI, not a doctor. Always consult healthcare professionals for medical decisions.
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
                response = process_query(prompt, st.session_state.get("vectorstore"))
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
            with st.spinner("Searching your health data..."):
                response = process_query(prompt, st.session_state.get("vectorstore"))
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()