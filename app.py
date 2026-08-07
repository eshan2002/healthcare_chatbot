# chatbot.py - UPDATED for latest LangChain
import os
import pandas as pd
import streamlit as st
from typing import List

# ============================================
# NEW IMPORT STRUCTURE (LangChain v0.1+)
# ============================================

# These are the correct imports for newer LangChain versions
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# ============================================
# Get API Key (works locally and on cloud)
# ============================================

try:
    # Running on Streamlit Cloud
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Running locally
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.error("⚠️ OPENAI_API_KEY not found! Please add it to .env file or Streamlit secrets.")
        st.stop()

# ============================================
# Load Data
# ============================================

@st.cache_resource
def load_data():
    """Load all CSV files and create documents"""
    documents = []
    
    try:
        # Load nutrition data
        nutrition_df = pd.read_csv('nutrition.csv')
        for _, row in nutrition_df.iterrows():
            doc = Document(
                page_content=f"""
                Food: {row['food_item'].replace('_', ' ')}
                Calories: {row['calories']} kcal
                Protein: {row['protein_g']}g
                Carbohydrates: {row['carbs_g']}g
                Fiber: {row['fiber_g']}g
                Sugar: {row['sugar_g']}g
                Fat: {row['fat_g']}g
                Recommended for: {row['recommended_for']}
                Avoid if: {row['avoid_if'] if pd.notna(row['avoid_if']) else 'None'}
                """,
                metadata={"type": "nutrition", "food": row['food_item']}
            )
            documents.append(doc)
    except FileNotFoundError:
        st.warning("nutrition.csv not found")
    
    try:
        # Load exercise data
        exercise_df = pd.read_csv('exercises.csv')
        for _, row in exercise_df.iterrows():
            doc = Document(
                page_content=f"""
                Exercise: {row['exercise_name'].replace('_', ' ')}
                Duration: {row['duration_min']} minutes
                Calories burned: {row['calories_burned']} kcal
                Difficulty: {row['difficulty']}
                Best for: {row['best_for']}
                Equipment needed: {row['equipment']}
                """,
                metadata={"type": "exercise", "exercise": row['exercise_name']}
            )
            documents.append(doc)
    except FileNotFoundError:
        st.warning("exercises.csv not found")
    
    try:
        # Load symptoms data
        symptoms_df = pd.read_csv('symptoms.csv')
        for _, row in symptoms_df.iterrows():
            doc = Document(
                page_content=f"""
                Symptom: {row['symptom'].replace('_', ' ')}
                Condition: {row['condition']}
                Severity: {row['severity']}
                Recommendation: {row['recommendation']}
                Should consult doctor: {row['should_consult_doctor']}
                """,
                metadata={"type": "symptom", "symptom": row['symptom']}
            )
            documents.append(doc)
    except FileNotFoundError:
        st.warning("symptoms.csv not found")
    
    try:
        # Load diabetes data
        diabetes_df = pd.read_csv('diabetes.csv')
        stats = f"""
        Diabetes Dataset Analysis:
        Total patients: {len(diabetes_df)}
        Patients with diabetes: {len(diabetes_df[diabetes_df['Outcome'] == 1])}
        
        Key risk factors:
        - Average Glucose level: {diabetes_df['Glucose'].mean():.1f}
        - Average BMI: {diabetes_df['BMI'].mean():.1f}
        - Average Age: {diabetes_df['Age'].mean():.1f}
        """
        doc = Document(
            page_content=stats,
            metadata={"type": "diabetes_analysis"}
        )
        documents.append(doc)
    except FileNotFoundError:
        st.warning("diabetes.csv not found")
    
    return documents

# ============================================
# Create Chatbot
# ============================================

@st.cache_resource
def create_chatbot():
    """Initialize the chatbot"""
    
    # Load data
    documents = load_data()
    
    if not documents:
        st.error("No data loaded. Please ensure CSV files are present.")
        return None
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vector_store.persist()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=openai_api_key
    )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )
    
    return chain

# ============================================
# Main App
# ============================================

def main():
    st.set_page_config(
        page_title="AI Healthcare Assistant",
        page_icon="❤️",
        layout="wide"
    )
    
    # Emergency banner
    st.markdown("""
    <div style="background-color:#ff1744; color:white; padding:1rem; border-radius:0.5rem; text-align:center; font-weight:bold; margin-bottom:1rem;">
        🚨 CALL EMERGENCY SERVICES (911/999) IF THIS IS A MEDICAL EMERGENCY
    </div>
    """, unsafe_allow_html=True)
    
    st.title("❤️ AI Healthcare Assistant")
    st.caption("Your intelligent health companion - Ask me anything about health, nutrition, fitness, and symptoms")
    
    # Initialize chatbot
    if "chain" not in st.session_state:
        with st.spinner("Loading healthcare knowledge..."):
            chain = create_chatbot()
            if chain:
                st.session_state.chain = chain
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """
            👋 Hello! I'm your AI Healthcare Assistant.
            
            I can help you with:
            - 🩺 Health assessments
            - 📊 BMI calculation
            - 🍎 Nutrition advice  
            - 🏋️ Exercise recommendations
            - 🔬 Symptom analysis
            - 📈 Diabetes risk assessment
            
            ⚠️ **Important**: I'm an AI assistant, not a doctor. Always consult healthcare professionals for medical decisions.
            
            How can I help you today?
            """}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input area
    if prompt := st.chat_input("Type your health question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query..."):
                try:
                    if "chain" in st.session_state:
                        result = st.session_state.chain({"question": prompt})
                        response = result["answer"]
                    else:
                        response = "Chatbot not initialized. Please check your API key and CSV files."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("💡 Make sure your OpenAI API key is set correctly")

if __name__ == "__main__":
    main()