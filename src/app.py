import sys
import os
from dotenv import load_dotenv
import streamlit as st

# 1. Load environment variables (OPENAI_API_KEY) before anything else
load_dotenv()

# 2. Add the project root to the Python path so absolute imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
st.set_page_config(
    page_title="CiteMentor 2.0",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup Navigation using modern Streamlit API
pages = {
    "Mentorship Engine": [
        st.Page("pages/1_Mentor.py", title="Mentor Chat", icon="💬"),
        st.Page("pages/4_About.py", title="About CiteMentor", icon="🧭"),
    ],
    "System Observability": [
        st.Page("pages/2_Dashboard.py", title="RAG Dashboard", icon="📊"),
        st.Page("pages/3_Ledger.py", title="Royalty Ledger", icon="💰")
    ]
}

nav = st.navigation(pages)
nav.run()
