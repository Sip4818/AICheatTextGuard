import streamlit as st
import requests
import json
import os
# Configuration
API_URL = os.getenv("BACKEND_URL")
# API_URL = 'http://127.0.0.1:8080/predict'
MIN_LEN = 250
MAX_LEN = 5000

st.set_page_config(
    page_title="🛡️ AI Text Guard",
    page_icon="🛡️",
    layout="centered"
)

# Custom Styling for the Probability Display
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #f63366; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ AI Text Guard")
st.info(f"Our detector works best on professional writing, essays, and reports between {MIN_LEN} and {MAX_LEN} characters.")

# Input Section
text_input = st.text_area("Paste text here for analysis:", height=300, placeholder="Start typing or paste your text...")

# Logic for dynamic character counting
# We use .strip() because your backend logic will ignore leading/trailing spaces
clean_char_count = len(text_input.strip())

# Layout for Character Counter and Buttons
col1, col2 = st.columns([1, 1])

with col1:
    if clean_char_count > 0:
        if clean_char_count < MIN_LEN:
            st.caption(f"⚠️ :red[{clean_char_count}] / {MAX_LEN} (Needs {MIN_LEN - clean_char_count} more)")
        elif clean_char_count > MAX_LEN:
            st.caption(f"⚠️ :red[{clean_char_count}] / {MAX_LEN} (Too long by {clean_char_count - MAX_LEN})")
        else:
            st.caption(f"✅ :green[{clean_char_count}] / {MAX_LEN} characters")

with col2:
    predict_btn = st.button("Analyze for AI Patterns", use_container_width=True)

# Main Logic
if predict_btn:
    # 1. Validation check before calling API
    if clean_char_count == 0:
        st.warning("Please enter some text to analyze.")
    
    elif clean_char_count < MIN_LEN:
        st.error(f"The input is too short. Please provide at least {MIN_LEN} characters for a reliable detection.")
        
    elif clean_char_count > MAX_LEN:
        st.error(f"The input is too long. Please limit your text to {MAX_LEN} characters.")
        
    else:
        # 2. API Request
        with st.spinner("🤖 Running Semantic & Statistical Analysis..."):
            try:
                payload = {"text": text_input}
                response = requests.post(API_URL, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result.get("probability", 0)
                    
                    st.divider()
                    st.subheader("Analysis Results")
                    
                    # 3. Visual Feedback based on Probability
                    if prob >= 0.85:
                        st.error(f"**Likely AI-Generated** ({prob:.1%})")
                        st.progress(prob)
                        st.write("This text matches high-confidence patterns found in Large Language Models.")
                    
                    elif prob >= 0.50:
                        st.warning(f"**Potential AI/Human Mix** ({prob:.1%})")
                        st.progress(prob)
                        st.write("The model detected some structural consistencies typical of AI, but it is not definitive.")
                    
                    else:
                        st.success(f"**Likely Human-Written** ({prob:.1%})")
                        st.progress(prob)
                        st.write("The text displays a level of 'burstiness' and semantic variety typical of human authors.")

                elif response.status_code == 422:
                    st.error("Backend Validation Error: The server rejected the text format or length.")
                else:
                    st.error(f"Server Error: Received status code {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("📡 Could not connect to the Backend API. Ensure your FastAPI server is running on port 8080.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

st.divider()
st.caption("AITextGuard v1.0 | Powered by Sentence-Transformers & XGBoost Stack")