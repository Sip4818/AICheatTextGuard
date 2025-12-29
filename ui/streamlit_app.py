import streamlit as st
import requests
import os

# API_URL = os.getenv("BACKEND_URL")
API_URL = 'http://127.0.0.1:8080/predict'

st.set_page_config(page_title="AI Cheat Text Guard", layout="centered")

st.title("🛡️ AI Cheat Text Guard")

text = st.text_area("text")

if st.button("Predict"):
    if not text:
        st.warning("Please fill all fields")
    else:
        payload = {"text": text}

        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prob = response.json()["probability"]
            st.success(f"Cheating probability: {prob:.2%}")
        else:
            st.error("Prediction failed")
