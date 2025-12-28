import streamlit as st
import requests
import os

API_URL = os.getenv("BACKEND_URL")

st.set_page_config(page_title="AI Cheat Text Guard", layout="centered")

st.title("🛡️ AI Cheat Text Guard")

topic = st.text_area("topic")
text = st.text_area("text")

if st.button("Predict"):
    if not topic or not id or not text:
        st.warning("Please fill all fields")
    else:
        payload = {"topic": topic, "answer": text}

        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prob = response.json()["probability"]
            st.success(f"Cheating probability: {prob:.2%}")
        else:
            st.error("Prediction failed")
