import streamlit as st
import requests

API_URL = "http://backend:8080/predict"

st.set_page_config(page_title="AI Cheat Text Guard", layout="centered")

st.title("🛡️ AI Cheat Text Guard")

id = st.text_input("id")
topic = st.text_area("topic")
answer = st.text_area("answer")

if st.button("Predict"):
    if not topic or not id or not answer:
        st.warning("Please fill all fields")
    else:
        payload = {"id": id, "topic": topic, "answer": answer}

        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prob = response.json()["probability"]
            st.success(f"Cheating probability: {prob:.2%}")
        else:
            st.error("Prediction failed")
