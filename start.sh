#!/bin/bash

# Start the Backend file (FastAPI)
# We use 'backend.app' because it's inside the backend folder
uvicorn app:app --host 0.0.0.0 --port 8080 &


# Start the UI file (Streamlit)
streamlit run ui/streamlit_app.py --server.port 7860 --server.address 0.0.0.0