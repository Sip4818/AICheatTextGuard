# AITextGuard  
**AI-Generated Text Detection using Machine Learning & Production-Ready MLOps**

AITextGuard is an end-to-end NLP system designed to detect AI-generated text using a stacked machine learning approach.  
The project demonstrates how to design, train, evaluate, containerize, monitor, and deploy a production-style ML system using modern MLOps practices.

The entire system can be launched locally with a single command using Docker Compose.

---

## 🚀 Live Demo & Source Code

- **Hugging Face Space:**  
  https://huggingface.co/spaces/Sahil4818/ai-text-guard

- **GitHub Repository:**  
  https://github.com/Sip4818/AICheatTextGuard

---

## 📌 Problem Statement

With the increasing use of large language models, distinguishing between human-written and AI-generated text has become critical for:

- Academic integrity  
- Content moderation  
- Plagiarism detection  
- Information reliability  

AITextGuard explores a machine learning–based approach combining statistical features and transformer embeddings to classify AI-generated text.

---

# 🏗️ System Architecture

The system runs as a multi-service Docker setup:

User → Streamlit UI → FastAPI Backend → Redis (Cache)  
                      ↓  
                    Prometheus (Monitoring)

---

## 🔄 Training Pipeline

1. Data ingestion from Google Cloud Storage (GCS)  
2. Schema validation  
3. Train–test split  
4. Feature engineering:
   - Statistical text features  
   - Transformer-based sentence embeddings  
5. Model training:
   - Level-1: Logistic Regression, XGBoost  
   - Level-2: Meta Logistic Regression (Stacking)  
6. Hyperparameter tuning (Optuna)  
7. Experiment tracking (MLflow)  
8. Model evaluation (ROC-AUC)  
9. Approved model stored in cloud storage  

Training is reproducible using DVC.

---

## ⚡ Inference Pipeline

1. Text submitted via Streamlit UI  
2. FastAPI backend processes request  
3. Feature transformation applied  
4. Stacked model predicts probability  
5. Result returned via REST API  
6. Prediction cached in Redis (10-minute TTL)  
7. Prometheus collects performance metrics  

---

# ✨ Key Features

- End-to-end ML pipeline  
- Stacked ensemble learning  
- Transformer-based embeddings  
- Reproducible training with DVC  
- Experiment tracking with MLflow  
- FastAPI inference API  
- Redis caching  
- Prometheus monitoring  
- Fully containerized architecture  
- CI/CD with GitHub Actions  
- Automated Docker image builds with model injection  

---

# 🛠️ Tech Stack

**Programming**
- Python  

**Machine Learning & NLP**
- Scikit-learn  
- XGBoost  
- Sentence Transformers  

**Data & MLOps**
- Pandas  
- NumPy  
- Optuna  
- MLflow  
- DVC  

**Backend & API**
- FastAPI  

**Monitoring**
- Prometheus  

**Caching**
- Redis  

**Deployment & DevOps**
- Docker  
- Docker Compose  
- GitHub Actions  
- Docker Hub  

**Cloud**
- Google Cloud Storage (Model artifacts)  
- Hugging Face Spaces (Live demo)  

---

# 🧠 Model Overview

**Features**
- Text statistics  
- Punctuation distribution  
- Sentence embeddings  

**Architecture**
- Level-1: Logistic Regression + XGBoost  
- Level-2: Meta Logistic Regression  

**Evaluation Metric**
- ROC-AUC  

---

# 🔄 CI/CD Pipeline

On every push to `main`:

1. Authenticate with Google Cloud  
2. Download trained model from GCS  
3. Build backend image (model included)  
4. Build UI image  
5. Push images to Docker Hub  

---

# ▶️ Run Locally

### Requirements
- Docker  
- Docker Compose  

### Start Everything

```bash
git clone https://github.com/Sip4818/AICheatTextGuard.git
cd AICheatTextGuard
docker compose up
