# AITextGuard  
**AI-generated Text Detection using Machine Learning and MLOps**

AITextGuard is an end-to-end NLP system designed to detect AI-generated text using machine learning.
The project demonstrates how a production-style ML pipeline can be built, trained, evaluated, and deployed
using modern MLOps practices.

A live demo is available, and the entire system can be run locally using Docker Compose with a single command.

---

## 🚀 Live Demo & Source Code

- **Hugging Face Space:**  
  https://huggingface.co/spaces/Sahil4818/ai-text-guard

- **GitHub Repository:**  
  https://github.com/Sip4818/AICheatTextGuard

---

## 📌 Problem Statement

With the increasing use of large language models, distinguishing between human-written and AI-generated text
has become important for content moderation, education, and information integrity.
This project explores a machine learning–based approach to identify AI-generated text
using a combination of traditional ML models and transformer-based embeddings.

---

## 🧠 Solution Overview

AITextGuard implements a modular and reproducible machine learning workflow that covers:

- Data ingestion from cloud storage
- Schema-based data validation
- Feature engineering using text statistics and transformer embeddings
- Model training with stacked ensemble learning
- Model evaluation and conditional promotion
- API-based inference and deployment

The system is designed to be reusable, reproducible, and easy to deploy.

---

## 🏗️ System Architecture (High Level)

### Training Pipeline
1. Data ingestion from Google Cloud Storage (GCS)
2. Train–test splitting and schema validation
3. Feature engineering:
   - Statistical text features
   - Transformer-based sentence embeddings
4. Model training:
   - Level-1 models (Logistic Regression, XGBoost)
   - Meta model for stacking
5. Hyperparameter tuning and experiment tracking
6. Model evaluation
7. Approved model stored in cloud storage

### Inference Pipeline
1. Input text received via FastAPI service
2. Same feature transformation pipeline applied
3. Trained model loaded from storage
4. Prediction returned through REST API

---

## ✨ Key Features

- End-to-end machine learning pipeline
- Schema-based data validation
- NLP feature engineering with transformer embeddings
- Stacked ensemble learning
- Hyperparameter tuning and experiment tracking
- Reproducible training with DVC
- Dockerized deployment with CI/CD
- One-command local setup using Docker Compose
- Live deployment on Hugging Face Spaces

---

## 🛠️ Tech Stack

**Programming Languages**
- Python

**Machine Learning & NLP**
- Scikit-learn
- XGBoost
- Sentence Transformers

**Data & MLOps**
- Pandas, NumPy
- Optuna
- MLflow
- DVC

**Deployment**
- FastAPI
- Docker
- Docker Compose
- GitHub Actions (CI/CD)

**Cloud**
- Google Cloud Storage (GCS)
- Hugging Face Spaces

---

## 📊 Model Approach (High Level)

- **Features**
  - Statistical text features (length, punctuation, etc.)
  - Transformer-based sentence embeddings

- **Models**
  - Level-1: Logistic Regression, XGBoost
  - Level-2: Meta Logistic Regression (stacking)

- **Evaluation Metric**
  - ROC-AUC

---

## 🔁 Reproducible Training

- Training pipelines are versioned using **DVC**
- Experiments and hyperparameter tuning are tracked using **MLflow**
- Artifacts (models, pipelines) are stored and versioned for reuse

---

## ▶️ Run Locally (Docker Compose)

### Requirements
- **Docker**
- **Docker Compose**

Make sure Docker is installed and running on your system.

### Steps

```bash
git clone https://github.com/Sip4818/AICheatTextGuard.git
cd AICheatTextGuard
docker compose up
