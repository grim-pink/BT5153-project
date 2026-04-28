# 🚀 Spam + Social Intent Detection System (Group 15)

A production-inspired, end-to-end machine learning pipeline for SMS analysis, combining deep learning classification with LLM-powered intent understanding.
Data and model artifacts are versioned using DVC and are not stored directly in the repository.

# 🔥 What This Project Does

We built a two-stage intelligent detection system:

🧠 Task 1 — Spam Detection (ML Model)
- Fine-tuned DistilBERT
- Classifies messages into:
  - spam
  - ham

💬 Task 2 — Social Intent Classification (LLM)
- Triggered only if message is spam
- Uses a locally hosted LLM via Ollama
- Uses prompting (zero-shot / few-shot)
- Extracts intent such as:
  - Financial / Reward Deception
  - Impersonation / Credential Theft
  - Adult / Solicitation
  - Benign

# 🏗️ System Architecture

🔹 Training Pipeline
```bash
Raw Data
   ↓
Data Cleaning
   ↓
Train/Test Split
   ↓
DistilBERT Training
   ↓
Model Artifacts (Saved)
```

🔹 Inference Pipeline (Real-time API)
```bash
User Input
   ↓
Text Preprocessing
   ↓
DistilBERT Spam Classifier
   ↓
IF spam → LLM Intent Classifier
   ↓
Structured API Response
```

```bash
💡 Key design principle: Use ML for fast classification, LLM for semantic understanding
```
  
## 🧩 Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Spam Model | DistilBERT |
| LLM Intent Model | Ollama |
| ML Framework | HuggingFace Transformers |
| Data Processing | Pandas |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Deployment | Docker |
| Monitoring | Prometheus |
| Logging | Audit logs (inference tracking) |

## 📈 Monitoring & Logging

- Prometheus metrics track:
  - prediction distribution
  - inference latency
  - intent classification frequency
- Audit logs store:
  - predictions and confidence
  - intent outputs
  - model version and latency

## 📁 Project Structure

```text
src/
  data/              Data preparation scripts
  preprocessing/     Text cleaning logic
  training/          Model training scripts
  inference/         Model loading and inference pipeline
  utils/             Config and helper functions

api/
  main.py            FastAPI backend
  static/            Frontend HTML UI

data/
  raw/               Raw input files
  processed/         Cleaned and split datasets

artifacts/
  distilbert_spam_model/   Saved DistilBERT model
  baseline/                Baseline model artifacts
  metrics/                 Evaluation metrics
```

## ⚙️ Setup Guide

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare raw data

Place the following files in `data/raw/`:

```text
spam.csv
dataset.csv
validation_intent.xlsx
```

### 3. Build processed dataset

```bash
PYTHONPATH=. python -m src.data.make_dataset \
  --spam_csv data/raw/spam.csv \
  --dataset_csv data/raw/dataset.csv \
  --output_path data/processed/cleaned_spam.csv
```

### 4. Split train/test data

```bash
PYTHONPATH=. python -m src.data.split_dataset \
  --input_path data/processed/cleaned_spam.csv \
  --train_output data/processed/train.csv \
  --test_output data/processed/test.csv
```

### 5. Train DistilBERT spam classifier

```bash
PYTHONPATH=. python -m src.training.train_distilbert \
  --train_path data/processed/train.csv \
  --test_path data/processed/test.csv \
  --model_output_dir artifacts/distilbert_spam_model \
  --metrics_output artifacts/metrics/distilbert_metrics.json
```

### 6. Run FastAPI app

```bash
uvicorn api.main:app --reload --port 8000
```

Open the web UI:

```text
http://127.0.0.1:8000/
```

## 🤖 Ollama Setup for Task 2

The intent classification module assumes Ollama is running locally or externally.

```bash
ollama serve
ollama pull qwen2.5:7b
```
### To evaluate intent classification

```bash
PYTHONPATH=. python src/intent/evaluate_intent.py \
  --input_path data/raw/validation_intent.xlsx \
  --output_dir artifacts/metrics
```

You can change the LLM model using:

```bash
OLLAMA_MODEL=qwen2.5:7b
```

## 🐳 Docker

```bash
docker build -t spam-intent-api .
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen2.5:7b \
  spam-intent-api
```

## 🧪 Example API Output

```json
{
  "text": "Claim your free prize now!",
  "cleaned_text": "claim your free prize now",
  "prediction": "spam",
  "confidence": 0.94,
  "intent": "promotion",
  "model_version": "distilbert_v1"
}
```
## 🔁 Reproducibility

To reproduce the pipeline:

```bash
dvc pull
dvc repro
```
## 📊 Key Design Decisions

- Two-stage design keeps the system efficient and interpretable.
- DistilBERT handles fast spam/ham classification.
- The LLM is only called for spam messages to reduce cost and latency.
- Training and inference code are separated for cleaner deployment.
- The FastAPI layer makes the model accessible through both API and web UI.

## 🎯 Why This Project Matters

This project demonstrates how machine learning can be moved beyond notebooks into a working application. It combines data processing, model training, artifact saving, API deployment, and LLM integration into one end-to-end pipeline.

## 💡 Future Improvements

- Improve performance on subtle intent categories (e.g., impersonation)
- Expand labeled dataset for intent classification
- Explore fine-tuning LLMs instead of prompting
- Deploy system to cloud for real-time scalability


