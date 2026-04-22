# BT5153 Spam + Intent Detection Pipeline

End-to-end two-stage SMS detection system:

1. **Task 1**: Binary spam/ham detection using a fine-tuned DistilBERT model.
2. **Task 2**: Intent classification for messages predicted as spam using an external Ollama-served LLM.

## Architecture

Raw data -> cleaning -> train/test split -> DistilBERT training -> saved model artifacts

Inference:
Input text -> clean text -> DistilBERT spam classifier -> if spam -> Ollama intent classifier -> API response

## Project structure

See the folder tree in this repository.

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional: initialize DVC

```bash
dvc init
```

### 3. Put raw files in `data/raw/`

Expected files:
- `spam.csv`
- `dataset.csv`
- `validation_intent.xlsx` 

### 4. Build processed dataset

```bash
python -m src.data.make_dataset \
  --spam_csv data/raw/spam.csv \
  --dataset_csv data/raw/dataset.csv \
  --output_path data/processed/cleaned_spam.csv
```

### 5. Split train/test

```bash
python -m src.data.split_dataset \
  --input_path data/processed/cleaned_spam.csv \
  --train_output data/processed/train.csv \
  --test_output data/processed/test.csv
```

### 6. Train DistilBERT

```bash
python -m src.training.train_distilbert \
  --train_path data/processed/train.csv \
  --test_path data/processed/test.csv \
  --model_output_dir artifacts/distilbert_spam_model \
  --metrics_output artifacts/metrics/distilbert_metrics.json
```

### 7. Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Task 2 / Ollama

The intent module assumes an **external Ollama service** is already running and has the target model pulled.

Example:

```bash
ollama serve
ollama pull qwen2.5:7b
```

You can change the model with the `OLLAMA_MODEL` environment variable.

## Docker

```bash
docker build -t bt5153-spam-intent .
docker run -p 8000:8000 bt5153-spam-intent
```

If using Task 2 inside Docker, the container must be able to reach the Ollama host.

## Notes

- Task 1 uses `ham=0`, `spam=1`.
- Task 2 is only called when Task 1 predicts spam.
- This project keeps training and inference code separate.
