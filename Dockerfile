FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY src ./src
COPY artifacts/distilbert_spam_model ./artifacts/distilbert_spam_model

EXPOSE 8000

ARG MODEL_VERSION="v1.0"
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MODEL_PATH=./artifacts/distilbert_spam_model
ENV MAX_LENGTH=64
ENV OLLAMA_MODEL=qwen2.5:7b
ENV OLLAMA_TEMPERATURE=0.0

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
