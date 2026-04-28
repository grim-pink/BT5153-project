from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.inference.load_spam_model import (
    get_device,
    get_model,
    get_tokenizer,
    load_model,
    load_tokenizer,
)
from src.intent.health import ollama_healthcheck
from src.inference.pipeline import full_pipeline_batch, full_pipeline_single
from src.utils.config import settings

app = FastAPI(
    title="Spam + Intent Classification Service",
    description="Two-stage SMS inference: Task 1 spam detection, Task 2 intent classification",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

Instrumentator().instrument(app).expose(app)

PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total count of predictions made by the model",
    ["predicted_class", "model_version"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Total request latency in seconds",
)

SPAM_LATENCY = Histogram(
    "spam_inference_latency_seconds",
    "Spam classifier latency in seconds",
)

INTENT_LATENCY = Histogram(
    "intent_inference_latency_seconds",
    "Intent classifier latency in seconds",
)

INTENT_COUNTER = Counter(
    "intent_predictions_total",
    "Total count of intent predictions",
    ["intent_label", "model_version"],
)

INTENT_FAILURE_COUNTER = Counter(
    "intent_failures_total",
    "Total count of Task 2 failures",
    ["model_version"],
)


class SingleRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


@app.on_event("startup")
async def startup_event() -> None:
    try:
        load_tokenizer()
        load_model()
        print(f"Model version: {settings.model_version}")
        print(f"Model path: {settings.model_path}")
        print(f"Using device: {get_device()}")
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Startup failed: {e}")
        raise

@app.get("/")
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")
    
@app.get("/health")
async def health() -> dict:
    model_loaded = get_model() is not None
    tokenizer_loaded = get_tokenizer() is not None
    ollama_ok = ollama_healthcheck()

    return {
        "status": "ok" if model_loaded and tokenizer_loaded else "degraded",
        "model_loaded": model_loaded,
        "tokenizer_loaded": tokenizer_loaded,
        "ollama_available": ollama_ok,
        "model_version": settings.model_version,
    }


@app.post("/predict")
async def predict_single(request: SingleRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        result = full_pipeline_single(request.text)

        PREDICTION_COUNTER.labels(
            predicted_class=result["prediction"],
            model_version=settings.model_version,
        ).inc()

        REQUEST_LATENCY.observe(result["total_latency_ms"] / 1000)
        SPAM_LATENCY.observe(result["spam_latency_ms"] / 1000)

        if result["intent"] is not None:
            INTENT_LATENCY.observe(result["intent_latency_ms"] / 1000)
            INTENT_COUNTER.labels(
                intent_label=result["intent"],
                model_version=settings.model_version,
            ).inc()

        if result.get("intent_error"):
            INTENT_FAILURE_COUNTER.labels(
                model_version=settings.model_version,
            ).inc()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(request: BatchRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Input texts list cannot be empty.")

    if any((not t or not t.strip()) for t in request.texts):
        raise HTTPException(status_code=400, detail="All texts must be non-empty strings.")

    try:
        results = full_pipeline_batch(request.texts)

        for item in results:
            PREDICTION_COUNTER.labels(
                predicted_class=item["prediction"],
                model_version=settings.model_version,
            ).inc()
            REQUEST_LATENCY.observe(item["total_latency_ms"] / 1000)
            SPAM_LATENCY.observe(item["spam_latency_ms"] / 1000)

            if item["intent"] is not None:
                INTENT_LATENCY.observe(item["intent_latency_ms"] / 1000)
                INTENT_COUNTER.labels(
                    intent_label=item["intent"],
                    model_version=settings.model_version,
                ).inc()

            if item.get("intent_error"):
                INTENT_FAILURE_COUNTER.labels(
                    model_version=settings.model_version,
                ).inc()

        return {
            "batch_size": len(results),
            "results": results,
            "model_version": settings.model_version,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")